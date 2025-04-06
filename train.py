"""
Main training script for brain tumor segmentation using TransUNet.

This script initializes the model, loads the dataset from a local file path,
trains the model, performs validation using standard and BraTS metrics,
visualizes results, and saves checkpoints locally.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt # Import for plotting history and visualization
import numpy as np # Often needed alongside torch and for visualization indexing

# Update imports to reflect the modular project structure
from config.config import get_config
# Ensure BrainMRIDataset is designed for local file loading
from data.brain_mri_dataset import BrainMRIDataset
from models.model import TransUNet
from losses.losses import TverskyFocalLoss, CompositeLoss
# Import all necessary utility functions
from utils.utils import (
    calculate_metrics,
    compute_class_weights,
    evaluate_brats_regions, # Renamed from evaluate_with_torchmetrics
    visualize_predictions
)

def train(train_loader, val_loader, cfg):
    """Train the TransUNet model.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        cfg (dict): Configuration dictionary.

    Returns:
        tuple: Trained model (best weights loaded) and training history dictionary.
    """
    device = torch.device(cfg['device'])
    print(f"Using device: {device} for training.")

    # Initialize the model and move to the configured device
    model = TransUNet(
        img_size=cfg['img_size'],
        in_channels=cfg['in_channels'],
        num_classes=cfg['num_classes'],
        backbone_name=cfg['model']['backbone_name'],
        vit_name=cfg['model']['vit_name'],
        pretrained=cfg['model']['pretrained']
    ).to(device)
    print("Model initialized.")

    # Initialize loss functions
    print(f"Initializing Loss Function: {cfg['loss']['type']}")
    # --- Choose Loss Function (Copied logic from previous merge) ---
    class_weights = cfg.get('class_weights', None) # Get weights from cfg if available
    if cfg['loss']['type'] == 'CompositeLoss':
         tversky_focal_loss_fn = TverskyFocalLoss(
             alpha=cfg['loss']['params']['tversky_alpha'],
             beta=cfg['loss']['params']['tversky_beta'],
             gamma=cfg['loss']['params']['tversky_gamma'],
             focal_alpha=cfg['loss']['params']['tversky_focal_alpha'],
             class_weights=class_weights, # Pass computed weights
             num_classes=cfg['num_classes']
         ).to(device)
         criterion = CompositeLoss(
             region_loss_fn=tversky_focal_loss_fn,
             boundary_loss_weight=cfg['loss']['params']['boundary_loss_weight'],
             num_classes=cfg['num_classes']
         ).to(device)
    elif cfg['loss']['type'] == 'TverskyFocalLoss':
          criterion = TverskyFocalLoss(
             alpha=cfg['loss']['params']['tversky_alpha'],
             beta=cfg['loss']['params']['tversky_beta'],
             gamma=cfg['loss']['params']['tversky_gamma'],
             focal_alpha=cfg['loss']['params']['tversky_focal_alpha'],
             class_weights=class_weights, # Pass computed weights
             num_classes=cfg['num_classes']
         ).to(device)
    else:
         raise ValueError(f"Unsupported loss type: {cfg['loss']['type']}")
    print("Loss function initialized.")

    # Setup the optimizer
    print(f"Initializing Optimizer: {cfg['optimizer']}")
    if cfg['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
        )
    # Add other optimizers if needed (e.g., Adam)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg['optimizer']}")
    print("Optimizer initialized.")

    # Setup learning rate schedulers
    print("Initializing Schedulers...")
    # Adjust T_max for cosine scheduler based on steps per epoch
    steps_per_epoch = len(train_loader) // cfg['grad_accum_steps']
    cosine_t_max = cfg['scheduler']['cosine']['T_max'] * steps_per_epoch
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_t_max if cosine_t_max > 0 else steps_per_epoch, # Ensure T_max is positive
        eta_min=cfg['scheduler']['cosine']['eta_min']
    )
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg['scheduler']['plateau']['mode'],
        factor=cfg['scheduler']['plateau']['factor'],
        patience=cfg['scheduler']['plateau']['patience'],
        threshold=cfg['scheduler']['plateau']['threshold'],
        cooldown=cfg['scheduler']['plateau']['cooldown'],
        verbose=True # Print message on LR reduction
    )
    print("Schedulers initialized.")

    best_dice = 0.0
    epochs_no_improve = 0
    use_amp = (cfg['precision'] == 'fp16')
    scaler = GradScaler(enabled=use_amp)
    print(f"Automatic Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")


    # History dictionary to store training metrics
    history = {
        'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': [], 'lr': []
    }

    # Checkpoint directory and best model path
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    best_model_path = os.path.join(cfg['checkpoint_dir'], cfg['best_model_name']) # Name from config


    print("\n--- Starting Training ---")
    for epoch in range(cfg['epochs']):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        optimizer.zero_grad() # Clear gradients at start

        # Wrap train_loader with tqdm for progress bar
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]")

        # Iterate over training batches
        for batch_idx, (x, y) in train_pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                pred = model(x)
                loss = criterion(pred, y)
                # Scale loss for gradient accumulation
                loss = loss / cfg['grad_accum_steps']

            # Backpropagation with gradient accumulation
            scaler.scale(loss).backward()
            # Accumulate loss for logging (use unscaled loss)
            epoch_loss += loss.item() * cfg['grad_accum_steps']

            # Optimizer step after accumulation steps
            if (batch_idx + 1) % cfg['grad_accum_steps'] == 0:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                # Clip gradients if enabled
                if cfg['grad_clip_value'] > 0:
                     torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip_value'])
                # Optimizer step
                scaler.step(optimizer)
                # Update scaler
                scaler.update()
                # Zero gradients for next cycle
                optimizer.zero_grad()
                # Step cosine scheduler (adjust based on effective steps)
                if cfg['scheduler']['type'] in ['hybrid', 'cosine']:
                    scheduler_cosine.step()

            # Update progress bar description
            train_pbar.set_postfix(loss=f"{loss.item()*cfg['grad_accum_steps']:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # --- End of Epoch ---
        avg_train_loss = epoch_loss / len(train_loader) # Average loss over batches

        # --- Validation ---
        if (epoch + 1) % cfg['validation_interval'] == 0:
            # Use the simple evaluate function (calculates foreground Dice/IoU)
            val_loss, avg_dice, avg_iou = evaluate(model, val_loader, criterion, device, cfg['num_classes'], cfg['precision'])

            print(f"\nEpoch {epoch+1} Summary: \n"
                  f"  Avg Train Loss: {avg_train_loss:.4f}\n"
                  f"  Validation Loss: {val_loss:.4f}\n"
                  f"  Validation Dice (FG): {avg_dice:.4f}\n" # Clarify it's foreground avg
                  f"  Validation IoU (FG):  {avg_iou:.4f}")  # Clarify it's foreground avg

            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_dice'].append(avg_dice) # Store foreground Dice from simple eval
            history['val_iou'].append(avg_iou)   # Store foreground IoU from simple eval
            history['lr'].append(optimizer.param_groups[0]['lr'])

            # --- Checkpointing & Early Stopping ---
            current_val_metric = avg_dice # Use foreground Dice for decisions
            is_best = current_val_metric > (best_dice + cfg['early_stopping']['threshold'])

            if is_best:
                print(f"  Validation Dice (FG) improved ({best_dice:.4f} --> {current_val_metric:.4f}). Saving best model...")
                best_dice = current_val_metric
                epochs_no_improve = 0
                # Save the *best* model state
                torch.save(model.state_dict(), best_model_path)
                print(f"  Best model saved to {best_model_path}")
            else:
                epochs_no_improve += 1
                print(f"  Validation Dice (FG) did not improve significantly. ({epochs_no_improve}/{cfg['early_stopping']['patience']})")

            # Save checkpoint every epoch if enabled
            if cfg['save_every_epoch']:
                 epoch_save_path = os.path.join(cfg['checkpoint_dir'], f"TransUnet_epoch_{epoch+1}.pth")
                 torch.save(model.state_dict(), epoch_save_path)
                 # print(f"  Epoch checkpoint saved to {epoch_save_path}") # Optional print

            # --- LR Scheduling (Plateau) ---
            if cfg['scheduler']['type'] in ['hybrid', 'plateau']:
                 scheduler_plateau.step(current_val_metric) # Step based on foreground Dice

            # --- Early Stopping Check ---
            if cfg['early_stopping']['enabled'] and epochs_no_improve >= cfg['early_stopping']['patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs due to no improvement in validation Dice (FG).")
                break # Exit training loop
        else:
             # Handle epochs where validation is skipped
             history['train_loss'].append(avg_train_loss)
             history['val_loss'].append(np.nan)
             history['val_dice'].append(np.nan)
             history['val_iou'].append(np.nan)
             history['lr'].append(optimizer.param_groups[0]['lr'])
             print(f"\nEpoch {epoch+1} Summary: \n"
                   f"  Avg Train Loss: {avg_train_loss:.4f}\n"
                   f"  (Validation skipped this epoch)")

    print("\n--- Training Finished ---")
    # Load the best model weights found during training for final evaluation
    if os.path.exists(best_model_path):
         print(f"Loading best model weights from {best_model_path}")
         try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("Best model weights loaded successfully.")
         except Exception as e:
             print(f"Warning: Failed to load best model weights from {best_model_path}. Using last epoch weights. Error: {e}")
    else:
        print("Warning: Best model checkpoint not found. Using model weights from the last epoch.")

    return model, history


def evaluate(model, loader, criterion, device, num_classes, precision='fp32'):
    """
    Evaluate the model using the simple per-class foreground Dice/IoU.
    (Modified slightly from original to accept num_classes and precision)
    """
    model.eval()
    total_loss = 0.0
    dice_total, iou_total = 0.0, 0.0
    num_batches = len(loader)

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation (FG Metrics)", leave=False):
            x, y = x.to(device), y.to(device)
            # Use autocast for consistency
            with autocast(enabled=(precision == 'fp16')):
                pred = model(x)
                # Calculate loss if criterion is provided (optional, could remove loss calc here)
                if criterion:
                    loss = criterion(pred, y)
                    total_loss += loss.item()
                else:
                    total_loss = 0 # Or handle appropriately

            # Use the utility function for foreground Dice/IoU
            dice, iou = calculate_metrics(pred.detach(), y, num_classes=num_classes)
            dice_total += dice # calculate_metrics returns float now
            iou_total += iou   # calculate_metrics returns float now

    avg_loss = total_loss / num_batches if criterion else 0.0
    avg_dice = dice_total / num_batches
    avg_iou = iou_total / num_batches

    return avg_loss, avg_dice, avg_iou


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Brain Tumor Segmentation Training Script (Local) ---")
    # Load configuration settings
    cfg = get_config()
    device = torch.device(cfg['device']) # Define device early

    # --- Dataset Setup (Local Path) ---
    # Set the local data path for the BraTS dataset
    # IMPORTANT: Make sure this path points to the root directory containing individual patient folders
    data_root = "./data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    print(f"Loading dataset from local path: {data_root}")

    # Create the dataset instance; ensure that BrainMRIDataset uses local file logic
    try:
        dataset = BrainMRIDataset(
            data_root=data_root,
            # Adjust pattern if your local folders have a different naming scheme
            pattern="BraTS20_Training_*", # Example pattern to find patient folders
            img_size=cfg['img_size']
        )
        if len(dataset) == 0:
            raise ValueError("Dataset is empty. Check data_root and pattern.")
        print(f"Dataset loaded successfully. Found {len(dataset)} slices.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load dataset from {data_root}. Error: {e}")
        exit() # Exit if dataset loading fails

    # Split the dataset into training and validation sets (e.g., 90/10 split)
    train_size = int(cfg['train_val_split'] * len(dataset))
    val_size = len(dataset) - train_size
    print(f"Splitting dataset: Train={train_size}, Validation={val_size}")
    # Ensure consistent splits if needed by setting torch.manual_seed() before random_split
    # torch.manual_seed(42) # Example for reproducible splits
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Compute class weights from the training set if enabled
    if cfg['loss']['class_weights']['enabled']:
        cfg['class_weights'] = compute_class_weights(
            train_dataset,
            num_classes=cfg['num_classes'],
            border_boost=cfg['loss']['class_weights']['border_boost'],
            batch_size=cfg['batch_size'] * 2,
            num_workers=cfg['num_workers']
            ).to(device) # Move weights to target device
        print(f"Computed Class Weights: {cfg['class_weights'].cpu().numpy()}")
    else:
        cfg['class_weights'] = None # Ensure it's None if disabled

    # Create DataLoaders for training and validation
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        persistent_workers=(cfg['num_workers'] > 0),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'] * 2, # Can use larger batch for validation
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True
    )
    print("DataLoaders created.")

    # --- Begin Training ---
    # Pass DataLoaders and config to the train function
    trained_model, training_history = train(train_loader, val_loader, cfg)

    # --- Post-Training Actions ---
    if trained_model:
        print("\n--- Training script completed successfully ---")

        # --- Perform Final Evaluation with BraTS Metrics ---
        # Uses the 'trained_model' which should have the best weights loaded by train()
        print("\n--- Performing Final Evaluation on Validation Set (BraTS Metrics) ---")
        # The val_loader and val_dataset defined earlier are accessible here
        brats_metrics = evaluate_brats_regions(
             trained_model,
             val_loader, # Use the same val_loader
             device,
             num_classes=cfg['num_classes'],
             precision=cfg['precision']
        )
        print("\n--- Final Validation Metrics (BraTS Regions) ---")
        print(f"  Dice ET: {brats_metrics.get('dice_et', 0.0):.4f}")
        print(f"  Dice TC: {brats_metrics.get('dice_tc', 0.0):.4f}")
        print(f"  Dice WT: {brats_metrics.get('dice_wt', 0.0):.4f}")
        print(f"  IoU  ET: {brats_metrics.get('iou_et', 0.0):.4f}")
        print(f"  IoU  TC: {brats_metrics.get('iou_tc', 0.0):.4f}")
        print(f"  IoU  WT: {brats_metrics.get('iou_wt', 0.0):.4f}")
        print(f"  ------------------------------------")
        print(f"  Average Dice (ET, TC, WT): {brats_metrics.get('avg_dice', 0.0):.4f}")
        print(f"  Average IoU  (ET, TC, WT): {brats_metrics.get('avg_iou', 0.0):.4f}")
        print("-" * 40)

        # --- Visualize Predictions ---
        print("\n--- Visualizing Sample Predictions ---")
        try:
             visualize_predictions(
                 trained_model,
                 val_dataset, # Use the validation dataset split
                 device,
                 num_samples=5, # Show 5 samples
                 num_classes=cfg['num_classes']
             )
        except Exception as e:
            print(f"Error during visualization: {e}")


        # --- Plotting History ---
        print("\n--- Generating Training History Plots ---")
        try:
            # Ensure matplotlib is imported
            plt.figure(figsize=(12, 5))

            # Plot Losses
            plt.subplot(1, 2, 1)
            plt.plot(training_history['train_loss'], label='Avg Train Loss per Batch')
            plt.plot(training_history['val_loss'], label='Validation Loss')
            plt.title('Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            # Plot Foreground Dice/IoU collected during training validation steps
            plt.subplot(1, 2, 2)
            plt.plot(training_history['val_dice'], label='Validation Dice (FG Avg)')
            plt.plot(training_history['val_iou'], label='Validation IoU (FG Avg)')
            plt.title('Validation Metrics (FG Avg) Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plot_filename = "training_plots.png"
            plt.savefig(plot_filename)
            print(f"Training plots saved to {plot_filename}")
            # Optionally uncomment to display plot interactively:
            # plt.show()
        except ImportError:
            print("Matplotlib not found. Skipping plot generation.")
        except Exception as e:
            print(f"Error generating plots: {e}")

    else:
        print("\n--- Training script failed. Check logs for errors. ---")