# train.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast # For mixed precision
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time # To track training time

# --- Local Project Imports ---
# Assume these modules exist in your project structure:
# config/config.py -> defines get_config()
# data/dataset.py -> defines BrainMRIDataset_Local (adapted for local files)
# models/transunet.py -> defines TransUNet
# losses/compound_losses.py -> defines CompositeLoss, TverskyFocalLoss
# utils/utils.py -> contains helper functions like compute_class_weights,
#                    calculate_metrics, compute_brats_hd95, evaluate_with_brats_metrics etc.

from config.config import get_config # Example: Function to load configuration
from data import BrainMRIDataset_Local # Use the local dataset class
from models.model import TransUNet # Import your model definition
from losses.losses import CompositeLoss, TverskyFocalLoss # Import loss definitions
from utils.utils import (
    compute_class_weights,
    calculate_metrics, # For basic Dice/IoU during validation epoch
    compute_brats_hd95, # For HD95 calculation during validation epoch
    evaluate_with_brats_metrics, # For comprehensive final evaluation
    visualize_predictions # Optional: visualize during/after training
)

# --- Evaluation Function (called each epoch) ---

def evaluate(model: torch.nn.Module,
             loader: DataLoader,
             criterion: torch.nn.Module,
             device: torch.device,
             use_amp: bool = False) -> tuple[float, float, float]:
    """Evaluates the model on a given dataset loader for one epoch.

    Calculates average loss, Dice score, and IoU score over the dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader for the validation or test dataset.
        criterion (torch.nn.Module): The loss function used for evaluation.
        device (torch.device): The device to run evaluation on ('cuda' or 'cpu').
        use_amp (bool): Whether to use automatic mixed precision for evaluation.

    Returns:
        tuple[float, float, float]: A tuple containing:
            - Average loss over the dataset.
            - Average Dice score over the dataset (foreground classes).
            - Average IoU score over the dataset (foreground classes).
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    dice_total = 0.0
    iou_total = 0.0
    num_batches = len(loader)

    with torch.no_grad(): # Disable gradient calculations
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x, y = x.to(device), y.to(device)

            # Use autocast context if AMP is enabled
            with autocast(enabled=use_amp):
                pred = model(x)
                loss = criterion(pred, y)

            total_loss += loss.item()
            # Detach predictions before calculating metrics if they require grads
            dice, iou = calculate_metrics(pred.detach(), y)
            dice_total += dice.item()
            iou_total += iou.item()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_dice = dice_total / num_batches if num_batches > 0 else 0.0
    avg_iou = iou_total / num_batches if num_batches > 0 else 0.0

    return avg_loss, avg_dice, avg_iou


# --- Training Function ---

def train_model(train_loader: DataLoader,
                val_loader: DataLoader,
                cfg: dict) -> tuple[torch.nn.Module, dict]:
    """Trains the TransUNet model using the provided configuration and data.

    Implements a training loop with gradient accumulation, mixed precision,
    learning rate scheduling, validation, early stopping, and best model saving.
    Also logs metrics and plots training history.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        cfg (dict): Configuration dictionary containing hyperparameters, paths,
                    device settings, class weights, etc. Expected keys include:
                    'device', 'model_save_path', 'log_dir', 'epochs', 'lr',
                    'weight_decay', 'precision', 'grad_accum', 'grad_clip',
                    'class_weights', 'loss_params', 'scheduler', 'early_stop',
                    'img_size', 'spacing'.

    Returns:
        tuple[torch.nn.Module, dict]: A tuple containing:
            - The model loaded with the best weights found during training.
            - A dictionary containing the training history (losses, metrics, lr).
    """
    start_time = time.time()
    device = cfg['device']
    use_amp = (cfg['precision'] == 'fp16') and (device.type == 'cuda')

    # --- Initialization ---
    print("Initializing model, optimizer, loss, and scaler...")
    # Model
    model = TransUNet(img_dim=cfg['img_size'], # Pass necessary model args from cfg
                      in_channels=4,           # Assuming 4 modalities
                      out_channels=4,          # Assuming 4 output classes (incl background)
                      head_num=cfg.get('model_head_num', 12), # Example model param
                      mlp_dim=cfg.get('model_mlp_dim', 3072), # Example model param
                      block_num=cfg.get('model_block_num', 12) # Example model param
                     ).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(),
                              lr=cfg['lr'],
                              weight_decay=cfg['weight_decay'])

    # Loss Function (ensure class weights are on the correct device)
    class_weights = cfg['class_weights'].to(device)
    criterion = CompositeLoss(
        TverskyFocalLoss(
            class_weights=class_weights,
            alpha=cfg['loss_params']['tversky_alpha'],
            beta=cfg['loss_params']['tversky_beta'],
            gamma=cfg['loss_params']['tversky_gamma'],
            focal_alpha=cfg['loss_params']['focal_alpha']
        ),
        boundary_loss_weight=cfg['loss_params']['boundary_loss_weight']
    ).to(device)

    # Learning Rate Schedulers
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['scheduler']['cosine']['T_max'],
        eta_min=cfg['scheduler']['cosine']['eta_min']
    )
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg['scheduler']['plateau']['mode'],
        factor=cfg['scheduler']['plateau']['factor'],
        patience=cfg['scheduler']['plateau']['patience'],
        threshold=cfg['scheduler']['plateau']['threshold'],
        cooldown=cfg['scheduler']['plateau']['cooldown'],
        verbose=True # Print message when LR reduces
    )

    # Mixed Precision Scaler (only if using fp16 on CUDA)
    scaler = GradScaler(enabled=use_amp)

    # --- Training State ---
    best_val_metric = 0.0 # Use the metric specified in ReduceLROnPlateau mode
    epochs_since_improvement = 0
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'val_hd95_et': [],
        'val_hd95_tc': [],
        'val_hd95_wt': [],
        'lr': []
    }

    # --- Training Loop ---
    print(f"Starting training for {cfg['epochs']} epochs on {device}...")
    for epoch in range(cfg['epochs']):
        epoch_start_time = time.time()
        model.train() # Set model to training mode
        epoch_train_loss = 0.0

        # Use tqdm for progress bar over batches
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} Training", leave=False)
        for batch_idx, (x, y) in enumerate(batch_iterator):
            x, y = x.to(device), y.to(device)

            # Forward pass with Automatic Mixed Precision
            with autocast(enabled=use_amp):
                pred = model(x)
                # Normalize loss by gradient accumulation steps
                loss = criterion(pred, y) / cfg['grad_accum']

            # Backward pass & Gradient Scaling
            scaler.scale(loss).backward()

            # Accumulate loss (before division by grad_accum)
            epoch_train_loss += loss.item() * cfg['grad_accum']

            # Gradient Accumulation & Optimizer Step
            if (batch_idx + 1) % cfg['grad_accum'] == 0:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                # Optimizer step
                scaler.step(optimizer)
                # Update scaler for next iteration
                scaler.update()
                # Zero gradients
                optimizer.zero_grad()
                # Step the cosine annealing scheduler after each optimizer step
                scheduler_cosine.step()

            # Update progress bar description (optional)
            # batch_iterator.set_postfix({'loss': loss.item() * cfg['grad_accum']})

        # --- End of Epoch ---
        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation Phase
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device, use_amp)

        # Calculate HD95 (might be slow, consider doing less frequently)
        # Ensure spacing is correctly obtained from config if needed, e.g., cfg['spacing']
        hd95_spacing = tuple(cfg.get('spacing', (1.0, 1.0, 1.0))) # Default if not in cfg
        hd95_scores = compute_brats_hd95(model, val_loader, device,
                                         spacing=hd95_spacing,
                                         precision=cfg['precision'])

        # Step the ReduceLROnPlateau scheduler based on validation Dice
        scheduler_plateau.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']

        # Log History
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        # Store HD95 scores, handling potential NaN values from compute_brats_hd95
        history['val_hd95_et'].append(hd95_scores.get('hd95_et', np.nan))
        history['val_hd95_tc'].append(hd95_scores.get('hd95_tc', np.nan))
        history['val_hd95_wt'].append(hd95_scores.get('hd95_wt', np.nan))
        history['lr'].append(current_lr)

        # Print Epoch Summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{cfg['epochs']} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Dice:   {val_dice:.4f}")
        print(f"  Val IoU:    {val_iou:.4f}")
        print(f"  Val HD95 (ET/TC/WT): {hd95_scores.get('hd95_et', np.nan):.2f} / "
              f"{hd95_scores.get('hd95_tc', np.nan):.2f} / "
              f"{hd95_scores.get('hd95_wt', np.nan):.2f} mm")
        print(f"  LR: {current_lr:.2e}")
        print(f"  Time: {epoch_time:.2f}s")


        # --- Checkpoint Saving & Early Stopping ---
        current_metric = val_dice # Metric to monitor for improvement

        if current_metric > best_val_metric + cfg['early_stop']['threshold']:
            best_val_metric = current_metric
            epochs_since_improvement = 0
            # Ensure save directory exists
            os.makedirs(os.path.dirname(cfg['model_save_path']), exist_ok=True)
            torch.save(model.state_dict(), cfg['model_save_path'])
            print(f"  Validation metric improved to {best_val_metric:.4f}. Saving best model to {cfg['model_save_path']}")
        else:
            epochs_since_improvement += 1
            print(f"  Validation metric did not improve for {epochs_since_improvement} epoch(s).")

        if epochs_since_improvement >= cfg['early_stop']['patience']:
            print(f"\nEarly stopping triggered at epoch {epoch+1} after {epochs_since_improvement} epochs without improvement.")
            break

    # --- End of Training ---
    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time / 60:.2f} minutes.")

    # --- Plotting Training History ---
    print("Plotting training history...")
    num_plots = 3 # Loss, Dice/IoU, HD95
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

    # Plot Loss
    axes[0].plot(history['epoch'], history['train_loss'], label='Train Loss')
    axes[0].plot(history['epoch'], history['val_loss'], label='Validation Loss')
    axes[0].set_title('Loss Progression')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Dice & IoU
    axes[1].plot(history['epoch'], history['val_dice'], label='Validation Dice')
    axes[1].plot(history['epoch'], history['val_iou'], label='Validation IoU', linestyle='--')
    axes[1].set_title('Validation Dice & IoU Scores')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True)

    # Plot HD95
    # Filter out potential NaNs before plotting HD95 for cleaner visualization
    epochs_with_hd95 = history['epoch'] # Assuming HD95 calculated every epoch
    axes[2].plot(epochs_with_hd95, [h if np.isfinite(h) else None for h in history['val_hd95_et']], label='Val HD95 ET', marker='.')
    axes[2].plot(epochs_with_hd95, [h if np.isfinite(h) else None for h in history['val_hd95_tc']], label='Val HD95 TC', marker='.')
    axes[2].plot(epochs_with_hd95, [h if np.isfinite(h) else None for h in history['val_hd95_wt']], label='Val HD95 WT', marker='.')
    axes[2].set_title('Validation HD95 (mm)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('HD95 (lower is better)')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    # Ensure log directory exists
    os.makedirs(cfg['log_dir'], exist_ok=True)
    plot_save_path = os.path.join(cfg['log_dir'], 'training_metrics.png')
    plt.savefig(plot_save_path)
    print(f"Training plot saved to {plot_save_path}")
    # plt.show() # Optionally display the plot interactively

    # Load best model weights before returning
    print(f"Loading best model weights from {cfg['model_save_path']}")
    if os.path.exists(cfg['model_save_path']):
         model.load_state_dict(torch.load(cfg['model_save_path'], map_location=device))
    else:
        print(f"Warning: Best model file '{cfg['model_save_path']}' not found. Returning model with weights from last epoch.")


    return model, history


# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Brain Tumor Segmentation Training ---")

    # 1. Load Configuration
    print("Loading configuration...")
    cfg = get_config() # Assumes get_config() returns a dictionary
    cfg['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define paths in config or here
    cfg['model_save_path'] = cfg.get('model_save_path', './best_model.pth') # Default save path
    cfg['log_dir'] = cfg.get('log_dir', './logs') # Default log directory

    print(f"Configuration loaded. Using device: {cfg['device']}")
    print(f"Model will be saved to: {cfg['model_save_path']}")
    print(f"Logs will be saved to: {cfg['log_dir']}")


    # 2. Initialize Dataset
    print("Initializing dataset...")
    # Ensure the dataset path in cfg points to your local data structure
    # Example: cfg['dataset_root'] = 'C:/Users/ashut/BrainTumorSegmentation_Project/TransUNet_BraTs2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    dataset = BrainMRIDataset_Local(
        root_dir=cfg['dataset_root'], # Get root directory from config
        img_size=cfg['img_size']
        # Add other necessary dataset arguments (e.g., specific modalities)
    )
    print(f"Dataset initialized. Found {len(dataset)} volumes/patients.")

    # 3. Split Dataset
    train_ratio = 0.9 # Example split ratio
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    print(f"Splitting dataset: {train_size} Train / {val_size} Validation")
    generator = torch.Generator().manual_seed(42) # for reproducible splits
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # 4. Compute Class Weights (using training set only)
    print("Computing class weights for training set...")
    # Consider parameters like border_boost from config if needed
    cfg['class_weights'] = compute_class_weights(
        train_dataset,
        border_boost=cfg.get('class_weight_border_boost', 1.5), # Example
        batch_size=cfg.get('class_weight_batch_size', 16) # Example
        # Pass other compute_class_weights args from cfg if needed
    ).to(cfg['device']) # Move weights to target device immediately

    # 5. Create DataLoaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        persistent_workers=True if cfg['num_workers'] > 0 else False # Avoid warning
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'] * 2, # Often use larger batch size for validation
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        persistent_workers=True if cfg['num_workers'] > 0 else False
    )
    print(f"Train DataLoader: {len(train_loader)} batches")
    print(f"Validation DataLoader: {len(val_loader)} batches")

    # 6. Start Training
    print("\n--- Starting Model Training ---")
    best_model, history = train_model(train_loader, val_loader, cfg)

    # 7. Final Evaluation (using the best model loaded by train_model)
    print("\n--- Final Evaluation on Validation Set using Best Model ---")
    # Use the comprehensive evaluation function from utils
    final_metrics = evaluate_with_brats_metrics(
        model=best_model, # train_model returns the best model
        loader=val_loader,
        device=cfg['device'],
        spacing=tuple(cfg.get('spacing', (1.0, 1.0, 1.0))) # Get spacing from cfg
    )

    # Print detailed final metrics
    print("\nDetailed Final BraTS Metrics (Validation Set):")
    print("="*50)
    print(f"{'Metric':<10} | {'ET':>8} | {'TC':>8} | {'WT':>8} | {'Mean':>8}")
    print("-"*50)
    # Use .get() with default np.nan for safety if a metric calculation failed
    print(f"{'Dice':<10} | {final_metrics.get('dice_et', np.nan):.4f} | {final_metrics.get('dice_tc', np.nan):.4f} | {final_metrics.get('dice_wt', np.nan):.4f} | {final_metrics.get('avg_dice', np.nan):.4f}")
    print(f"{'IoU':<10} | {final_metrics.get('iou_et', np.nan):.4f} | {final_metrics.get('iou_tc', np.nan):.4f} | {final_metrics.get('iou_wt', np.nan):.4f} | {final_metrics.get('avg_iou', np.nan):.4f}")
    print(f"{'HD95 (mm)':<10} | {final_metrics.get('hd95_et', np.nan):>8.2f} | {final_metrics.get('hd95_tc', np.nan):>8.2f} | {final_metrics.get('hd95_wt', np.nan):>8.2f} | {final_metrics.get('avg_hd95', np.nan):>8.2f}")
    print(f"{'HD (mm)':<10} | {final_metrics.get('hd_et', np.nan):>8.2f} | {final_metrics.get('hd_tc', np.nan):>8.2f} | {final_metrics.get('hd_wt', np.nan):>8.2f} | {final_metrics.get('avg_hd', np.nan):>8.2f}")
    print("="*50)

    # Optional: Visualize some predictions from the best model
    # print("\nVisualizing predictions from best model...")
    # visualize_predictions(best_model, val_dataset, cfg['device'], num_samples=5)

    # 8. Save Training History
    history_save_path = os.path.join(cfg['log_dir'], 'training_history.pt')
    torch.save(history, history_save_path)
    print(f"Training history saved to {history_save_path}")

    print("\n--- Training Script Finished ---")