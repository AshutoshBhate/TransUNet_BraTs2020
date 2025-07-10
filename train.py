"""Main training script for the TransUNet Brain Tumor Segmentation model.

This script orchestrates the entire training and evaluation process. It handles:
- Setting up the configuration.
- Initializing the datasets and dataloaders.
- Initializing the model, optimizer, scheduler, and loss function.
- Running the main training loop, including validation, checkpointing, and
  early stopping.
- Plotting training metrics and saving the final model.
- Performing a final evaluation on the best model.
"""

import os
from sklearn.model_selection import train_test_split
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Local Project Imports ---
from config.config import get_config
from data.BrainMRIDataset_Class import BrainMRIDataset_Optimized, PatientSampler
from losses.losses import DiceFocalLoss
from models.model import TransUNet
from utils.utils import (
    get_train_transforms,
    compute_enhanced_class_weights,
    evaluate,
    compute_brats_hd95,
    evaluate_with_torchmetrics,
    visualize_predictions
)

def train(model, optimizer, scheduler, criterion, train_loader, val_loader, cfg_dict):
    """The main training and validation loop.

    Args:
        model (nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        criterion (nn.Module): The loss function.
        train_loader (DataLoader): The DataLoader for the training set.
        val_loader (DataLoader): The DataLoader for the validation set.
        cfg_dict (dict): The configuration dictionary containing all hyperparameters.

    Returns:
        tuple[nn.Module, dict]: A tuple containing the trained model and a
                                dictionary of its training history.
    """
    
    scaler = GradScaler(enabled=(cfg_dict['precision']=='fp16' and cfg_dict['device']!='cpu'))

    start_epoch = 0
    best_val_dice = 0.0
    epochs_no_improve = 0 # For early stopping
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': [],
        'val_hd95_et': [], 'val_hd95_tc': [], 'val_hd95_wt': [], 'lr': []
    }

    total_epochs = 0
    # Attempt to load checkpoint
    if cfg_dict.get('checkpoint_load_path') and os.path.exists(cfg_dict['checkpoint_load_path']):
        print(f"Loading checkpoint from {cfg_dict['checkpoint_load_path']}")
        try:
            checkpoint = torch.load(cfg_dict['checkpoint_load_path'], map_location=cfg_dict['device'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("Loaded GradScaler state from checkpoint.")    
                 
            start_epoch = checkpoint.get('epoch', 0)
            total_epochs = checkpoint.get('total_epochs', 0)  # Cumulative
            print(f"Resuming: Previously trained {total_epochs} total epochs")
            history = checkpoint.get('history', history) # Use loaded history if available
            best_val_dice = checkpoint.get('best_val_metric', best_val_dice) # Use 'best_val_metric' or 'best_dice'
            # scaler.load_state_dict(checkpoint['scaler_state_dict']) # If also saving scaler state
            print(f"Resumed training from epoch {start_epoch}. Best Val Dice: {best_val_dice:.4f}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")
            # Potentially delete corrupted checkpoint os.remove(cfg_dict['checkpoint_load_path'])
            
    # Add this before training loop:
    if total_epochs >= cfg_dict['target_total_epochs']:
        print("Already reached target epochs!")
        return model, history

    for epoch in range(start_epoch, start_epoch + cfg_dict['session_epochs']):
        model.train()
        epoch_train_loss = 0.0
        optimizer.zero_grad() # Clear gradients at the start of epoch
        
        # After calculating avg_train_loss
        current_total = total_epochs + 1
        
        if current_total >= cfg_dict['target_total_epochs']:
            print(f"Reached target {cfg_dict['target_total_epochs']} total epochs!")
            break  # Exit loop early

        for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + cfg_dict['session_epochs']} (Total: {current_total}/{cfg_dict['target_total_epochs']})"
)):
            x_batch, y_batch = x_batch.to(cfg_dict['device']), y_batch.to(cfg_dict['device'])

            with autocast(enabled=(cfg_dict['precision']=='fp16')):
                pred_logits = model(x_batch)
                loss = criterion(pred_logits, y_batch)
                if cfg_dict['grad_accum'] > 1:
                    loss = loss / cfg_dict['grad_accum']
            
            scaler.scale(loss).backward()
            epoch_train_loss += loss.item() * cfg_dict['grad_accum'] # Accumulate original loss

            if (batch_idx + 1) % cfg_dict['grad_accum'] == 0 or (batch_idx + 1) == len(train_loader):
                if cfg_dict['grad_clip'] > 0:
                    scaler.unscale_(optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_dict['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad()

        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, cfg_dict['device'])
        scheduler.step()

        # Compute HD95 only every N epochs (here N=5)
        if (epoch + 1) % 5 == 0:
            hd95_scores = compute_brats_hd95(
                model,
                val_loader,
                cfg_dict['device'],
                spacing=(1.0, 1.0),
                precision=cfg_dict['precision']
            )
        else:
            # insert NaNs so that history stays aligned and plots omit these points
            hd95_scores = {'hd95_et': np.nan, 'hd95_tc': np.nan, 'hd95_wt': np.nan}
            
        total_epochs += 1

        # Log history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['val_hd95_et'].append(hd95_scores.get('hd95_et', np.nan))
        history['val_hd95_tc'].append(hd95_scores.get('hd95_tc', np.nan))
        history['val_hd95_wt'].append(hd95_scores.get('hd95_wt', np.nan))
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"\nEpoch {epoch+1}/{start_epoch + cfg_dict['session_epochs']} (Total: {current_total}/{cfg_dict['target_total_epochs']}) | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Val HD95: ET={hd95_scores.get('hd95_et', np.nan):.2f}, TC={hd95_scores.get('hd95_tc', np.nan):.2f}, WT={hd95_scores.get('hd95_wt', np.nan):.2f}")

        # Save checkpoint
        current_checkpoint_state = {
            'epoch': epoch + 1,
            'total_epochs': total_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_val_metric': best_val_dice, # Save best_val_dice
            'config': cfg_dict, # Optional: save the config
            'scaler_state_dict': scaler.state_dict()
        }
        if (epoch + 1) % cfg_dict['checkpoint_save_interval'] == 0 or current_total >= cfg_dict['target_total_epochs']:
            torch.save(current_checkpoint_state, cfg_dict['checkpoint_save_path'])
            print(f"Checkpoint saved to {cfg_dict['checkpoint_save_path']} at epoch {epoch+1}")

        # Save best model based on validation Dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), cfg_dict['best_model_path']) # Save only model weights for best
            print(f"Best model weights updated and saved to {cfg_dict['best_model_path']} (Val Dice: {best_val_dice:.4f})")
            epochs_no_improve = 0 # Reset counter
            torch.save(current_checkpoint_state, cfg_dict['best_checkpoint_path'])
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= cfg_dict['early_stop']['patience'] and current_total > cfg_dict.get('min_epochs_for_early_stop', 20):
            print(f"\nEarly stopping triggered at epoch {epoch+1} after {epochs_no_improve} epochs without improvement on Val Dice.")
            break
        
        if total_epochs >= cfg_dict['target_total_epochs']:
            print(f"Reached target of {cfg_dict['target_total_epochs']} total epochs!")
            break # Exit the loop after this epoch is fully completed and saved.


    # Plotting training history
    if history['epoch']: # Check if history is not empty
        plt.figure(figsize=(20, 12))
        plt.subplot(2, 3, 1)
        plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
        plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
        plt.title('Loss Progression'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

        plt.subplot(2, 3, 2)
        plt.plot(history['epoch'], history['val_dice'], label='Validation Dice')
        plt.title('Validation Dice Score'); plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.legend(); plt.grid(True)

        plt.subplot(2, 3, 3)
        plt.plot(history['epoch'], history['val_iou'], label='Validation IoU', color='green')
        plt.title('Validation IoU Score'); plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.legend(); plt.grid(True)
        
        plt.subplot(2, 3, 4)
        
        # --- NEW LOGIC TO CONNECT HD95 POINTS ---
        
        # Filter data for ET (Enhancing Tumor)
        hd95_et_epochs = [epoch for epoch, score in zip(history['epoch'], history['val_hd95_et']) if not np.isnan(score)]
        hd95_et_scores = [score for score in history['val_hd95_et'] if not np.isnan(score)]
        if hd95_et_epochs:
            plt.plot(hd95_et_epochs, hd95_et_scores, label='HD95 ET', linestyle=':', marker='o')

        # Filter data for TC (Tumor Core)
        hd95_tc_epochs = [epoch for epoch, score in zip(history['epoch'], history['val_hd95_tc']) if not np.isnan(score)]
        hd95_tc_scores = [score for score in history['val_hd95_tc'] if not np.isnan(score)]
        if hd95_tc_epochs:
            plt.plot(hd95_tc_epochs, hd95_tc_scores, label='HD95 TC', linestyle='--', marker='o')

        # Filter data for WT (Whole Tumor)
        hd95_wt_epochs = [epoch for epoch, score in zip(history['epoch'], history['val_hd95_wt']) if not np.isnan(score)]
        hd95_wt_scores = [score for score in history['val_hd95_wt'] if not np.isnan(score)]
        if hd95_wt_epochs:
            plt.plot(hd95_wt_epochs, hd95_wt_scores, label='HD95 WT', linestyle='-.', marker='o')
        
        # Adding markers ('o') is good practice to show where actual data points exist.
        plt.title('Validation HD95 (mm)'); plt.xlabel('Epoch'); plt.ylabel('HD95'); plt.legend(); plt.grid(True);

        plt.subplot(2, 3, 5)
        plt.plot(history['epoch'], history['lr'], label='Learning Rate', color='purple')
        plt.title('Learning Rate Schedule'); plt.xlabel('Epoch'); plt.ylabel('LR'); plt.legend(); plt.grid(True)

        plt.tight_layout()
        os.makedirs(cfg_dict['log_dir'], exist_ok=True)
        plt.savefig(os.path.join(cfg_dict['log_dir'], 'training_metrics_transunet.png'))
        print(f"Training metrics plot saved to {os.path.join(cfg_dict['log_dir'], 'training_metrics_transunet.png')}")
        # plt.show()
    else:
        print("No history to plot (e.g. training did not run or resume correctly).")

    return model, history

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TransUNet for Brain Tumor Segmentation")
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to the root BraTS 2020 training data directory (e.g., '.../MICCAI_BraTS2020_TrainingData')")
    parser.add_argument('--epochs', type=int, help="Override number of session epochs from config.")
    parser.add_argument('--batch_size', type=int, help="Override batch size from config.")
    parser.add_argument('--lr', type=float, help="Override learning rate from config.")
    parser.add_argument('--resume', action='store_true',
                        help="Set this flag to resume training from the latest checkpoint.")
    args = parser.parse_args()

    # 2. LOAD AND UPDATE CONFIG
    cfg = get_config()

    # Override config with command-line arguments if provided
    if args.epochs:
        cfg['session_epochs'] = args.epochs
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.lr:
        cfg['lr'] = args.lr

    # Handle checkpoint resuming
    if args.resume and os.path.exists(cfg['checkpoint_save_path']):
        cfg['checkpoint_load_path'] = cfg['checkpoint_save_path']
        print(f"Resuming training from checkpoint: {cfg['checkpoint_load_path']}")
    else:
        print("Starting training from scratch.")


    # 3. CREATE OUTPUT DIRECTORIES
    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    print(f"Logs will be saved to: {os.path.abspath(cfg['log_dir'])}")
    print(f"Checkpoints will be saved to: {os.path.abspath(cfg['checkpoint_dir'])}")


    # 4. DATASET AND DATALOADER SETUP (USING LOCAL PATH)
    data_root = args.data_path  # Use the path from the command line
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    all_patients = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    print(f"Found {len(all_patients)} patient folders in {data_root}")

    # do a 90/10 patient‐level split
    train_ids, val_ids = train_test_split(
        all_patients, test_size=0.1, random_state=42
    )

    train_augs = get_train_transforms(cfg['img_size'])

    # then build TWO separate BrainMRIDatasets
    train_dataset = BrainMRIDataset_Optimized(
        root_dir=data_root,
        img_size=cfg['img_size'],
        patient_list=train_ids,
        negative_slice_ratio=cfg['negative_slice_ratio'], # From new config
        transforms=train_augs                              # Pass augmentations
    )
    val_dataset = BrainMRIDataset_Optimized(
        root_dir=data_root,
        img_size=cfg['img_size'],
        patient_list=val_ids,
        negative_slice_ratio=cfg['negative_slice_ratio'], # Use same ratio for validation
        transforms=None                                    # No augmentations for validation
    )

    print(f"Train patients: {len(train_ids)}, Validation patients: {len(val_ids)}")
    print(f"Train slices: {len(train_dataset)}, Val slices: {len(val_dataset)}")

    # Compute class weights (on training set only)
    if len(train_dataset) > 0:
        # This calls the new, fast, patient-centric function
        cfg['class_weights'] = compute_enhanced_class_weights(train_dataset).to(cfg['device']) 
    else:
        print("Warning: Training dataset is empty, cannot compute class weights.")
        cfg['class_weights'] = torch.ones(4, device=cfg['device']) / 4.0

    # DataLoaders
    
    train_sampler = PatientSampler(train_dataset.slice_pointers)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['batch_size'], 
        sampler=train_sampler,  # <--- USE THE CUSTOM SAMPLER
        shuffle=False,         # <--- REMOVE OR SET TO FALSE
        num_workers=cfg['num_workers'], 
        pin_memory=True, 
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=False,
        num_workers=cfg['num_workers'], 
        pin_memory=True, 
        persistent_workers=True
    )
    # Initialize model, optimizer, schedulers, criterion
    model = TransUNet().to(cfg['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg['scheduler']['T_0'],
        T_mult=cfg['scheduler']['T_mult'],
        eta_min=cfg['scheduler']['eta_min']
    )
    
    criterion = DiceFocalLoss(
        dice_weight=cfg['loss_params']['dice_weight'],
        focal_alpha=cfg['loss_params']['focal_alpha'],
        focal_gamma=cfg['loss_params']['focal_gamma'],
        class_weights=cfg['class_weights']
    ).to(cfg['device'])
    
    # Start training
    trained_model, training_history = train(
        model, optimizer, scheduler, criterion,
        train_loader, val_loader, cfg
    )
    
    # --- Final Evaluation using the best model weights ---
    print("\nLoading best model for final evaluation...")
    best_model_eval = TransUNet().to(cfg['device'])
    try:
        # Load the best model weights saved during training
        best_model_path = cfg['best_model_path']
        if os.path.exists(best_model_path):
            best_model_eval.load_state_dict(torch.load(best_model_path, map_location=cfg['device']))
            best_model_eval.eval()
            print(f"Successfully loaded best model from {best_model_path}")

            # Run a full evaluation with all metrics
            final_metrics = evaluate_with_torchmetrics(best_model_eval, val_loader, cfg['device'])
            print("\n--- Final Best Model Performance ---")
            print(f"Mean Dice: {final_metrics['avg_dice']:.4f}, Mean IoU: {final_metrics['avg_iou']:.4f}")
            print(f"Val HD95 (ET/TC/WT): {final_metrics['hd95_et']:.2f} / {final_metrics['hd95_tc']:.2f} / {final_metrics['hd95_wt']:.2f} mm")

            # Visualize some predictions from the validation set
            print("\nGenerating prediction visualizations...")
            visualize_predictions(best_model_eval, val_dataset, cfg['device'], num_samples=5)

        else:
            print(f"Warning: Best model path not found at '{best_model_path}'. Skipping final evaluation.")

    except Exception as e:
        print(f"Error during final evaluation: {e}")
