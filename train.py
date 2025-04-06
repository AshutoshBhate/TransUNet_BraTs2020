# train.py

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

# Local imports
from config.config import get_config
from data.BrainMRIDataset_Class import BrainMRIDataset
from models.model import TransUNet
from losses.losses import TverskyFocalLoss, CompositeLoss
from utils.utils import (
    compute_class_weights,
    calculate_metrics,
    compute_brats_metrics,
    visualize_predictions,
    evaluate_with_torchmetrics
)


def train(train_loader: DataLoader, 
         val_loader: DataLoader, 
         cfg: dict) -> tuple[torch.nn.Module, dict]:
    """Train TransUNet model with validation and early stopping.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        cfg: Configuration dictionary
        
    Returns:
        tuple: (trained model, training history dictionary)
    """
    # Initialize model
    model = TransUNet().to(cfg['device'])
    
    # Initialize loss functions
    tversky_loss_fn = TverskyFocalLoss(
        class_weights=cfg['class_weights'],
        alpha=cfg['loss_params']['tversky_alpha'],
        beta=cfg['loss_params']['tversky_beta'],
        gamma=cfg['loss_params']['tversky_gamma'],
        focal_alpha=cfg['loss_params']['focal_alpha']
    ).to(cfg['device'])
    
    criterion = CompositeLoss(
        tversky_loss_fn, 
        boundary_loss_weight=cfg['loss_params']['boundary_loss_weight']
    ).to(cfg['device'])
    
    # Initialize optimizer and schedulers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['scheduler']['cosine']['T_max'],
        eta_min=cfg['scheduler']['cosine']['eta_min']
    )
    
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg['scheduler']['plateau']['mode'],
        factor=cfg['scheduler']['plateau']['factor'],
        patience=cfg['scheduler']['plateau']['patience'],
        threshold=cfg['scheduler']['plateau']['threshold'],
        cooldown=cfg['scheduler']['plateau']['cooldown']
    )
    
    # Training state
    best_dice = 0.0
    epochs_no_improve = 0
    history = {
        'train_loss': [],
        'val_dice': [],
        'val_iou': [],
        'lr': [],
        'brats_metrics': []
    }
    
    # Mixed precision training
    scaler = GradScaler(enabled=(cfg['precision'] == 'fp16'))
    
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        train_dice, train_iou = 0.0, 0.0
        
        # Training phase
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            x = x.to(cfg['device'], non_blocking=True)
            y = y.to(cfg['device'], non_blocking=True)
            
            with autocast(enabled=(cfg['precision'] == 'fp16')):
                pred = model(x)
                loss = criterion(pred, y) / cfg['grad_accum']
                dice, iou = calculate_metrics(pred.detach(), y)
                
            # Backpropagation
            scaler.scale(loss).backward()
            epoch_loss += loss.item * cfg['grad_accum']
            train_dice += dice
            train_iou += iou
            
            # Gradient accumulation
            if (batch_idx + 1) % cfg['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler_cosine.step()
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, criterion, cfg['device'])
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # BraTS metrics calculation
        if epoch == 0 or (epoch + 1) % 5 == 0:  # Reduce computation frequency
            brats_metrics = evaluate_with_torchmetrics(model, val_loader, cfg['device'])
            history['brats_metrics'].append(brats_metrics)
            print(f"BraTS Metrics - ET: {brats_metrics['dice_et']:.3f} | "
                  f"TC: {brats_metrics['dice_tc']:.3f} | WT: {brats_metrics['dice_wt']:.3f}")
        
        # Learning rate scheduling
        scheduler_plateau.step(val_metrics['dice'])
        
        # Early stopping check
        if val_metrics['dice'] > (best_dice + cfg['early_stop']['threshold']):
            best_dice = val_metrics['dice']
            epochs_no_improve = 0
            torch.save(model.state_dict(), "./best_model.pth")
        else:
            epochs_no_improve += 1
            
        # Print epoch summary
        print(f"Epoch {epoch+1}/{cfg['epochs']}")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"Val Dice: {val_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}\n")
        
        # Visualize first epoch predictions
        if epoch == 0:
            visualize_predictions(model, val_loader.dataset, cfg['device'], num_samples=3)
        
        # Early stopping
        if epochs_no_improve >= cfg['early_stop']['patience']:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return model, history


def evaluate(model: torch.nn.Module, 
            loader: DataLoader, 
            criterion: torch.nn.Module,
            device: torch.device) -> dict:
    """Evaluate model on validation set.
    
    Args:
        model: Initialized model
        loader: Validation data loader
        criterion: Loss function
        device: Target computation device
        
    Returns:
        dict: Dictionary containing validation metrics
    """
    model.eval()
    total_loss = 0.0
    dice_total, iou_total = 0.0, 0.0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            with autocast(enabled=True):
                pred = model(x)
                loss = criterion(pred, y)
                
            total_loss += loss.item()
            dice, iou = calculate_metrics(pred, y)
            dice_total += dice
            iou_total += iou
    
    return {
        'loss': total_loss / len(loader),
        'dice': dice_total / len(loader),
        'iou': iou_total / len(loader)
    }


if __name__ == "__main__":
    # Load configuration
    cfg = get_config()
    
    # Initialize dataset
    dataset = BrainMRIDataset(
        root_dir="BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        img_size=cfg['img_size']
    )
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, len(dataset) - train_size]
    )
    
    # Compute class weights
    cfg['class_weights'] = compute_class_weights(train_dataset).to(cfg['device'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True
    )
    
    # Train model
    model, history = train(train_loader, val_loader, cfg)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_metrics = evaluate_with_torchmetrics(model, val_loader, cfg['device'])
    print(f"Dice Scores - ET: {final_metrics['dice_et']:.3f} | "
          f"TC: {final_metrics['dice_tc']:.3f} | WT: {final_metrics['dice_wt']:.3f}")
    
    # Save final artifacts
    torch.save(model.state_dict(), "./final_model.pth")
    torch.save(history, "./training_history.pt")
    print("\nTraining completed and artifacts saved!")