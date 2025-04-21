"""Configuration module for setting up hyperparameters and training settings."""

import torch


def get_config() -> dict:
    """Get configuration settings for the TransUNet training pipeline.
    
    Returns:
        dict: Configuration dictionary with these main sections:
            - Hardware settings (device, precision, etc.)
            - Data settings (image size, batch size, etc.)
            - Training settings (epochs, optimizer, etc.)
            - Learning rate scheduling
            - Loss function parameters
            - Metrics and early stopping
            - Checkpointing and logging paths
    """
    config = {
        # Hardware settings
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'precision': 'fp16',  # Options: 'fp32' or 'fp16'
        'num_workers': 4,     # Number of parallel data loading processes
        
        # Data settings
        'img_size': 160,      # Input image size (square)
        'batch_size': 8,      # Actual batch size per forward/backward pass
        'grad_accum': 4,      # Effective batch size = batch_size * grad_accum
        'val_interval': 1,    # Run validation every N epochs
        
        # Training settings
        'epochs': 84,
        'optimizer': 'AdamW', # Supported: 'AdamW', 'SGD', etc.
        'lr': 3e-4,          # Initial learning rate
        'weight_decay': 1e-4, # L2 regularization
        'grad_clip': 1.0,     # Gradient clipping value
        
        # Learning Rate Scheduler
        'scheduler': {
            'type': 'hybrid', # Options: 'hybrid', 'cosine', 'plateau'
            'cosine': {
                'T_max': 10,  # Number of iterations for cosine cycle
                'eta_min': 1e-6  # Minimum learning rate
            },
            'plateau': {
                'mode': 'max', # Metric to monitor ('max' or 'min')
                'factor': 0.5, # LR reduction factor
                'patience': 7, # Epochs to wait before reducing LR
                'threshold': 0.001, # Minimum improvement threshold
                'cooldown': 2  # Epochs to wait after LR reduction
            }
        },
        
        # Loss Function Parameters
        'loss_params': {
            'tversky_alpha': 0.7,  # Weight for false positives
            'tversky_beta': 0.3,   # Weight for false negatives
            'tversky_gamma': 2.0,  # Focal loss focusing parameter
            'focal_alpha': 0.8,    # Weight between focal and Tversky
            'boundary_loss_weight': 0.3  # Weight for boundary loss
        },
        
        # Metrics and Early Stopping
        'class_weights': None,  # Optional class weights tensor
        'metrics': ['dice', 'iou'],  # Metrics to track
        'early_stop': {
            'patience': 10,    # Epochs to wait before stopping
            'threshold': 0.001  # Minimum improvement threshold
        },
        
        # Path Settings
        'checkpoint_dir': './checkpoints',  # Model save directory
        'log_dir': './logs'                 # Training logs directory
    }
    
    # Post-processing validation
    if config['precision'] == 'fp16' and config['device'] == 'cpu':
        print("Warning: FP16 not supported on CPU, defaulting to FP32")
        config['precision'] = 'fp32'
        
    return config