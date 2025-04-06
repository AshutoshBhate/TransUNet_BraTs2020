"""
Configuration module for setting up hyperparameters and training settings.
"""

import torch


def get_config():
    """Get configuration settings for the TransUNet training pipeline.
    
    Returns:
        dict: Dictionary containing configuration settings.
    """
    return {
        # Hardware settings
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'precision': 'fp16',
        'num_workers': 4,
        
        # Data settings
        'img_size': 160,
        'batch_size': 8,
        'grad_accum': 4,
        'val_interval': 1,
        
        # Training settings
        'epochs': 120,
        'optimizer': 'AdamW',
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        
        # Learning Rate Scheduler settings
        'scheduler': {
            'type': 'hybrid',
            'cosine': {
                'T_max': 10,
                'eta_min': 1e-6
            },
            'plateau': {
                'mode': 'max',
                'factor': 0.5,
                'patience': 7,
                'threshold': 0.001,
                'cooldown': 2
            }
        },
        
        # Loss & Metrics settings
        'loss_params': {
            'tversky_alpha': 0.7,
            'tversky_beta': 0.3,
            'tversky_gamma': 2.0,
            'focal_alpha': 0.8,
            'boundary_loss_weight': 0.3
        },
        'class_weights': None,
        'metrics': ['dice', 'iou'],
        'early_stop': {
            'patience': 10,
            'threshold': 0.001
        },
        
        # Checkpointing & Logging settings (local paths)
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs'
    }
