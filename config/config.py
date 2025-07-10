"""Configuration module for the Brain Tumor Segmentation project.

This module defines a function that returns a dictionary containing all the
hyperparameters and settings for data loading, model training, and evaluation.
Centralizing the configuration makes it easy to experiment with different
settings without modifying the core logic of the training pipeline.
"""

import torch

def get_config():
    
    """Returns a dictionary of configuration settings for the project.

    The configuration includes hardware settings, data parameters, training
    hyperparameters, loss function parameters, and paths for logging and
    checkpointing.

    Returns:
        dict: A dictionary containing all configuration parameters.
    """
    
    cfg = {
        # Hardware settings
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'precision': 'fp16',
        'num_workers': 4,

        # Data settings
        'img_size': 224,
        'batch_size': 8,
        'grad_accum': 4,
        'val_interval': 1,
        'negative_slice_ratio': 0.25,

        # Training settings
        'session_epochs': 23,
        'target_total_epochs': 120,
        'optimizer': 'AdamW',
        'lr': 3e-4,
        'weight_decay': 5e-4,
        'grad_clip': 1.0,

        # Loss & Metrics
        'loss_params': {
            'dice_weight': 0.6,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0
        },
        
        'class_weights': None,
        'metrics': ['dice', 'iou', 'hd95'],
        'early_stop': {
            'patience': 15,
            'threshold': 0.001
        },

        # Checkpointing & Logging
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'best_model_path': "checkpoints/best_model_transunet.pth",
        'checkpoint_save_path': "checkpoints/transunet_checkpoint.pth.tar",
        'best_checkpoint_path': "checkpoints/best_transunet_checkpoint.pth.tar",
        'checkpoint_load_path': None, # This will be set dynamically
        'checkpoint_save_interval': 1
    }

    # Adjust precision if CPU is used
    if cfg['device'] == 'cpu':
        cfg['precision'] = 'fp32'

    # Scheduler
    cfg['scheduler'] = {
            'type': 'CosineAnnealingWarmRestarts',
            'T_0': 15,      # Epochs for the first restart.
            'T_mult': 2,    # Multiplier to expand the cycle after each restart (15, 30, 60...)
            'eta_min': 5e-6 # Minimum learning rate.
        }

    return cfg