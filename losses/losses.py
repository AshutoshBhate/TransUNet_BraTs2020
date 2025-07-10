"""Custom loss functions for medical image segmentation.

This module contains custom loss functions tailored for challenges commonly
found in medical imaging, such as severe class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceFocalLoss(nn.Module):
    
    """A combined Dice and Focal Loss for robust segmentation.

    This loss function merges Dice Loss, which is effective for handling class
    imbalance by focusing on overlap, with Focal Loss, which prioritizes
    hard-to-classify examples by down-weighting the loss for well-classified
    pixels. This combination often leads to better performance on imbalanced
    segmentation tasks like brain tumor segmentation.

    Args:
        dice_weight (float, optional): The weight assigned to the Dice loss
            component. Defaults to 0.6. The Focal loss weight will be
            (1 - dice_weight).
        focal_alpha (float, optional): The alpha balancing factor in Focal Loss.
            Defaults to 0.25.
        focal_gamma (float, optional): The gamma focusing parameter in Focal
            Loss. Defaults to 2.0.
        class_weights (torch.Tensor, optional): A tensor of weights to apply
            to each class in the cross-entropy calculation of Focal Loss.
            Defaults to None.
        smooth (float, optional): A small epsilon value to prevent division by
            zero in Dice calculation. Defaults to 1e-6.
    """
    
    def __init__(self, dice_weight=0.6, focal_alpha=0.25, focal_gamma=2.0, class_weights=None, smooth=1e-6):
        """Initializes the DiceFocalLoss module."""
        
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights
        self.smooth = smooth

    def forward(self, pred_logits, target):
        """Calculates the combined Dice and Focal loss.

        Args:
            pred_logits (torch.Tensor): The model's raw output (logits) of shape
                (B, C, H, W), where C is the number of classes.
            target (torch.Tensor): The ground truth mask of shape (B, H, W).

        Returns:
            torch.Tensor: The calculated combined loss value.
        """
        
        # --- Focal Loss Component ---
        # Standard cross-entropy, but with weighting for hard examples.
        ce_loss = F.cross_entropy(pred_logits, target, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss) # Probability of the correct class
        focal_loss = (self.focal_alpha * (1 - pt)**self.focal_gamma * ce_loss).mean()

        # --- Dice Loss Component ---
        # Calculates overlap, ignoring the background class.
        pred_softmax = F.softmax(pred_logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred_logits.shape[1]).permute(0, 3, 1, 2).float()

        # We compute Dice per-class and then average the scores.
        # This is more stable than a single macro-Dice score.
        # We ignore the background class (index 0).
        intersection = (pred_softmax[:, 1:] * target_onehot[:, 1:]).sum(dim=(0, 2, 3))
        union = pred_softmax[:, 1:].sum(dim=(0, 2, 3)) + target_onehot[:, 1:].sum(dim=(0, 2, 3))
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean() # Average the Dice score of the 3 tumor classes

        # --- Final Combined Loss ---
        # A weighted average of the two losses.
        combined_loss = (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
        
        return combined_loss