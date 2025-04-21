# losses/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyFocalLoss(nn.Module):
    """Combination of Focal Loss and Tversky Loss for class-imbalanced segmentation.
    
    Attributes:
        alpha (float): Tversky alpha parameter (controls false positives)
        beta (float): Tversky beta parameter (controls false negatives)
        gamma (float): Focal loss focusing parameter
        smooth (float): Smoothing factor for numerical stability
        class_weights (Tensor): Optional class weighting tensor
        focal_alpha (float): Weighting between focal and Tversky components
    """

    def __init__(self, 
                 alpha: float = 0.7, 
                 beta: float = 0.3, 
                 gamma: float = 2.0,
                 smooth: float = 1e-6,
                 class_weights: torch.Tensor = None,
                 focal_alpha: float = 0.8):
        """Initialize Tversky-Focal loss parameters.
        
        Args:
            alpha: Weight for false positives in Tversky loss (0-1)
            beta: Weight for false negatives in Tversky loss (0-1)
            gamma: Focusing parameter for hard examples in Focal loss
            smooth: Additive smoothing factor to avoid division by zero
            class_weights: Per-class weights tensor of shape (num_classes,)
            focal_alpha: Weighting factor between Focal and Tversky losses
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined Tversky-Focal loss.
        
        Args:
            pred: Network predictions (B, C, H, W)
            target: Ground truth segmentation masks (B, H, W)
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Convert target to one-hot encoding
        target_onehot = F.one_hot(target, num_classes=4).permute(0, 3, 1, 2).float()
        
        # Calculate focal loss component
        logpt = self.log_softmax(pred)
        pt = torch.exp(logpt)
        ce_loss = -logpt * target_onehot
        
        if self.class_weights is not None:
            ce_loss = ce_loss * self.class_weights.view(1, -1, 1, 1)
            
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        # Calculate Tversky loss component for tumor classes
        tversky_loss = 0.0
        for c in [1, 2, 3]:  # Skip background class
            pc = F.softmax(pred, dim=1)[:, c]
            tc = target_onehot[:, c]
            
            tp = (pc * tc).sum(dim=(1, 2))
            fp = (pc * (1 - tc)).sum(dim=(1, 2))
            fn = ((1 - pc) * tc).sum(dim=(1, 2))
            
            numerator = tp + self.smooth
            denominator = numerator + self.alpha * fp + self.beta * fn + self.smooth
            
            tversky_loss += 1 - (numerator / denominator)
            
        tversky_loss = tversky_loss.mean() / 3  # Average over tumor classes

        # Combine losses
        return self.focal_alpha * focal_loss + (1 - self.focal_alpha) * tversky_loss


class BoundaryLoss(nn.Module):
    """Boundary-aware loss using Laplacian edge detection.
    
    Attributes:
        kernel (Tensor): Laplacian kernel for edge detection
        loss_fn (nn.Module): Base loss function (L1 by default)
    """

    def __init__(self):
        """Initialize boundary loss with 3x3 Laplacian kernel."""
        super().__init__()
        kernel = torch.tensor([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ], dtype=torch.float32)
        
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))  # (1,1,3,3)
        self.loss_fn = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate boundary alignment loss.
        
        Args:
            pred: Network predictions (B, C, H, W)
            target: One-hot encoded ground truth (B, C, H, W)
            
        Returns:
            torch.Tensor: Boundary loss value
        """
        # Expand kernel for multi-channel input
        B, C, H, W = pred.shape
        kernel = self.kernel.expand(C, 1, 3, 3)
        
        # Detect boundaries using Laplacian
        pred_boundary = F.conv2d(pred, weight=kernel, padding=1, groups=C)
        target_boundary = F.conv2d(target, weight=kernel, padding=1, groups=C)
        
        # Calculate L1 loss between boundary magnitudes
        return self.loss_fn(torch.abs(pred_boundary), torch.abs(target_boundary))


class CompositeLoss(nn.Module):
    """Combined region and boundary-aware segmentation loss.
    
    Attributes:
        tversky_focal_loss (nn.Module): Region-based loss component
        boundary_loss (nn.Module): Boundary-aware loss component
        boundary_loss_weight (float): Relative weight for boundary loss
    """

    def __init__(self, 
                 tversky_focal_loss: nn.Module,
                 boundary_loss_weight: float = 0.3):
        """Initialize composite loss components.
        
        Args:
            tversky_focal_loss: Initialized TverskyFocalLoss instance
            boundary_loss_weight: Weight for boundary loss (0-1)
        """
        super().__init__()
        self.tversky_focal_loss = tversky_focal_loss
        self.boundary_loss = BoundaryLoss()
        self.boundary_loss_weight = boundary_loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined segmentation loss.
        
        Args:
            pred: Network predictions (B, C, H, W)
            target: Ground truth segmentation masks (B, H, W)
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Convert predictions to probabilities and target to one-hot
        pred_prob = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Calculate component losses
        region_loss = self.tversky_focal_loss(pred, target)
        edge_loss = self.boundary_loss(pred_prob, target_onehot)
        
        # Return weighted combination
        return ((1 - self.boundary_loss_weight) * region_loss 
                + self.boundary_loss_weight * edge_loss)
    