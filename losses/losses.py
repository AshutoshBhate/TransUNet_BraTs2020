import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyFocalLoss(nn.Module):
    """Computes a composite Tversky + Focal loss for multi-class segmentation."""

    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-6, 
                 class_weights=None, focal_alpha=0.8):
        """
        Initializes TverskyFocalLoss.

        Args:
            alpha (float): Weight for false positives in Tversky loss.
            beta (float): Weight for false negatives in Tversky loss.
            gamma (float): Focusing parameter for Focal loss.
            smooth (float): Small constant to avoid division by zero.
            class_weights (Tensor, optional): Class-specific weights for cross-entropy.
            focal_alpha (float): Weight to balance between Focal and Tversky losses.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma
        self.focal_alpha = focal_alpha
        self.class_weights = class_weights
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, pred, target):
        """
        Computes the loss.

        Args:
            pred (Tensor): Raw logits of shape [B, C, H, W].
            target (Tensor): Ground truth labels of shape [B, H, W].

        Returns:
            Tensor: Scalar loss value.
        """
        # Convert ground truth to one-hot encoding
        target_onehot = F.one_hot(target, num_classes=4).permute(0, 3, 1, 2).float()
        
        # Compute log-probabilities and probabilities
        logpt = self.log_softmax(pred)
        pt = torch.exp(logpt)
        
        # Standard cross-entropy loss
        ce_loss = -logpt * target_onehot
        if self.class_weights is not None:
            ce_loss = ce_loss * self.class_weights.view(1, -1, 1, 1)
        
        # Focal loss calculation
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        # Tversky loss across classes 1, 2, 3 (ignoring background)
        tversky_loss = 0.0
        for c in [1, 2, 3]:
            pc = F.softmax(pred, dim=1)[:, c]   # Predicted probability for class c
            tc = target_onehot[:, c]            # Ground truth mask for class c

            # Compute Tversky components
            tp = (pc * tc).sum(dim=(1, 2))
            fp = (pc * (1 - tc)).sum(dim=(1, 2))
            fn = ((1 - pc) * tc).sum(dim=(1, 2))

            numerator = tp + self.smooth
            denominator = numerator + self.alpha * fp + self.beta * fn + self.smooth
            tversky_loss += 1 - (numerator / denominator)

        tversky_loss = tversky_loss.mean() / 3

        # Combine focal and Tversky losses
        return self.focal_alpha * focal_loss + (1 - self.focal_alpha) * tversky_loss


class BoundaryLoss(nn.Module):
    """Computes boundary loss using a Laplacian edge detector."""

    def __init__(self):
        """
        Initializes the BoundaryLoss with a fixed Laplacian kernel.
        """
        super(BoundaryLoss, self).__init__()
        kernel = torch.tensor([[1, 1, 1],
                               [1, -8, 1],
                               [1, 1, 1]], dtype=torch.float32)
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))  # Shape: [1, 1, 3, 3]
        self.loss_fn = nn.L1Loss()

    def forward(self, pred, target):
        """
        Computes boundary loss between predicted and target masks.

        Args:
            pred (Tensor): Predicted probability maps of shape [B, C, H, W].
            target (Tensor): One-hot encoded ground truth of shape [B, C, H, W].

        Returns:
            Tensor: Scalar loss value.
        """
        B, C, H, W = pred.shape

        # Expand kernel to match number of channels for group-wise convolution
        kernel = self.kernel.expand(C, 1, 3, 3)

        # Apply Laplacian kernel to extract boundaries
        pred_boundary = F.conv2d(pred, weight=kernel, bias=None, stride=1, padding=1, groups=C)
        target_boundary = F.conv2d(target, weight=kernel, bias=None, stride=1, padding=1, groups=C)

        # Use absolute values to emphasize edge magnitude
        pred_boundary = torch.abs(pred_boundary)
        target_boundary = torch.abs(target_boundary)

        # Compute L1 loss between boundary maps
        loss = self.loss_fn(pred_boundary, target_boundary)
        return loss


class CompositeLoss(nn.Module):
    """Combines Tversky-Focal loss with boundary-aware loss."""

    def __init__(self, tversky_focal_loss, boundary_loss_weight=0.3):
        """
        Initializes CompositeLoss.

        Args:
            tversky_focal_loss (nn.Module): Instance of TverskyFocalLoss.
            boundary_loss_weight (float): Weight of boundary loss (e.g., 0.3).
        """
        super(CompositeLoss, self).__init__()
        self.tversky_focal_loss = tversky_focal_loss
        self.boundary_loss = BoundaryLoss()
        self.boundary_loss_weight = boundary_loss_weight

    def forward(self, pred, target):
        """
        Computes combined segmentation + boundary loss.

        Args:
            pred (Tensor): Raw logits of shape [B, C, H, W].
            target (Tensor): Ground truth labels of shape [B, H, W].

        Returns:
            Tensor: Scalar combined loss.
        """
        # Region-based loss using Tversky + Focal
        tversky_loss = self.tversky_focal_loss(pred, target)

        # Convert logits to probabilities
        pred_prob = F.softmax(pred, dim=1)

        # Convert target to one-hot encoding
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        # Compute boundary loss between predicted and true masks
        boundary_loss = self.boundary_loss(pred_prob, target_onehot)

        # Weighted combination of both losses
        composite_loss = (1 - self.boundary_loss_weight) * tversky_loss + self.boundary_loss_weight * boundary_loss
        return composite_loss


# Example usage in training loop:
# tversky_loss_fn = TverskyFocalLoss(alpha=0.5, beta=0.5, gamma=2.0)
# loss_fn = CompositeLoss(tversky_loss_fn, boundary_loss_weight=0.3)
