import pytest
import torch
import torch.nn.functional as F
import sys
import os

# Add the parent directory to the path so we can import the losses module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.losses import TverskyFocalLoss, BoundaryLoss, CompositeLoss


class TestTverskyFocalLoss:
    def test_initialization(self):
        """Test that TverskyFocalLoss initializes with default parameters."""
        loss_fn = TverskyFocalLoss()
        assert loss_fn.alpha == 0.7
        assert loss_fn.beta == 0.3
        assert loss_fn.gamma == 2.0
        assert loss_fn.smooth == 1e-6
        assert loss_fn.focal_alpha == 0.8
        assert loss_fn.class_weights is None

    def test_initialization_with_params(self):
        """Test that TverskyFocalLoss initializes with custom parameters."""
        class_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        loss_fn = TverskyFocalLoss(alpha=0.4, beta=0.6, gamma=1.5, 
                                   smooth=1e-5, class_weights=class_weights, 
                                   focal_alpha=0.6)
        assert loss_fn.alpha == 0.4
        assert loss_fn.beta == 0.6
        assert loss_fn.gamma == 1.5
        assert loss_fn.smooth == 1e-5
        assert torch.all(loss_fn.class_weights == class_weights)
        assert loss_fn.focal_alpha == 0.6

    def test_perfect_prediction(self):
        """Test that loss is minimal with perfect prediction."""
        loss_fn = TverskyFocalLoss()
        batch_size, num_classes, height, width = 2, 4, 10, 10
        
        # Create perfect one-hot prediction 
        target = torch.zeros((batch_size, height, width), dtype=torch.long)
        target[:, 3:7, 3:7] = 1  # Class 1 in the middle
        target[:, 7:9, 7:9] = 2  # Class 2 in bottom right
        
        # Convert target to logits (high values for correct class)
        pred = torch.ones((batch_size, num_classes, height, width)) * -10.0
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    pred[b, target[b, h, w], h, w] = 10.0  # High logit for correct class
        
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1, f"Loss for perfect prediction should be near zero, got {loss.item()}"

    def test_worst_prediction(self):
        """Test that loss is high with completely wrong prediction."""
        loss_fn = TverskyFocalLoss()
        batch_size, num_classes, height, width = 2, 4, 10, 10
        
        # Create target with classes 0, 1, 2, 3
        target = torch.zeros((batch_size, height, width), dtype=torch.long)
        target[:, 3:7, 3:7] = 1  # Class 1 in the middle
        target[:, 7:9, 7:9] = 2  # Class 2 in bottom right
        
        # Create opposite prediction (high probability for wrong classes)
        pred = torch.ones((batch_size, num_classes, height, width)) * -10.0
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    wrong_class = (target[b, h, w] + 1) % num_classes
                    pred[b, wrong_class, h, w] = 10.0  # High logit for wrong class
        
        loss = loss_fn(pred, target)
        assert loss.item() > 1.0, f"Loss for worst prediction should be high, got {loss.item()}"

    def test_gradients(self):
        """Test that gradients are computed correctly."""
        loss_fn = TverskyFocalLoss()
        batch_size, num_classes, height, width = 2, 4, 10, 10
        
        # Create random prediction and target
        pred = torch.randn((batch_size, num_classes, height, width), requires_grad=True)
        target = torch.randint(0, num_classes, (batch_size, height, width))
        
        # Forward and backward pass
        loss = loss_fn(pred, target)
        loss.backward()
        
        # Check that gradients are not None or zero
        assert pred.grad is not None
        assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))


class TestBoundaryLoss:
    def test_initialization(self):
        """Test that BoundaryLoss initializes correctly."""
        loss_fn = BoundaryLoss()
        # Check kernel shape
        assert loss_fn.kernel.shape == (1, 1, 3, 3)
        # Check kernel values (Laplacian kernel)
        kernel_sum = loss_fn.kernel.sum().item()
        assert kernel_sum == 0, f"Laplacian kernel should sum to 0, got {kernel_sum}"
        assert loss_fn.kernel[0, 0, 1, 1].item() == -8, "Center of Laplacian kernel should be -8"

    def test_boundary_detection(self):
        """Test that boundary loss detects edges."""
        loss_fn = BoundaryLoss()
        batch_size, num_classes, height, width = 2, 4, 16, 16
        
        # Create a simple square target with clear boundaries
        target_onehot = torch.zeros((batch_size, num_classes, height, width))
        # Class 1 square in the middle
        target_onehot[:, 1, 4:12, 4:12] = 1.0
        
        # Create prediction with slightly offset squares
        pred = torch.zeros((batch_size, num_classes, height, width))
        # Background everywhere
        pred[:, 0] = 1.0
        # Class 1 square slightly offset
        pred[:, 0, 5:13, 5:13] = 0.0
        pred[:, 1, 5:13, 5:13] = 1.0
        
        loss = loss_fn(pred, target_onehot)
        assert loss.item() > 0, "Boundary loss should be positive for misaligned boundaries"
        
        # Test with identical prediction and target
        loss_identical = loss_fn(target_onehot, target_onehot)
        assert loss_identical.item() < loss.item(), "Loss should be lower for identical boundaries"

    def test_gradients(self):
        """Test that gradients flow through boundary loss."""
        loss_fn = BoundaryLoss()
        batch_size, num_classes, height, width = 2, 4, 16, 16
        
        # Create random prediction requiring gradients
        # Use original tensor to avoid non-leaf tensor issues
        pred_original = torch.rand((batch_size, num_classes, height, width), requires_grad=True)
        
        # Create target
        target = torch.zeros((batch_size, num_classes, height, width))
        target[:, 0, :8, :] = 1.0  # Class 0 in top half
        target[:, 1, 8:, :] = 1.0  # Class 1 in bottom half
        
        # Manually normalize to sum to 1 across classes
        pred_sum = pred_original.sum(dim=1, keepdim=True)
        pred = pred_original / pred_sum
        
        # Forward and backward pass
        loss = loss_fn(pred, target)
        loss.backward()
        
        # Check gradients on the original tensor
        assert pred_original.grad is not None
        assert not torch.allclose(pred_original.grad, torch.zeros_like(pred_original.grad))


class TestCompositeLoss:
    def test_initialization(self):
        """Test that CompositeLoss initializes correctly."""
        tversky_loss = TverskyFocalLoss()
        composite_loss = CompositeLoss(tversky_loss, boundary_loss_weight=0.3)
        
        assert composite_loss.tversky_focal_loss == tversky_loss
        assert isinstance(composite_loss.boundary_loss, BoundaryLoss)
        assert composite_loss.boundary_loss_weight == 0.3

    def test_forward_pass(self):
        """Test forward pass of CompositeLoss."""
        batch_size, num_classes, height, width = 2, 4, 16, 16
        
        # Create random prediction and target
        pred = torch.randn((batch_size, num_classes, height, width))
        target = torch.randint(0, num_classes, (batch_size, height, width))
        
        # Initialize losses
        tversky_loss = TverskyFocalLoss()
        composite_loss = CompositeLoss(tversky_loss, boundary_loss_weight=0.3)
        
        # Compute individual losses for comparison
        tversky_loss_value = tversky_loss(pred, target)
        
        # Convert for boundary loss
        pred_prob = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        boundary_loss_value = composite_loss.boundary_loss(pred_prob, target_onehot)
        
        # Expected combined loss
        expected_loss = 0.7 * tversky_loss_value + 0.3 * boundary_loss_value
        
        # Actual combined loss
        combined_loss = composite_loss(pred, target)
        
        # Check that combined loss matches expected combined loss
        assert torch.isclose(combined_loss, expected_loss, rtol=1e-4), \
            f"Expected {expected_loss.item()}, got {combined_loss.item()}"

    def test_gradients(self):
        """Test that gradients flow through composite loss."""
        batch_size, num_classes, height, width = 2, 4, 16, 16
        
        # Create random prediction requiring gradients
        pred = torch.randn((batch_size, num_classes, height, width), requires_grad=True)
        target = torch.randint(0, num_classes, (batch_size, height, width))
        
        # Initialize losses
        tversky_loss = TverskyFocalLoss()
        composite_loss = CompositeLoss(tversky_loss, boundary_loss_weight=0.3)
        
        # Forward and backward pass
        loss = composite_loss(pred, target)
        loss.backward()
        
        # Check gradients
        assert pred.grad is not None
        assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))


if __name__ == "__main__":
    # Simple test runner
    pytest.main(["-xvs", __file__])