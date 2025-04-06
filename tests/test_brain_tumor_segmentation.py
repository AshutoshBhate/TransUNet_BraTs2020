import os
import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import tempfile
import shutil
from unittest.mock import MagicMock, patch

# Import your modules – adjust the paths if needed.
from config.config import get_config
from data.brain_mri_dataset import BrainMRIDataset  # Now expects: root_dir, pattern, img_size
from models.model import TransUNet  # Assumes __init__ takes no extra parameters
from losses.losses import TverskyFocalLoss, CompositeLoss  # CompositeLoss expects: (loss_fn, boundary_loss_weight)
from utils.utils import calculate_metrics, compute_class_weights, evaluate_brats_regions, visualize_predictions

# Import training script functions.
from train import train, evaluate  # evaluate(model, loader, criterion, device, num_classes)


class TestBrainTumorSegmentation(unittest.TestCase):
    """Test suite for the Brain Tumor Segmentation TransUNet implementation."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create a temporary directory for test artifacts.
        cls.test_dir = tempfile.mkdtemp()
        # Set fixed random seeds for reproducibility.
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Determine device.
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running tests on device: {cls.device}")
        
        # Create a minimal test configuration.
        # Note: in_channels is set to 5 to match the model's expected input.
        cls.test_config = {
            'device': str(cls.device),
            'img_size': 128,         # Use a smaller size for faster tests.
            'in_channels': 5,        # Updated to 5 channels.
            'num_classes': 4,        # Background + 3 tumor regions.
            'batch_size': 2,
            'epochs': 2,             # Minimal epochs for testing.
            'validation_interval': 1,
            'train_val_split': 0.8,
            'grad_accum_steps': 1,
            'grad_clip_value': 1.0,
            'precision': 'fp32',     # Use fp32 for consistency.
            'lr': 0.0001,
            'weight_decay': 0.0001,
            'optimizer': 'AdamW',
            'scheduler': {
                'type': 'hybrid',
                'cosine': {'T_max': 10, 'eta_min': 1e-6},
                'plateau': {
                    'mode': 'max', 'factor': 0.5, 'patience': 5,
                    'threshold': 0.0001, 'cooldown': 0
                }
            },
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'threshold': 0.0001
            },
            'loss': {
                'type': 'TverskyFocalLoss',
                'params': {
                    'tversky_alpha': 0.7,
                    'tversky_beta': 0.3,
                    'tversky_gamma': 1.0,
                    'tversky_focal_alpha': 0.8,
                    'boundary_loss_weight': 0.0,
                },
                'class_weights': {'enabled': False, 'border_boost': 1.0}
            },
            'model': {
                # These fields are not used to instantiate the model in tests.
                'backbone_name': 'resnet34',
                'vit_name': 'R50-ViT-B_16',
                'pretrained': False
            },
            'checkpoint_dir': cls.test_dir,
            'best_model_name': 'test_best_model.pth',
            'save_every_epoch': False,
            'num_workers': 0  # Avoid multiprocessing in tests.
        }
        
        # Create synthetic data.
        cls.create_synthetic_dataset()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests are done."""
        shutil.rmtree(cls.test_dir)

    @classmethod
    def create_synthetic_dataset(cls):
        """Create synthetic data for testing."""
        # Create a small synthetic dataset (8 samples).
        batch_size = 8
        img_size = cls.test_config['img_size']
        in_channels = cls.test_config['in_channels']
        num_classes = cls.test_config['num_classes']
        
        # Create random input images with shape: (batch, in_channels, img_size, img_size).
        cls.test_images = torch.rand(batch_size, in_channels, img_size, img_size)
        # For targets, create label indices (not one-hot) as many losses expect labels.
        cls.test_labels = torch.randint(0, num_classes, (batch_size, img_size, img_size))
        
        # Create a DataLoader wrapping a TensorDataset.
        cls.test_dataset = TensorDataset(cls.test_images, cls.test_labels)
        cls.test_loader = DataLoader(
            cls.test_dataset, 
            batch_size=cls.test_config['batch_size'], 
            shuffle=True
        )

    def test_model_initialization(self):
        """Test that the TransUNet model initializes correctly."""
        print("\nTesting model initialization...")
        try:
            # Instantiate the model without extra parameters.
            model = TransUNet().to(self.device)
            
            # Test a forward pass using synthetic images.
            with torch.no_grad():
                test_batch = self.test_images[:2].to(self.device)
                output = model(test_batch)
                # Expected output shape: (batch, num_classes, img_size, img_size).
                expected_shape = (2, self.test_config['num_classes'], self.test_config['img_size'], self.test_config['img_size'])
                self.assertEqual(output.shape, expected_shape)
            print("Model initialization test passed!")
        except Exception as e:
            self.fail(f"Model initialization failed with error: {e}")

    def test_loss_functions(self):
        """Test the loss functions used in training."""
        print("\nTesting loss functions...")
        try:
            # Create a TverskyFocalLoss instance.
            tversky_loss = TverskyFocalLoss(
                alpha=self.test_config['loss']['params']['tversky_alpha'],
                beta=self.test_config['loss']['params']['tversky_beta'],
                gamma=self.test_config['loss']['params']['tversky_gamma'],
                focal_alpha=self.test_config['loss']['params']['tversky_focal_alpha'],
                class_weights=None
            ).to(self.device)
            
            # Create a CompositeLoss instance.
            composite_loss = CompositeLoss(
                tversky_loss,  # positional argument for loss function
                self.test_config['loss']['params']['boundary_loss_weight']
            ).to(self.device)
            
            # Create synthetic inputs and targets (targets as label indices).
            inputs = torch.randn(2, self.test_config['num_classes'], self.test_config['img_size'], self.test_config['img_size']).to(self.device)
            targets = torch.randint(0, self.test_config['num_classes'], (2, self.test_config['img_size'], self.test_config['img_size'])).to(self.device)
            
            tversky_loss_val = tversky_loss(inputs, targets)
            composite_loss_val = composite_loss(inputs, targets)
            
            self.assertTrue(torch.isfinite(tversky_loss_val))
            self.assertTrue(torch.isfinite(composite_loss_val))
            print("Loss functions test passed!")
        except Exception as e:
            self.fail(f"Loss functions test failed with error: {e}")

    def test_metrics_calculation(self):
        """Test the metrics calculation functions."""
        print("\nTesting metrics calculation...")
        try:
            # Create synthetic predictions and targets.
            preds = torch.randn(2, self.test_config['num_classes'], self.test_config['img_size'], self.test_config['img_size']).to(self.device)
            targets = torch.randint(0, self.test_config['num_classes'], (2, self.test_config['img_size'], self.test_config['img_size'])).to(self.device)
            preds = torch.softmax(preds, dim=1)
            
            # calculate_metrics expects only (pred, target).
            dice, iou = calculate_metrics(preds, targets)
            
            self.assertTrue(0 <= dice <= 1.0)
            self.assertTrue(0 <= iou <= 1.0)
            print("Metrics calculation test passed!")
        except Exception as e:
            self.fail(f"Metrics calculation test failed with error: {e}")

    @patch('matplotlib.pyplot.savefig')  # Avoid actual file saving.
    @patch('matplotlib.pyplot.figure')
    def test_visualization(self, mock_figure, mock_savefig):
        """Test visualization functions."""
        print("\nTesting visualization functions...")
        try:
            # Create a mock model that returns softmax predictions.
            mock_model = MagicMock()
            mock_model.eval.return_value = None
            
            def side_effect(x):
                batch_size = x.size(0)
                preds = torch.zeros(batch_size, self.test_config['num_classes'], self.test_config['img_size'], self.test_config['img_size']).to(self.device)
                for b in range(batch_size):
                    for c in range(self.test_config['num_classes']):
                        preds[b, c] = torch.rand_like(preds[b, c])
                return torch.softmax(preds, dim=1)
            
            mock_model.side_effect = side_effect
            
            # Test visualize_predictions.
            with patch('matplotlib.pyplot.subplots', return_value=(MagicMock(), MagicMock())):
                visualize_predictions(
                    mock_model,
                    self.test_dataset,
                    self.device,
                    num_samples=2,
                    num_classes=self.test_config['num_classes']
                )
            
            print("Visualization test passed!")
        except Exception as e:
            self.fail(f"Visualization test failed with error: {e}")

    def test_class_weights_computation(self):
        """Test the class weights computation function."""
        print("\nTesting class weights computation...")
        try:
            # Directly call compute_class_weights without patching non-existent functions.
            weights = compute_class_weights(
                self.test_dataset,
                border_boost=self.test_config['loss']['class_weights']['border_boost']
            )
            # Expect weights to have shape (num_classes,) and sum to num_classes.
            self.assertEqual(weights.shape, (self.test_config['num_classes'],))
            self.assertAlmostEqual(weights.sum().item(), float(self.test_config['num_classes']), delta=0.1)
            print("Class weights computation test passed!")
        except Exception as e:
            self.fail(f"Class weights computation test failed with error: {e}")

    def test_train_and_evaluate(self):
        """Test the training and evaluation functions."""
        print("\nTesting training and evaluation functions...")
        try:
            # For a fast test, mock the model and loss function.
            with patch('models.model.TransUNet') as mock_model_class:
                # Create a mock model that returns synthetic predictions.
                mock_model = MagicMock()
                mock_model.to.return_value = mock_model
                
                def model_forward(x):
                    batch_size = x.size(0)
                    return torch.randn(batch_size, self.test_config['num_classes'], 
                                         self.test_config['img_size'], self.test_config['img_size']).to(self.device)
                mock_model.side_effect = model_forward
                mock_model_class.return_value = mock_model
                
                # Mock the loss function.
                with patch('losses.losses.TverskyFocalLoss') as mock_loss_class:
                    mock_loss = MagicMock()
                    mock_loss.to.return_value = mock_loss
                    mock_loss.return_value = torch.tensor(0.5, device=self.device)
                    mock_loss_class.return_value = mock_loss
                    
                    # Mock calculate_metrics to return fixed values.
                    with patch('utils.utils.calculate_metrics', return_value=(0.7, 0.6)):
                        # Call evaluate with 5 arguments: (model, loader, criterion, device, num_classes)
                        val_loss, avg_dice, avg_iou = evaluate(
                            mock_model,
                            self.test_loader,
                            mock_loss,
                            self.device,
                            self.test_config['num_classes']
                        )
                        
                        self.assertTrue(isinstance(val_loss, float))
                        self.assertTrue(isinstance(avg_dice, float))
                        self.assertTrue(isinstance(avg_iou, float))
                        self.assertTrue(0 <= avg_dice <= 1.0)
                        self.assertTrue(0 <= avg_iou <= 1.0)
                        
                        # For train, avoid file I/O by patching os.makedirs and torch.save.
                        with patch('os.makedirs'):
                            with patch('torch.save'):
                                model_out, history = train(
                                    self.test_loader,  # Use same loader for train/val.
                                    self.test_loader,
                                    self.test_config
                                )
                                expected_keys = ['train_loss', 'val_loss', 'val_dice', 'val_iou', 'lr']
                                for key in expected_keys:
                                    self.assertIn(key, history)
                                self.assertEqual(len(history['train_loss']), self.test_config['epochs'])
            print("Training and evaluation test passed!")
        except Exception as e:
            self.fail(f"Training and evaluation test failed with error: {e}")

    @patch('builtins.print')  # Suppress print output in this test.
    def test_dataset_loading(self, mock_print):
        """Test dataset loading functionality."""
        print("\nTesting dataset loading...")
        try:
            # Patch the BrainMRIDataset class.
            with patch('data.brain_mri_dataset.BrainMRIDataset') as MockDataset:
                mock_dataset_instance = MagicMock()
                mock_dataset_instance.__len__.return_value = 100
                MockDataset.return_value = mock_dataset_instance

                def getitem(idx):
                    return (
                        self.test_images[idx % len(self.test_images)], 
                        self.test_labels[idx % len(self.test_labels)]
                    )
                mock_dataset_instance.__getitem__.side_effect = getitem

                # Use the keyword 'root_dir' as expected by your dataset.
                dataset = BrainMRIDataset(
                    root_dir="./mock_path",
                    pattern="BraTS20_Training_*",
                    img_size=self.test_config['img_size']
                )

                self.assertEqual(len(dataset), 100)
                train_size = int(self.test_config['train_val_split'] * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
                self.assertEqual(len(train_dataset) + len(val_dataset), len(dataset))
                self.assertEqual(len(train_dataset), train_size)
                self.assertEqual(len(val_dataset), val_size)
            print("Dataset loading test passed!")
        except Exception as e:
            self.fail(f"Dataset loading test failed with error: {e}")

    def test_config_loading(self):
        """Test configuration loading and validation."""
        print("\nTesting configuration loading...")
        try:
            with patch('config.config.get_config', return_value=get_config()):
                cfg = get_config()
                essential_keys = [
                    'device', 'precision', 'num_workers', 'img_size', 'batch_size',
                    'grad_accum', 'val_interval', 'epochs', 'optimizer', 'lr',
                    'weight_decay', 'grad_clip', 'scheduler', 'loss_params',
                    'class_weights', 'metrics', 'early_stop', 'checkpoint_dir', 'log_dir'
                ]
                for key in essential_keys:
                    self.assertIn(key, cfg)
            print("Configuration loading test passed!")
        except Exception as e:
            self.fail(f"Configuration loading test failed with error: {e}")


class TestBrainMRIDataset(unittest.TestCase):
    """Test suite specifically for the BrainMRIDataset class."""
    
    def setUp(self):
        """Set up for each test."""
        # Create a temporary directory that mimics the BraTS dataset structure.
        self.test_dir = tempfile.mkdtemp()
        # Create mock patient directories.
        for i in range(3):
            patient_dir = os.path.join(self.test_dir, f"BraTS20_Training_{i:03}")
            os.makedirs(patient_dir, exist_ok=True)
            
            # Create mock MRI files (e.g., t1, t1ce, t2, flair) with minimal content.
            for modality in ['t1', 't1ce', 't2', 'flair']:
                with open(os.path.join(patient_dir, f"{modality}.nii.gz"), 'wb') as f:
                    f.write(b'\0' * 1024)
            
            # Create a mock segmentation file.
            with open(os.path.join(patient_dir, "seg.nii.gz"), 'wb') as f:
                f.write(b'\0' * 1024)
    
    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.test_dir)
    
    @patch('glob.glob')
    def test_dataset_initialization(self, mock_glob):
        """Test dataset initialization with mocked file operations."""
        # Mock glob to return our test directories.
        mock_glob.return_value = [
            os.path.join(self.test_dir, f"BraTS20_Training_{i:03}") 
            for i in range(3)
        ]
        
        # Initialize dataset using 'root_dir' (as expected).
        with patch('os.path.exists', return_value=True):
            dataset = BrainMRIDataset(
                root_dir=self.test_dir,
                pattern="BraTS20_Training_*",
                img_size=128
            )
            
            # Check that 3 patient directories were found.
            self.assertEqual(len(dataset.patient_dirs), 3)
            
            # Test __getitem__ with mocked _load_slice.
            with patch.object(dataset, '_load_slice', return_value=(
                torch.rand(4, 128, 128),
                torch.zeros(4, 128, 128)
            )):
                img, mask = dataset[0]
                self.assertEqual(img.shape, (4, 128, 128))
                self.assertEqual(mask.shape, (4, 128, 128))
                self.assertTrue(isinstance(img, torch.Tensor))
                self.assertTrue(isinstance(mask, torch.Tensor))
            print("BrainMRIDataset initialization test passed!")


if __name__ == '__main__':
    unittest.main()
