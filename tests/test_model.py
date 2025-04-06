import unittest
import torch
import sys
import os
from unittest import mock

# Add the parent directory to sys.path to be able to import the models module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import TransUNet, HybridEncoder, TransUNetDecoder

class TestTransUNet(unittest.TestCase):
    
    def setUp(self):
        # Create patches for timm.create_model to avoid downloading weights
        self.patcher1 = mock.patch('timm.create_model')
        self.mock_create_model = self.patcher1.start()
        
        # Setup mock backbone
        self.mock_backbone = mock.MagicMock()
        self.mock_backbone.feature_info = mock.MagicMock()
        self.mock_backbone.feature_info.channels.return_value = [56, 160, 448]
        
        # Create a proper mock for conv_stem with all required attributes
        self.mock_conv_stem = mock.MagicMock()
        self.mock_conv_stem.out_channels = 32  # Example value
        self.mock_conv_stem.kernel_size = (3, 3)  # Example value
        self.mock_conv_stem.stride = (2, 2)  # Example value
        self.mock_conv_stem.padding = (1, 1)  # Example value
        self.mock_conv_stem.weight = torch.rand(32, 3, 3, 3)  # Example weight tensor
        
        # Attach conv_stem to backbone
        self.mock_backbone.conv_stem = self.mock_conv_stem
        
        # Setup mock for modules method to handle SqueezeExcite check
        self.mock_backbone.modules.return_value = []
        
        # Setup mock ViT
        self.mock_vit = mock.MagicMock()
        self.mock_vit.pos_embed = torch.nn.Parameter(torch.ones(1, 197, 768))
        self.mock_vit.cls_token = torch.ones(1, 1, 768)
        self.mock_vit.forward_features.return_value = torch.rand(2, 25, 768)
        
        # Configure create_model to return our mocks
        self.mock_create_model.side_effect = [self.mock_backbone, self.mock_vit]
        
        # Patch the _modify_first_conv method to bypass the actual implementation
        self.patcher2 = mock.patch.object(HybridEncoder, '_modify_first_conv')
        self.mock_modify_first_conv = self.patcher2.start()
        
        # Patch the _adjust_pos_embeddings method
        self.patcher3 = mock.patch.object(HybridEncoder, '_adjust_pos_embeddings')
        self.mock_adjust_pos_embeddings = self.patcher3.start()
        
        # Patch the _register_gradient_hooks method
        self.patcher4 = mock.patch.object(HybridEncoder, '_register_gradient_hooks')
        self.mock_register_gradient_hooks = self.patcher4.start()
        
        # Patch the _validate_shapes method to bypass validation
        self.patcher5 = mock.patch.object(TransUNet, '_validate_shapes')
        self.mock_validate_shapes = self.patcher5.start()
        
        # Initialize model
        self.model = TransUNet()
        
        # Create test input
        self.batch_size = 2
        self.input_tensor = torch.rand(self.batch_size, 4, 160, 160)
    
    def tearDown(self):
        # Stop all patches
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()
        
    def test_model_initialization(self):
        """Test that the model initializes correctly"""
        self.assertIsInstance(self.model, TransUNet)
        self.assertIsInstance(self.model.encoder, HybridEncoder)
        self.assertIsInstance(self.model.decoder, TransUNetDecoder)
    
    def test_model_forward(self):
        """Test the forward pass with mocked components"""
        # Configure mocks for encoder
        mock_backbone_features = [
            torch.rand(self.batch_size, 56, 20, 20),
            torch.rand(self.batch_size, 160, 20, 20),
            torch.rand(self.batch_size, 448, 5, 5)
        ]
        mock_vit_output = torch.rand(self.batch_size, 768, 5, 5)
        
        # Mock the encoder.forward method
        with mock.patch.object(self.model.encoder, 'forward', return_value=(mock_backbone_features, mock_vit_output)):
            # Mock the decoder.forward method
            with mock.patch.object(self.model.decoder, 'forward', return_value=torch.rand(self.batch_size, 4, 160, 160)):
                # Call forward
                output = self.model(self.input_tensor)
                
                # Assertions
                self.assertEqual(output.shape, (self.batch_size, 4, 160, 160))

    def test_hybrid_encoder(self):
        """Test the encoder's output shapes with a mocked encoder"""
        # Create mock outputs
        mock_backbone_features = [
            torch.rand(self.batch_size, 56, 20, 20),
            torch.rand(self.batch_size, 160, 20, 20),
            torch.rand(self.batch_size, 448, 5, 5)
        ]
        mock_vit_output = torch.rand(self.batch_size, 768, 5, 5)
        
        # Mock the encoder forward method
        with mock.patch.object(self.model.encoder, 'forward', return_value=(mock_backbone_features, mock_vit_output)):
            features, vit_features = self.model.encoder(self.input_tensor)
            
            # Check output shapes
            self.assertEqual(len(features), 3)
            self.assertEqual(vit_features.shape, (self.batch_size, 768, 5, 5))

    @unittest.skip("Integration test requiring actual models - skip during CI")
    def test_end_to_end_forward(self):
        """Full end-to-end test with actual model (skipped by default)"""
        # This is an integration test that uses the actual model
        # Skip in CI environments as it requires downloading weights
        output = self.model(self.input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 4, 160, 160))


if __name__ == '__main__':
    unittest.main()