"""TransUNet model architecture for brain tumor segmentation.

This module implements the complete TransUNet model, which combines a
convolutional neural network (CNN) encoder with a Vision Transformer (ViT)
and a U-Net-style decoder.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchviz import make_dot

class HybridEncoder(nn.Module):
    
    """A hybrid CNN-Transformer encoder.

    Uses an EfficientNet-B4 as the CNN backbone to extract hierarchical feature
    maps and a Vision Transformer (ViT) to process the highest-level feature
    map, capturing global context.
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=True,
            features_only=True,
            out_indices=[2, 3, 4],
            in_chans=3, 
            act_layer=partial(nn.SiLU, inplace=False)  # Corrected
)

        # Override SqueezeExcite activations to ensure inplace=False
        for module in self.backbone.modules():
            if module.__class__.__name__ == "SqueezeExcite":
                module.act1 = nn.SiLU(inplace=False)
        
        self._modify_first_conv()
        
        self.feature_channels = self.backbone.feature_info.channels()
        
        # Step 3: Initialize ViT
        self.vit_input_size = 7 # Define the target grid size explicitly
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            in_chans=self.feature_channels[-1], 
            img_size=self.vit_input_size,      
            patch_size=1,                     
            num_classes=0                 
        )
        self._adjust_pos_embeddings()
        self.grad_norms = {} # Initialize the dict for storing norms 
        self._register_gradient_hooks() 

    def _modify_first_conv(self):
        """Replace first conv to handle 4 channels using pretrained weights"""
        old_conv = self.backbone.conv_stem
        
        
        new_conv = nn.Conv2d(
            4,  
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        
        with torch.no_grad():
            # Copy first 3 channels from pretrained weights
            new_conv.weight[:, :3] = old_conv.weight.clone()
            
            # Initialize 4th channel as mean of RGB channels
            new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
        
        # Replace the backbone's first conv layer
        self.backbone.conv_stem = new_conv

    def _adjust_pos_embeddings(self):
        # Remove the class token.
        patch_pos_embed = self.vit.pos_embed[:, 1:]

        # Compute the original grid size from the number of tokens.
        n_patches_orig = patch_pos_embed.shape[1]
        orig_grid_size = int(math.sqrt(n_patches_orig))

        patch_pos_embed = patch_pos_embed.reshape(1, orig_grid_size, orig_grid_size, -1).permute(0, 3, 1, 2)

        # Interpolate to match the target grid size
        target_grid_size = self.vit_input_size 
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(target_grid_size, target_grid_size), 
            mode="bicubic",
            align_corners=False
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)

        # Update the vit model's parameters
        self.vit.pos_embed = nn.Parameter(patch_pos_embed)
        self.vit.cls_token = None 

    def _register_gradient_hooks(self):
        """CHANGE 7: Add hooks to monitor gradient flow"""
        def _backward_hook(module, grad_input, grad_output):
            name = str(module.__class__).split(".")[-1].split("'")[0]
            self.grad_norms[name] = grad_output[0].abs().mean().item()

        # Monitor critical layers
        for name, layer in self.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.register_full_backward_hook(_backward_hook)

    def forward(self, x):
        
        # Step 1: Extract backbone features
        backbone_features = self.backbone(x)
    
        # Step 2: ViT processing
        cnn_feature = backbone_features[-1]   
    
        # Step 3: Feed through ViT
        vit_output = self.vit.forward_features(cnn_feature)
    
        # Step 4: Reshape ViT output back to spatial grid
        B, N, C = vit_output.shape   
        H = W = self.vit_input_size  
        vit_output = vit_output.permute(0, 2, 1).reshape(B, C, H, W) 
    
        # Step 5: Return both backbone features and ViT features
        return backbone_features, vit_output
    

class AttentionGate(nn.Module):
    
    """Attention Gate (AG) for filtering features in skip connections.

    Implements the attention mechanism described in "Attention U-Net" to
    suppress irrelevant regions in the skip connection feature maps while
    highlighting salient features useful for the specific task.
    """
    
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.GroupNorm(4, F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.GroupNorm(4, F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.Sigmoid()
        )
        self.res_conv = nn.Conv2d(F_l, F_l, 1) if F_g != F_l else nn.Identity()
        nn.init.constant_(self.psi[0].weight, 0)
        nn.init.constant_(self.psi[0].bias, 0)

    def forward(self, g, x):
        
        if g.shape[-2:] != x.shape[-2:]:
            x = F.interpolate(x, size=g.shape[-2:], mode='bilinear', align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return self.res_conv(x) * psi + x
    

class FinalUpsampleBlock(nn.Module):
    
    """A simple upsampling block used in the final stages of the decoder."""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.GELU()
        )
    
    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)
    

class DecoderBlock(nn.Module):
    
    """A decoder block for the U-Net architecture.

    This block takes features from the previous decoder layer and a skip
    connection from the encoder. It upsamples the features, uses an
    Attention Gate on the skip connection, concatenates them, and passes
    them through convolutional layers.
    """
    
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # CHANGE 1: Use transposed conv for learnable upsampling
        self.upsample = nn.ConvTranspose2d(
            in_ch, in_ch, kernel_size=2, stride=2
        )
        # CHANGE 2: Simplified attention gate
        self.attn = AttentionGate(in_ch, skip_ch, in_ch//2)
        
        # CHANGE 3: Channel matching convs
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_ch, in_ch, 1),
            nn.GroupNorm(4, in_ch)
        )
        
        # CHANGE 4: Bottleneck-style processing
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch*2, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.GELU()
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        
        # CHANGE 5: Let AttentionGate handle spatial alignment
        attn_skip = self.attn(x, skip)
        attn_skip = self.skip_conv(attn_skip)
        
        # CHANGE 6: Concatenate -> dense fusion
        x = torch.cat([x, attn_skip], dim=1)
        return self.conv(x)

class TransUNetDecoder(nn.Module):
    
    """The decoder part of the TransUNet model.

    This module reconstructs the segmentation mask from the features provided
    by the HybridEncoder. It uses several DecoderBlocks and FinalUpsampleBlocks
    to progressively increase the spatial resolution back to the original
    image size.
    """
    
    def __init__(self, feature_channels, num_classes=4, dropout_rate=0.3):
        super().__init__()
        
        vit_channels = 768 # From ViT
        
        # decoder3 takes the ViT output (upsampled to 14x14) and the 14x14 skip connection (160 ch).
        self.decoder3 = DecoderBlock(vit_channels, feature_channels[1], 256)
        # decoder2 takes the output of decoder3 (upsampled to 28x28) and the 28x28 skip connection (56 ch).
        self.decoder2 = DecoderBlock(256, feature_channels[0], 128)          

        # The output of decoder2 will be at 28x28 resolution.
        # We need to upsample from 28x28 to the target 224x224.
        # 28 -> 56 -> 112 -> 224. This requires 3 upsampling steps.
        self.upsample_1 = FinalUpsampleBlock(128, 64) # From 28x28 to 56x56
        self.upsample_2 = FinalUpsampleBlock(64, 32)  # From 56x56 to 112x112
        self.upsample_3 = FinalUpsampleBlock(32, 16)  # From 112x112 to 224x224

        # Final convolution to map to class channels
        self.final_conv = nn.Sequential(
            nn.Dropout2d(dropout_rate, inplace=False),
            nn.Conv2d(16, num_classes, kernel_size=3, padding=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, backbone_features, vit_output):
       
        # Unpack the skip connections from the encoder
        s3_skip, s4_skip, _ = backbone_features # We don't use the 3rd feature (s5) as a skip

        # Pass features with clear names
        x = self.decoder3(vit_output, s4_skip) # decoder3 uses skip from stage 4 (14x14 -> 28x28)
        x = self.decoder2(x, s3_skip)   
        
        # Pass through the final upsampling chain
        x = self.upsample_1(x)
        x = self.upsample_2(x)
        x = self.upsample_3(x)
        
        # Final prediction map
        return self.final_conv(x)
    
    
class TransUNet(nn.Module):
    
    """The main TransUNet model.

    This class encapsulates the entire architecture by connecting the
    HybridEncoder and the TransUNetDecoder. It also includes methods for
    validating the internal shapes and visualizing gradients.

    Args:
        None

    Methods:
        forward(x): Performs a forward pass through the network.
    """
    
    def __init__(self):
        super().__init__()
        self.encoder = HybridEncoder()
        self.decoder = TransUNetDecoder(self.encoder.feature_channels)
        
        # CHANGE 1: Gradient monitoring
        self.gradients = {}
        self._register_hooks()
        self.expected_features = self.encoder.feature_channels
        
        self.expected_vit_channels = self.encoder.vit.embed_dim # Programmatically get 768
        self.expected_vit_spatial_dim = self.encoder.vit_input_size
        self.expected_vit_shape = (
            self.expected_vit_channels,
            self.expected_vit_spatial_dim,
            self.expected_vit_spatial_dim
        )

    def _register_hooks(self):
        """Monitor gradients using torchviz-compatible hooks"""
        def _save_grad(name):
            def hook(grad):
                self.gradients[name] = grad.detach()
            return hook
    
        # For parameters, use register_hook
        self.encoder.vit.pos_embed.register_hook(_save_grad('vit_pos_embed'))
        
        # For modules, you can use register_full_backward_hook
        self.decoder.decoder3.attn.psi[0].register_full_backward_hook(
            lambda module, grad_in, grad_out: self.gradients.update({'decoder_attn': grad_out[0].detach()})
        )

    def forward(self, x):
        # Forward pass with validation
        backbone_features, vit_output = self.encoder(x)
        
        # CHANGE 3: Architecture validation
        self._validate_shapes(backbone_features, vit_output)
        
        return self.decoder(backbone_features, vit_output)

    def _validate_shapes(self, backbone_features, vit_output):
        """Ensure encoder-decoder shape compatibility"""
        # Validate ViT output
        assert vit_output.shape[1:] == self.expected_vit_shape, \
            f"ViT output {vit_output.shape[1:]} ≠ expected {self.expected_vit_shape}"
            
        # Validate encoder features
        for i, (feat, exp) in enumerate(zip(backbone_features, self.expected_features)):
            assert feat.shape[1] == exp, \
                f"Encoder layer {i} channels {feat.shape[1]} ≠ expected {exp}"

    def visualize_gradients(self, x):
        """Generate torchviz gradient graph"""
        pred = self(x)
        make_dot(pred.mean(), params=dict(self.named_parameters())).render("grad_graph", format="png")
        
    def get_gradient_norms(self):
        """Return gradient L2 norms for monitoring"""
        return {
            'encoder': self.encoder.grad_norms,
            'decoder': self.decoder.grad_norms
        }
