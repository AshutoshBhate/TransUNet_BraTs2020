import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from functools import partial

class HybridEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Step 1: Create backbone with in_chans=3 (pretrained weights)
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=True,
            features_only=True,
            out_indices=[2, 3, 4],
            in_chans=3,  # Start with default 3 channels
            act_layer=partial(nn.SiLU, inplace=False)  # Corrected
        )

        # Override SqueezeExcite activations to ensure inplace=False
        for module in self.backbone.modules():
            if module.__class__.__name__ == "SqueezeExcite":
                module.act1 = nn.SiLU(inplace=False)
        
        # Step 2: Modify first conv to handle 4 channels
        self._modify_first_conv()
        
        self.feature_channels = self.backbone.feature_info.channels()
        
        # Step 3: Initialize ViT (unchanged)
        self.vit_input_size = 5 # Define the target grid size explicitly
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            in_chans=self.feature_channels[-1], # Should be 448 for effnetb4 layer 4
            img_size=self.vit_input_size,      # <<< CHANGE: Use the defined size (5)
            patch_size=1,                      # Keep patch_size=1 for dense features
            num_classes=0                      # No classification head
        )
        self._adjust_pos_embeddings()
        self.grad_norms = {} # Initialize the dict for storing norms
        self._register_gradient_hooks()

    def _modify_first_conv(self):
        """Replace first conv to handle 4 channels using pretrained weights"""
        old_conv = self.backbone.conv_stem
        
        # Create new conv layer with 4 input channels
        new_conv = nn.Conv2d(
            4,  # Change to 4 input channels
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        
        # Initialize weights: copy pretrained RGB weights, initialize 4th channel
        with torch.no_grad():
            # Copy first 3 channels from pretrained weights
            new_conv.weight[:, :3] = old_conv.weight.clone()
            
            # Initialize 4th channel as mean of RGB channels
            new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
        
        # Replace the backbone's first conv layer
        self.backbone.conv_stem = new_conv

    def _adjust_pos_embeddings(self):
        """Correct position embedding interpolation dynamically."""
        # Remove the class token.
        patch_pos_embed = self.vit.pos_embed[:, 1:] # Shape: [1, num_patches_orig, embed_dim]

        # Compute the original grid size from the number of tokens.
        n_patches_orig = patch_pos_embed.shape[1]
        orig_grid_size = int(math.sqrt(n_patches_orig)) # Should be 14 for ViT-B/16 224x224

        # Reshape into (1, orig_grid_size, orig_grid_size, -1) then permute.
        patch_pos_embed = patch_pos_embed.reshape(1, orig_grid_size, orig_grid_size, -1).permute(0, 3, 1, 2)
        # Shape: [1, embed_dim, orig_grid_size, orig_grid_size]

        # Interpolate to match the target grid size (MODIFIED)
        target_grid_size = self.vit_input_size # Get the target size (5)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(target_grid_size, target_grid_size), # <<< CHANGE: Interpolate to target size
            mode="bicubic",
            align_corners=False
        )
        # Shape: [1, embed_dim, target_grid_size, target_grid_size]

        # Flatten back to a sequence and update the positional embeddings.
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        # Shape: [1, target_grid_size*target_grid_size, embed_dim]

        # Update the vit model's parameters
        self.vit.pos_embed = nn.Parameter(patch_pos_embed)
        self.vit.cls_token = None # Ensure class token is None if not used

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
        """
        Args:
            x: Input tensor of shape [B, 4, 160, 160]
        
        Returns:
            backbone_features: List of feature maps from EfficientNet [layer2, layer3, layer4]
            vit_output: ViT features reshaped to [B, 768, 5, 5]
        """
        # Step 1: Extract backbone features
        backbone_features = self.backbone(x)
        # backbone_features shapes e.g.: [B,56,20,20], [B,160,20,20], [B,448,5,5]
    
        # Step 2: ViT processing
        cnn_feature = backbone_features[-1]    # Actual shape seems to be [B, 448, 5, 5]
        
        # Step 3: Feed through ViT
        vit_output = self.vit.forward_features(cnn_feature)
        # vit_output shape: [B, 25, 768] since 5x5 grid with patch_size=1
    
        # Step 4: Reshape ViT output back to spatial grid
        B, N, C = vit_output.shape    # [B, 25, 768]
        H = W = self.vit_input_size  # 5
        vit_output = vit_output.permute(0, 2, 1).reshape(B, C, H, W)  # [B, 768, 5, 5]
    
        # Step 5: Return both backbone features and ViT features
        return backbone_features, vit_output


class AttentionGate(nn.Module):
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
        # Ensure x (skip) is upsampled to match g's spatial size
        if g.shape[-2:] != x.shape[-2:]:
            x = F.interpolate(x, size=g.shape[-2:], mode='bilinear', align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return self.res_conv(x) * psi + x


class FinalUpsampleBlock(nn.Module):
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
    def __init__(self, feature_channels, num_classes=4):
        super().__init__()
        # Existing decoder blocks
        self.decoder4 = DecoderBlock(768, feature_channels[2], 512)  # ViT(5x5) -> Layer4(5x5), Out: 10x10
        self.decoder3 = DecoderBlock(512, feature_channels[1], 256)   # -> Layer3(20x20), Out: 20x20
        self.decoder2 = DecoderBlock(256, feature_channels[0], 128)   # -> Layer2(20x20), Out: 40x40

        # Upsampling stages to restore resolution
        # Renamed original final_upsample
        self.upsample1 = FinalUpsampleBlock(128, 64) # Input 40x40 -> Output 80x80

        # <<< NEW: Add another upsampling block >>>
        self.upsample0 = FinalUpsampleBlock(64, 32)  # Input 80x80 -> Output 160x160

        # Final convolution to get class scores
        self.final_conv = nn.Sequential(
            # <<< CHANGE: Input channels from 64 to 32 >>>
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
            # Consider adding a final activation like Softmax if needed AFTER the model output,
            # usually handled by the loss function (like CrossEntropyLoss) or inference pipeline.
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): # Include ConvTranspose2d
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, backbone_features, vit_output):
        # Assumes backbone_features = [layer2, layer3, layer4]
        # where layer shapes are e.g., [B,56,20,20], [B,160,20,20], [B,448,5,5]
        # and vit_output shape is [B, 768, 5, 5]
        layer2, layer3, layer4 = backbone_features

        x = self.decoder4(vit_output, layer4) # Output: [B, 512, 10, 10]
        x = self.decoder3(x, layer3)          # Output: [B, 256, 20, 20]
        x = self.decoder2(x, layer2)          # Output: [B, 128, 40, 40]
        x = self.upsample1(x)                 # Output: [B, 64, 80, 80]

        # <<< NEW: Pass through the added upsampling block >>>
        x = self.upsample0(x)                # Output: [B, 32, 160, 160]

        # Final prediction map
        return self.final_conv(x)            # Output: [B, 4, 160, 160]


class TransUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HybridEncoder()
        self.decoder = TransUNetDecoder(self.encoder.feature_channels)
        
        # CHANGE 1: Gradient monitoring
        self.gradients = {}
        self._register_hooks()
        
        self.expected_vit_channels = 768
        self.expected_vit_spatial_dim = self.encoder.vit_input_size # Get from encoder (5)
        self.expected_vit_shape = (
            self.expected_vit_channels,
            self.expected_vit_spatial_dim,
            self.expected_vit_spatial_dim
        ) # <<< CHANGE: Expected shape is now (768, 5, 5)
        self.expected_features = [56, 160, 448] # Channels from EfficientNet layers

    def _register_hooks(self):
        """Monitor gradients using torchviz-compatible hooks"""
        def _save_grad(name):
            def hook(grad):
                self.gradients[name] = grad.detach()
            return hook
    
        # For parameters, use register_hook
        self.encoder.vit.pos_embed.register_hook(_save_grad('vit_pos_embed'))
        
        # For modules, you can use register_full_backward_hook
        self.decoder.decoder4.attn.psi[0].register_full_backward_hook(
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
        from torchviz import make_dot
        pred = self(x)
        make_dot(pred.mean(), params=dict(self.named_parameters())).render("grad_graph", format="png")
        
    def get_gradient_norms(self):
        """Return gradient L2 norms for monitoring"""
        return {
            'encoder': self.encoder.grad_norms,
            'decoder': self.decoder.grad_norms
        }