# models/model.py

import math
import torch
import timm
from functools import partial
import torch.nn as nn
import torch.nn.functional as F


class HybridEncoder(nn.Module):
    """Hybrid encoder combining EfficientNet backbone with Vision Transformer.
    
    Attributes:
        backbone (nn.Module): EfficientNet feature extractor
        vit (nn.Module): Vision Transformer module
        feature_channels (list): List of output channels from backbone layers
        vit_input_size (int): Spatial dimension of input to ViT
    """

    def __init__(self):
        """Initialize hybrid encoder with modified EfficientNet and ViT."""
        super().__init__()
        
        # Backbone configuration
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=True,
            features_only=True,
            out_indices=[2, 3, 4],
            in_chans=3,
            act_layer=partial(nn.SiLU, inplace=False)
        )
        
        # Modify first convolution for 4 input channels
        self._modify_first_conv()
        
        # Configure Squeeze-Excite layers
        for module in self.backbone.modules():
            if isinstance(module, timm.layers.SqueezeExcite):
                module.act1 = nn.SiLU(inplace=False)

        # ViT configuration
        self.feature_channels = self.backbone.feature_info.channels()
        self.vit_input_size = 5  # Target grid size for positional embeddings
        
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            in_chans=self.feature_channels[-1],
            img_size=self.vit_input_size,
            patch_size=1,
            num_classes=0)
        
        # Adjust ViT positional embeddings
        self._adjust_pos_embeddings()
        
        # Gradient monitoring
        self.grad_norms = {}
        self._register_gradient_hooks()

    def _modify_first_conv(self):
        """Modify first convolution layer to accept 4 input channels.
        
        Copies weights from pretrained RGB channels and initializes 
        fourth channel with mean of existing weights.
        """
        old_conv = self.backbone.conv_stem
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False)
        
        # Initialize weights
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight.clone()
            new_conv.weight[:, 3] = old_conv.weight.mean(dim=1)
        
        self.backbone.conv_stem = new_conv

    def _adjust_pos_embeddings(self):
        """Adjust ViT positional embeddings for feature map input."""
        patch_pos_embed = self.vit.pos_embed[:, 1:]
        orig_size = int(math.sqrt(patch_pos_embed.shape[1]))
        
        # Reshape and interpolate positional embeddings
        patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, -1)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(self.vit_input_size, self.vit_input_size),
            mode="bicubic",
            align_corners=False)
        
        # Update ViT parameters
        patch_pos_embed = patch_pos_embed.flatten(2).permute(0, 2, 1)
        self.vit.pos_embed = nn.Parameter(patch_pos_embed)
        self.vit.cls_token = None

    def _register_gradient_hooks(self):
        """Register hooks to monitor gradient flow through critical layers."""
        def backward_hook(module, grad_input, grad_output):
            name = module.__class__.__name__
            self.grad_norms[name] = grad_output[0].abs().mean().item()

        for _, layer in self.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.register_full_backward_hook(backward_hook)

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through hybrid encoder.
        
        Args:
            x: Input tensor of shape (B, 4, H, W)
            
        Returns:
            tuple: (backbone_features, vit_output)
                - backbone_features: List of feature maps from EfficientNet
                - vit_output: ViT features of shape (B, 768, 5, 5)
        """
        backbone_features = self.backbone(x)
        vit_features = self.vit.forward_features(backbone_features[-1])
        
        # Reshape ViT output
        B, N, C = vit_features.shape
        vit_output = vit_features.permute(0, 2, 1).reshape(
            B, C, self.vit_input_size, self.vit_input_size)
        
        return backbone_features, vit_output


class AttentionGate(nn.Module):
    """Attention gate with skip connection for feature fusion.
    
    Attributes:
        W_g (nn.Sequential): Processing for gating signal
        W_x (nn.Sequential): Processing for skip connection
        psi (nn.Sequential): Attention computation
        res_conv (nn.Module): Residual connection
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of channels in gating signal
            F_l: Number of channels in skip connection
            F_int: Intermediate channels for attention computation
        """
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(4, F_int))
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(4, F_int))
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.Sigmoid())
        
        self.res_conv = (
            nn.Conv2d(F_l, F_l, kernel_size=1)
            if F_g != F_l else nn.Identity())
        
        # Initialize final conv layer
        nn.init.constant_(self.psi[0].weight, 0)
        nn.init.constant_(self.psi[0].bias, 0)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention modulation.
        
        Args:
            g: Gating signal from decoder
            x: Skip connection from encoder
            
        Returns:
            torch.Tensor: Attention-modulated skip connection
        """
        if g.shape[-2:] != x.shape[-2:]:
            x = F.interpolate(
                x, size=g.shape[-2:], mode='bilinear', align_corners=True)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        
        return self.res_conv(x) * psi + x


class DecoderBlock(nn.Module):
    """Decoder block with attention gate and feature fusion.
    
    Attributes:
        upsample (nn.Module): Transposed convolution for upsampling
        attn (AttentionGate): Attention gate module
        skip_conv (nn.Sequential): Skip connection processing
        conv (nn.Sequential): Feature fusion convolution
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        """
        Args:
            in_ch: Number of input channels
            skip_ch: Number of channels in skip connection
            out_ch: Number of output channels
        """
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_ch, in_ch, kernel_size=2, stride=2)
        
        self.attn = AttentionGate(in_ch, skip_ch, in_ch//2)
        
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_ch, in_ch, kernel_size=1),
            nn.GroupNorm(4, in_ch))
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch*2, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.GELU())

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder block.
        
        Args:
            x: Input features from previous decoder stage
            skip: Skip connection from encoder
            
        Returns:
            torch.Tensor: Processed features at higher resolution
        """
        x = self.upsample(x)
        attn_skip = self.attn(x, skip)
        attn_skip = self.skip_conv(attn_skip)
        x = torch.cat([x, attn_skip], dim=1)
        return self.conv(x)


class TransUNet(nn.Module):
    """TransUNet architecture combining CNN and Transformer features.
    
    Attributes:
        encoder (HybridEncoder): Hybrid CNN-Transformer encoder
        decoder (TransUNetDecoder): Multi-scale decoder with attention
    """

    def __init__(self):
        super().__init__()
        self.encoder = HybridEncoder()
        self.decoder = TransUNetDecoder(self.encoder.feature_channels)
        
        # Gradient monitoring
        self.gradients = {}
        self._register_hooks()
        
        # Architecture validation parameters
        self.expected_vit_shape = (768, 5, 5)
        self.expected_features = [56, 160, 448]

    def _register_hooks(self):
        """Register hooks for gradient visualization."""
        def save_grad(name):
            def hook(grad):
                self.gradients[name] = grad.detach()
            return hook
        
        self.encoder.vit.pos_embed.register_hook(save_grad('vit_pos_embed'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through full TransUNet.
        
        Args:
            x: Input tensor of shape (B, 4, H, W)
            
        Returns:
            torch.Tensor: Output segmentation map (B, 4, H, W)
        """
        backbone_features, vit_output = self.encoder(x)
        self._validate_shapes(backbone_features, vit_output)
        return self.decoder(backbone_features, vit_output)

    def _validate_shapes(self, backbone_features: list, vit_output: torch.Tensor):
        """Validate feature map dimensions.
        
        Args:
            backbone_features: List of feature maps from encoder
            vit_output: Output from ViT module
            
        Raises:
            AssertionError: If feature dimensions don't match expectations
        """
        assert vit_output.shape[1:] == self.expected_vit_shape, \
            f"ViT output shape {vit_output.shape[1:]} != {self.expected_vit_shape}"
            
        for i, (feat, exp) in enumerate(zip(backbone_features, self.expected_features)):
            assert feat.shape[1] == exp, \
                f"Encoder layer {i} channels {feat.shape[1]} != {exp}"


class TransUNetDecoder(nn.Module):
    """TransUNet decoder with progressive upsampling.
    
    Attributes:
        decoder4 (DecoderBlock): First decoder block
        decoder3 (DecoderBlock): Second decoder block
        decoder2 (DecoderBlock): Third decoder block
        upsample1 (FinalUpsampleBlock): First upsampling block
        upsample0 (FinalUpsampleBlock): Second upsampling block
        final_conv (nn.Sequential): Final convolution layer
    """

    def __init__(self, feature_channels: list, num_classes: int = 4):
        """
        Args:
            feature_channels: List of channel counts from encoder
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.decoder4 = DecoderBlock(768, feature_channels[2], 512)
        self.decoder3 = DecoderBlock(512, feature_channels[1], 256)
        self.decoder2 = DecoderBlock(256, feature_channels[0], 128)
        
        self.upsample1 = FinalUpsampleBlock(128, 64)
        self.upsample0 = FinalUpsampleBlock(64, 32)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1))
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, backbone_features: list, vit_output: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.
        
        Args:
            backbone_features: List of feature maps from encoder
            vit_output: Output from ViT module
            
        Returns:
            torch.Tensor: Output segmentation map
        """
        layer2, layer3, layer4 = backbone_features
        
        x = self.decoder4(vit_output, layer4)
        x = self.decoder3(x, layer3)
        x = self.decoder2(x, layer2)
        x = self.upsample1(x)
        x = self.upsample0(x)
        
        return self.final_conv(x)


class FinalUpsampleBlock(nn.Module):
    """Final upsampling block with transposed convolution.
    
    Attributes:
        upsample (nn.Module): Transposed convolution layer
        conv (nn.Sequential): Convolutional processing block
    """

    def __init__(self, in_ch: int, out_ch: int):
        """
        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels
        """
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through upsampling block."""
        x = self.upsample(x)
        return self.conv(x)