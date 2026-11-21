"""
CropGuardNet — Core Architecture
=================================
A unified lightweight CNN–Transformer framework with multi-scale attention
and crop-aware conditioning for cross-crop leaf disease detection.

Architecture:
    Input (224×224×3)
      → Preprocessing (Rescale)
      → Lightweight Backbone (MobileNetV3Small alpha=0.50)
      → Multi-Scale Feature Pyramid (3 scales)
      → Depthwise Separable Refinement Block
      → Squeeze-and-Excitation Channel Attention
      → Compact Transformer Encoder (2 heads, 1 layer)
      → CNN–Transformer Feature Fusion
      → [Optional] Crop-Aware Conditioning
      → Softmax Classifier (N classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ══════════════════════════════════════════
# Building Blocks
# ══════════════════════════════════════════

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (as in MangoXNet).

    Decomposes a standard convolution into:
    1. Depthwise conv: one filter per input channel (spatial filtering)
    2. Pointwise conv: 1×1 conv to combine channels
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SqueezeAndExcitation(nn.Module):
    """Squeeze-and-Excitation block (as in MangoXNet).

    Recalibrates channel-wise feature responses by:
    1. Squeeze: Global average pooling → channel descriptor
    2. Excitation: Two FC layers → channel attention weights
    3. Scale: Multiply input features by attention weights
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=False),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class CompactTransformerEncoder(nn.Module):
    """Compact Transformer encoder (inspired by MangoXNet).

    Converts spatial feature maps into a sequence of patch tokens,
    applies multi-head self-attention, and returns enriched features.
    """
    def __init__(self, embed_dim: int, num_heads: int = 2,
                 num_layers: int = 1, mlp_ratio: float = 2.0,
                 dropout: float = 0.1, num_tokens: int = 49):
        super().__init__()
        self.embed_dim = embed_dim

        # Positional encoding (learnable, pre-registered for optimizer & state_dict)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_tokens, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            Transformer-encoded features (B, C, H, W)
        """
        B, C, H, W = x.shape
        # Reshape to sequence: (B, H*W, C)
        seq = x.flatten(2).transpose(1, 2)  # (B, N, C) where N = H*W

        # Add positional encoding (interpolate if spatial token count differs)
        pos = self.pos_embed
        if seq.shape[1] != pos.shape[1]:
            pos = F.interpolate(
                pos.transpose(1, 2), size=seq.shape[1],
                mode='linear', align_corners=False
            ).transpose(1, 2)
        seq = seq + pos

        # Transformer encoding
        seq = self.transformer(seq)
        seq = self.norm(seq)

        # Reshape back to spatial: (B, C, H, W)
        out = seq.transpose(1, 2).view(B, C, H, W)
        return out


class MultiScaleFeaturePyramid(nn.Module):
    """Multi-Scale Feature Pyramid (NOVEL component).

    Extracts features at 3 spatial scales from the backbone output
    to capture both fine-grained lesion details and large-area symptoms.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Fine scale: 1×1 conv (preserves spatial resolution)
        self.fine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=False)
        )
        # Medium scale: 3×3 conv with dilation
        self.medium = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=False)
        )
        # Coarse scale: 5×5 effective receptive field via 3×3 + dilation
        self.coarse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=False)
        )
        # Fusion: combine all scales
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_fine   = self.fine(x)
        f_medium = self.medium(x)
        f_coarse = self.coarse(x)
        fused = torch.cat([f_fine, f_medium, f_coarse], dim=1)
        return self.fuse(fused)


class CropAwareConditioning(nn.Module):
    """Crop-Aware Conditioning module (NOVEL component).

    Learns a crop-type embedding and fuses it with the disease features
    so the classifier can learn crop-specific disease patterns.

    In multi-crop training, the crop type is known for each sample.
    The embedding is broadcast and concatenated with the feature vector.
    """
    def __init__(self, num_crops: int, embed_dim: int, feature_dim: int):
        super().__init__()
        self.crop_embedding = nn.Embedding(num_crops, embed_dim)
        self.fuse = nn.Sequential(
            nn.Linear(feature_dim + embed_dim, feature_dim),
            nn.SiLU(inplace=False),
            nn.Dropout(0.1)
        )

    def forward(self, features: torch.Tensor,
                crop_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) pooled feature vector
            crop_ids: (B,) integer crop type labels
        Returns:
            Conditioned features (B, D)
        """
        crop_emb = self.crop_embedding(crop_ids)  # (B, embed_dim)
        combined = torch.cat([features, crop_emb], dim=1)
        return self.fuse(combined)


# ══════════════════════════════════════════
# CropGuardNet
# ══════════════════════════════════════════

class CropGuardNet(nn.Module):
    """
    CropGuardNet: Unified Lightweight CNN–Transformer for Cross-Crop
    Leaf Disease Detection.

    Architecture flow:
        Backbone → Multi-Scale Pyramid → DSR → SE → Transformer → Fusion
        → [Crop Conditioning] → Classifier

    Args:
        num_classes: Number of disease classes
        num_crops: Number of crop types (for conditioning)
        backbone: Backbone model name from timm
        pretrained: Whether to use pretrained backbone weights
        feature_dim: Internal feature dimension after pyramid
        use_multi_scale: Enable multi-scale feature pyramid
        use_se: Enable squeeze-and-excitation attention
        use_transformer: Enable compact Transformer encoder
        use_crop_conditioning: Enable crop-aware conditioning
        transformer_heads: Number of attention heads in Transformer
        transformer_layers: Number of Transformer encoder layers
        se_reduction: SE block channel reduction ratio
        dropout: Dropout rate for classifier
    """
    def __init__(
        self,
        num_classes: int = 38,
        num_crops: int = 8,
        backbone: str = "mobilenetv3_small_050",
        pretrained: bool = True,
        feature_dim: int = 32,
        use_multi_scale: bool = True,
        use_se: bool = True,
        use_transformer: bool = True,
        use_crop_conditioning: bool = True,
        use_dual_fusion: bool = True,
        transformer_heads: int = 2,
        transformer_layers: int = 1,
        se_reduction: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_multi_scale = use_multi_scale
        self.use_se = use_se
        self.use_transformer = use_transformer
        self.use_crop_conditioning = use_crop_conditioning
        self.use_dual_fusion = use_dual_fusion

        # ── 1. Backbone ──
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True
        )
        # Get output channels and spatial token count from the last feature level
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feats = self.backbone(dummy)
            backbone_channels = feats[-1].shape[1]
            num_tokens = feats[-1].shape[2] * feats[-1].shape[3]
        print(f"[CropGuardNet] Backbone '{backbone}' → {backbone_channels} channels")

        # ── 2. Multi-Scale Feature Pyramid ──
        if use_multi_scale:
            self.multi_scale = MultiScaleFeaturePyramid(
                backbone_channels, feature_dim
            )
        else:
            self.multi_scale = nn.Sequential(
                nn.Conv2d(backbone_channels, feature_dim, 1, bias=False),
                nn.BatchNorm2d(feature_dim),
                nn.SiLU(inplace=False)
            )

        # ── 3. Depthwise Separable Refinement ──
        self.dsr = DepthwiseSeparableConv(feature_dim, feature_dim)

        # ── 4. Squeeze-and-Excitation ──
        if use_se:
            self.se = SqueezeAndExcitation(feature_dim, se_reduction)

        # ── 5. Compact Transformer Encoder ──
        if use_transformer:
            self.transformer = CompactTransformerEncoder(
                embed_dim=feature_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout,
                num_tokens=num_tokens,
            )

        # ── 6. CNN–Transformer Feature Fusion ──
        if use_transformer and use_dual_fusion:
            self.cnn_gap = nn.AdaptiveAvgPool2d(1)
            self.trans_gap = nn.AdaptiveAvgPool2d(1)
            classifier_in = feature_dim * 2  # CNN + Transformer branches
        elif use_transformer:
            self.trans_gap = nn.AdaptiveAvgPool2d(1)
            classifier_in = feature_dim
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
            classifier_in = feature_dim

        # ── 7. Crop-Aware Conditioning ──
        if use_crop_conditioning:
            crop_embed_dim = 16
            self.crop_cond = CropAwareConditioning(
                num_crops, crop_embed_dim, classifier_in
            )

        # ── 8. Classifier Head ──
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 128),
            nn.SiLU(inplace=False),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # Store for XAI hooks
        self._feature_maps = {}

    def _register_hooks(self):
        """Register forward hooks to capture feature maps for XAI."""
        def hook_fn(name):
            def hook(module, input, output):
                self._feature_maps[name] = output
            return hook

        self.dsr.register_forward_hook(hook_fn('dsr'))
        if self.use_se:
            self.se.register_forward_hook(hook_fn('se'))
        if self.use_transformer:
            self.transformer.register_forward_hook(hook_fn('transformer'))

    def forward(self, x: torch.Tensor,
                crop_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, 224, 224)
            crop_ids: Crop type labels (B,) — required if use_crop_conditioning=True
        Returns:
            Class logits (B, num_classes)
        """
        # Backbone feature extraction
        feats = self.backbone(x)
        x = feats[-1]  # Use last (highest-level) feature map

        # Multi-scale feature pyramid
        x = self.multi_scale(x)

        # Depthwise separable refinement
        x_cnn = self.dsr(x)

        # Squeeze-and-excitation
        if self.use_se:
            x_cnn = self.se(x_cnn)

        # CNN–Transformer fusion
        if self.use_transformer and self.use_dual_fusion:
            x_trans = self.transformer(x_cnn)

            # Dual-path pooling
            cnn_pooled = self.cnn_gap(x_cnn).flatten(1)     # (B, D)
            trans_pooled = self.trans_gap(x_trans).flatten(1)  # (B, D)
            features = torch.cat([cnn_pooled, trans_pooled], dim=1)  # (B, 2D)
        elif self.use_transformer:
            x_trans = self.transformer(x_cnn)
            features = self.trans_gap(x_trans).flatten(1)
        else:
            features = self.gap(x_cnn).flatten(1)

        # Crop-aware conditioning
        if self.use_crop_conditioning and crop_ids is not None:
            features = self.crop_cond(features, crop_ids)

        # Classification
        logits = self.classifier(features)
        return logits

    def count_parameters(self) -> dict:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "size_mb": total * 4 / (1024 ** 2)  # float32
        }


# ══════════════════════════════════════════
# LeafXNet Baseline (Reproduced)
# ══════════════════════════════════════════

class LeafXNet(nn.Module):
    """Reproduced LeafXNet architecture (Al Masud et al., 2026).

    Lightweight CNN with Multi-Head Self-Attention.
    ~79,717 parameters.

    Architecture:
        Conv2d(64) → MaxPool → Conv2d(64) → MaxPool →
        Conv2d(32) → MaxPool → Conv2d(32) → MaxPool →
        Reshape → MultiHeadAttention → Reshape →
        GAP → Dense(128) → Dense(num_classes)
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224→222→111
            nn.Conv2d(3, 64, 3), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            # Block 2: 111→109→54
            nn.Conv2d(64, 64, 3), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            # Block 3: 54→52→26
            nn.Conv2d(64, 32, 3), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            # Block 4: 26→24→12
            nn.Conv2d(32, 32, 3), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        # Multi-head attention: reshape (12,12,32) → (144, 32)
        self.attention = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, batch_first=True
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor,
                crop_ids: torch.Tensor = None) -> torch.Tensor:
        x = self.features(x)  # (B, 32, 12, 12)
        B, C, H, W = x.shape

        # Reshape for attention: (B, H*W, C)
        seq = x.flatten(2).transpose(1, 2)  # (B, 144, 32)
        attn_out, _ = self.attention(seq, seq, seq)

        # Reshape back
        x = attn_out.transpose(1, 2).view(B, C, H, W)

        # GAP + classifier
        x = self.gap(x).flatten(1)
        return self.classifier(x)


# ══════════════════════════════════════════
# MangoXNet Baseline (Reproduced)
# ══════════════════════════════════════════

class MangoXNet(nn.Module):
    """Reproduced MangoXNet architecture (Subramanian et al., 2026).

    Lightweight hybrid CNN–Transformer with MobileNetV3Small backbone.
    ~4.43M parameters (4,426,374 params at 38 classes).

    Architecture:
        MobileNetV3Small → DSR → SE → Transformer → CNN–Transformer Fusion
        → Dense(Swish) → BN → Dropout → Softmax
    """
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        # Backbone
        self.backbone = timm.create_model(
            "mobilenetv3_small_100", pretrained=pretrained, features_only=True
        )
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feats = self.backbone(dummy)
            backbone_ch = feats[-1].shape[1]
            num_tokens = feats[-1].shape[2] * feats[-1].shape[3]

        # DSR
        self.dsr = DepthwiseSeparableConv(backbone_ch, backbone_ch)

        # SE
        self.se = SqueezeAndExcitation(backbone_ch, reduction=4)

        # Compact Transformer
        self.transformer = CompactTransformerEncoder(
            embed_dim=backbone_ch, num_heads=2, num_layers=1, num_tokens=num_tokens
        )

        # Fusion
        self.cnn_gap = nn.AdaptiveAvgPool2d(1)
        self.trans_gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        fused_dim = backbone_ch * 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.SiLU(inplace=False),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor,
                crop_ids: torch.Tensor = None) -> torch.Tensor:
        feats = self.backbone(x)
        x = feats[-1]

        x_cnn = self.dsr(x)
        x_cnn = self.se(x_cnn)

        x_trans = self.transformer(x_cnn)

        cnn_pooled = self.cnn_gap(x_cnn).flatten(1)
        trans_pooled = self.trans_gap(x_trans).flatten(1)
        features = torch.cat([cnn_pooled, trans_pooled], dim=1)

        return self.classifier(features)


# ══════════════════════════════════════════
# Model Factory
# ══════════════════════════════════════════

def build_model(model_name: str, num_classes: int,
                num_crops: int = 8, **kwargs) -> nn.Module:
    """Factory to build any model by name."""
    models = {
        "cropguardnet": lambda: CropGuardNet(
            num_classes=num_classes, num_crops=num_crops, **kwargs
        ),
        "leafxnet": lambda: LeafXNet(num_classes=num_classes),
        "mangoxnet": lambda: MangoXNet(num_classes=num_classes),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name]()


# ══════════════════════════════════════════
# Quick Test
# ══════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CropGuardNet Architecture Test")
    print("=" * 60)

    # Test CropGuardNet
    model = CropGuardNet(num_classes=38, num_crops=8)
    params = model.count_parameters()
    print(f"\nCropGuardNet Parameters:")
    print(f"  Total:     {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Size:      {params['size_mb']:.2f} MB")

    x = torch.randn(2, 3, 224, 224)
    crop_ids = torch.tensor([0, 3])
    out = model(x, crop_ids)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")

    # Test LeafXNet
    print(f"\n{'='*60}")
    leafxnet = LeafXNet(num_classes=38)
    lp = sum(p.numel() for p in leafxnet.parameters())
    print(f"LeafXNet Parameters: {lp:,}")
    out = leafxnet(x)
    print(f"  Output: {out.shape}")

    # Test MangoXNet
    print(f"\n{'='*60}")
    mangoxnet = MangoXNet(num_classes=38)
    mp = sum(p.numel() for p in mangoxnet.parameters())
    print(f"MangoXNet Parameters: {mp:,}")
    out = mangoxnet(x)
    print(f"  Output: {out.shape}")
