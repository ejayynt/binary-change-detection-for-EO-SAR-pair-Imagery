"""
Asymmetric Pseudo-Siamese U-Net for EO-SAR Binary Change Detection.

Architecture overview
---------------------
* EO Encoder  : ResNet-34 pre-trained on ImageNet (3-channel RGB input).
* SAR Encoder : ResNet-34 trained from scratch (1-channel grayscale input).
* Bottleneck  : Concatenation of both deepest feature maps (1024-ch)
                followed by a 3×3 conv reducing to 512-ch.
* Decoder     : Four upsampling stages with tri-partite skip connections
                (upsampled features + EO skip + SAR skip) fused at each level.
* Head        : 1×1 conv → raw logit map (no sigmoid — use BCEWithLogitsLoss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    """
    Standard U-Net decoder block.

    Upsample by 2× (transposed conv), concatenate skip connections from
    both encoders, then apply two 3×3 convolutions with BN + ReLU.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        Args:
            in_channels   : Channels of the incoming feature map (from previous decoder stage).
            skip_channels : Channels of *each* skip connection (EO and SAR share the same depth).
            out_channels  : Output channels after this block.
        """
        super().__init__()
        up_ch = in_channels // 2
        self.up    = nn.ConvTranspose2d(in_channels, up_ch, kernel_size=2, stride=2)
        concat_ch  = up_ch + 2 * skip_channels          # upsampled + eo_skip + sar_skip
        self.conv1 = nn.Conv2d(concat_ch, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor,
                skip_eo: torch.Tensor,
                skip_sar: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against off-by-one from odd spatial dimensions
        if x.shape != skip_eo.shape:
            x = F.interpolate(x, size=skip_eo.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_eo, skip_sar], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class PseudoSiameseUNet(nn.Module):
    """
    Asymmetric dual-stream U-Net for heterogeneous EO-SAR change detection.
    Returns raw logits (single channel). Apply sigmoid + threshold at inference.
    """

    def __init__(self):
        super().__init__()

        # ── 1. EO Encoder (ImageNet pre-trained) ─────────────────────
        resnet_eo = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.eo_init   = nn.Sequential(resnet_eo.conv1, resnet_eo.bn1, resnet_eo.relu)  # → 64ch, H/2
        self.eo_pool   = resnet_eo.maxpool                                               # → 64ch, H/4
        self.eo_layer1 = resnet_eo.layer1   # → 64ch,  H/4
        self.eo_layer2 = resnet_eo.layer2   # → 128ch, H/8
        self.eo_layer3 = resnet_eo.layer3   # → 256ch, H/16
        self.eo_layer4 = resnet_eo.layer4   # → 512ch, H/32

        # ── 2. SAR Encoder (random init, 1-channel stem) ──────────────
        resnet_sar = models.resnet34(weights=None)
        resnet_sar.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.sar_init   = nn.Sequential(resnet_sar.conv1, resnet_sar.bn1, resnet_sar.relu)
        self.sar_pool   = resnet_sar.maxpool
        self.sar_layer1 = resnet_sar.layer1
        self.sar_layer2 = resnet_sar.layer2
        self.sar_layer3 = resnet_sar.layer3
        self.sar_layer4 = resnet_sar.layer4

        # ── 3. Fusion Bottleneck ──────────────────────────────────────
        # EO layer4 (512) + SAR layer4 (512) = 1024 → 512
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ── 4. Decoder (tri-partite skip connections) ─────────────────
        # Channels at each ResNet stage: [64, 64, 128, 256, 512]
        #   init → 64, layer1 → 64, layer2 → 128, layer3 → 256, layer4 → 512
        self.dec4 = DecoderBlock(512, 256, 256)    # bottleneck(512) → 256
        self.dec3 = DecoderBlock(256, 128, 128)    # dec4(256)       → 128
        self.dec2 = DecoderBlock(128,  64,  64)    # dec3(128)       → 64
        self.dec1 = DecoderBlock( 64,  64,  32)    # dec2(64)        → 32  (uses init skips)

        # ── 5. Classification Head ────────────────────────────────────
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    # ------------------------------------------------------------------
    def forward(self, eo: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eo  : (B, 3, H, W)  — normalised EO (optical) image
            sar : (B, 1, H, W)  — normalised SAR image
        Returns:
            logits : (B, 1, H, W) — raw (pre-sigmoid) predictions
        """
        # ── EO encoder path ───────────────────────────────────────────
        eo_s0 = self.eo_init(eo)          # H/2,  64ch  ← first skip
        eo_p  = self.eo_pool(eo_s0)       # H/4,  64ch
        eo_s1 = self.eo_layer1(eo_p)      # H/4,  64ch
        eo_s2 = self.eo_layer2(eo_s1)     # H/8,  128ch
        eo_s3 = self.eo_layer3(eo_s2)     # H/16, 256ch
        eo_s4 = self.eo_layer4(eo_s3)     # H/32, 512ch

        # ── SAR encoder path ──────────────────────────────────────────
        sar_s0 = self.sar_init(sar)
        sar_p  = self.sar_pool(sar_s0)
        sar_s1 = self.sar_layer1(sar_p)
        sar_s2 = self.sar_layer2(sar_s1)
        sar_s3 = self.sar_layer3(sar_s2)
        sar_s4 = self.sar_layer4(sar_s3)

        # ── Fusion bottleneck ─────────────────────────────────────────
        fused = self.bottleneck(torch.cat([eo_s4, sar_s4], dim=1))  # H/32, 512ch

        # ── Decoder with skip connections ─────────────────────────────
        d4 = self.dec4(fused, eo_s3, sar_s3)   # H/16, 256ch
        d3 = self.dec3(d4,   eo_s2, sar_s2)    # H/8,  128ch
        d2 = self.dec2(d3,   eo_s1, sar_s1)    # H/4,   64ch
        d1 = self.dec1(d2,   eo_s0, sar_s0)    # H/2,   32ch

        # ── Head + bilinear upsample to input resolution ──────────────
        logits = self.head(d1)                  # H/2,   1ch
        logits = F.interpolate(logits, size=(eo.shape[2], eo.shape[3]),
                               mode='bilinear', align_corners=False)  # H, W
        return logits
