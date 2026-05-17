# ============================================================================
# breastnet3d_v6_updates.py
# Copy each "## CELL" block into a new notebook cell in breastnet3d_v6.ipynb
# Insert these AFTER the existing "Model Definitions" cell (Cell 2)
# and BEFORE the "Training" cell (Cell 3).
# ============================================================================

# ════════════════════════════════════════════════════════════════════════════
# ## CELL A — New U-Net Model with Skip Connections
# ════════════════════════════════════════════════════════════════════════════
# Paste this as a NEW CODE CELL right after the existing model definitions cell.

"""BreastNet3D_UNet — Encoder-Decoder with 2D→3D skip connections."""
import math, torch, torch.nn as nn, torch.nn.functional as F

# ── Skip Projection: bridges 2D encoder features to 3D decoder space ──
class SkipProjection(nn.Module):
    """Projects a 2D feature map (B,C,H,W) to 3D (B,out_ch,D,H,W)."""
    def __init__(self, in_ch, out_ch, depth):
        super().__init__()
        self.depth = depth
        self.proj = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, feat_2d):
        feat_3d = feat_2d.unsqueeze(2).expand(-1, -1, self.depth, -1, -1)
        return self.proj(feat_3d)

# ── Decoder block with skip concatenation ──
class DecoderBlock3D(nn.Module):
    """Upsample + concat skip + double conv."""
    def __init__(self, in_ch, skip_ch, out_ch, drop=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(True),
            nn.Dropout3d(drop) if drop > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(True),
        )
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.block(x)

# ── Combined U-Net Model ──
# Architecture: 6-stage 2D encoder → 1000-d bottleneck → 6-stage 3D decoder
# Skip connections on enc2–enc5 (4 levels). enc1 skipped (128³ too large),
# enc6 skipped (too close to bottleneck).
#
# Skip channel budget (memory-safe for 128³):
#   enc5 (512ch, 8²)  → proj5 → 32ch @ 8³   → cat with dec stage 2
#   enc4 (256ch, 16²) → proj4 → 16ch @ 16³  → cat with dec stage 3
#   enc3 (128ch, 32²) → proj3 → 8ch  @ 32³  → cat with dec stage 4
#   enc2 (64ch,  64²) → proj2 → 4ch  @ 64³  → cat with dec stage 5

class BreastNet3D_UNet(nn.Module):
    def __init__(self, drop=0.25):
        super().__init__()
        # ── 2D Encoder (same weights as Encoder2D) ──
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv2D(5, 32, 0)
        self.enc2 = DoubleConv2D(32, 64, 0)
        self.enc3 = DoubleConv2D(64, 128, drop)
        self.enc4 = DoubleConv2D(128, 256, drop)
        self.enc5 = DoubleConv2D(256, 512, drop)
        self.enc6 = DoubleConv2D(512, 512, drop)
        self.enc_fc = nn.Sequential(nn.Dropout(drop), nn.Linear(512*2*2, 1000))

        # ── Skip Projections (2D → 3D) ──
        self.proj2 = SkipProjection(64,  4,  depth=64)
        self.proj3 = SkipProjection(128, 8,  depth=32)
        self.proj4 = SkipProjection(256, 16, depth=16)
        self.proj5 = SkipProjection(512, 32, depth=8)

        # ── Bottleneck ──
        self.dec_fc = nn.Linear(1000, 512*2*2*2)

        # ── 3D Decoder with skip injection ──
        # Stage 1: 2³→4³, no skip
        self.up1 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.d1  = DoubleConv3D(256, 256, drop)
        # Stage 2: 4³→8³, skip from enc5 (32ch)
        self.dec2 = DecoderBlock3D(256, 32, 128, drop)
        # Stage 3: 8³→16³, skip from enc4 (16ch)
        self.dec3 = DecoderBlock3D(128, 16, 64, drop)
        # Stage 4: 16³→32³, skip from enc3 (8ch)
        self.dec4 = DecoderBlock3D(64, 8, 32, 0)
        # Stage 5: 32³→64³, skip from enc2 (4ch)
        self.dec5 = DecoderBlock3D(32, 4, 16, 0)
        # Stage 6: 64³→128³, no skip
        self.up6 = nn.ConvTranspose3d(16, 8, 2, stride=2)
        self.d6  = DoubleConv3D(8, 8, 0)

        # ── Visual hull fusion (identical to Decoder3D) ──
        self.fusion = nn.Sequential(
            nn.Conv3d(8 + 1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(8), nn.ReLU(True),
            nn.Conv3d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.apply(_init)
        if self.fusion[3].bias is not None:
            nn.init.constant_(self.fusion[3].bias, -4.0)

    def forward(self, x, visual_hull=None):
        # ── Encoder (store skip features before pool) ──
        s1 = self.enc1(x)              # (B, 32,  128, 128)
        s2 = self.enc2(self.pool(s1))  # (B, 64,   64,  64)
        s3 = self.enc3(self.pool(s2))  # (B, 128,  32,  32)
        s4 = self.enc4(self.pool(s3))  # (B, 256,  16,  16)
        s5 = self.enc5(self.pool(s4))  # (B, 512,   8,   8)
        s6 = self.enc6(self.pool(s5))  # (B, 512,   4,   4)
        z  = self.enc_fc(self.pool(s6).view(x.size(0), -1))  # (B, 1000)

        # ── Bottleneck → 3D seed ──
        x3 = self.dec_fc(z).view(x.size(0), 512, 2, 2, 2)

        # ── Decoder with skip connections ──
        x3 = self.d1(self.up1(x3))                    # (B,256, 4³)
        x3 = self.dec2(x3, self.proj5(s5))             # (B,128, 8³)
        x3 = self.dec3(x3, self.proj4(s4))             # (B, 64,16³)
        x3 = self.dec4(x3, self.proj3(s3))             # (B, 32,32³)
        x3 = self.dec5(x3, self.proj2(s2))             # (B, 16,64³)
        x3 = self.d6(self.up6(x3))                     # (B,  8,128³)

        # ── Visual hull fusion ──
        if visual_hull is None:
            visual_hull = torch.zeros(
                x3.size(0), 1, x3.size(2), x3.size(3), x3.size(4),
                device=x3.device, dtype=x3.dtype)
        x3 = torch.cat([x3, visual_hull], dim=1)
        return self.fusion(x3) * visual_hull


# ════════════════════════════════════════════════════════════════════════════
# ## CELL B — Shape Prior (defined but NOT wired into training)
# ════════════════════════════════════════════════════════════════════════════

def build_dual_ellipsoid_prior(volume_size=128, device='cuda'):
    V = volume_size
    zz, yy, xx = torch.meshgrid(
        torch.linspace(0, 1, V, device=device),
        torch.linspace(0, 1, V, device=device),
        torch.linspace(0, 1, V, device=device),
        indexing='ij')
    cx_L, cx_R = 0.30, 0.70
    cy, cz = 0.55, 0.50
    ax, ay, az = 0.20, 0.35, 0.28
    def ellipsoid_sdf(cx):
        return ((xx-cx)**2/ax**2 + (yy-cy)**2/ay**2 + (zz-cz)**2/az**2)
    e_L = ellipsoid_sdf(cx_L)
    e_R = ellipsoid_sdf(cx_R)
    prior = torch.sigmoid(8.0 * (1.0 - torch.minimum(e_L, e_R)))
    return prior.unsqueeze(0).unsqueeze(0)  # (1,1,V,V,V)

class ShapePriorLoss(nn.Module):
    def __init__(self, volume_size=128, device='cuda', weight=0.3):
        super().__init__()
        self.weight = weight
        prior = build_dual_ellipsoid_prior(volume_size, device).detach()
        self.register_buffer('prior', prior)
    def forward(self, V_pred):
        outside_mass = V_pred * (1.0 - self.prior)
        return self.weight * outside_mass.mean()


# ════════════════════════════════════════════════════════════════════════════
# ## CELL C — Runtime Flags
# ════════════════════════════════════════════════════════════════════════════

USE_SHAPE_PRIOR = False  # set True to activate Upgrade 2


# ════════════════════════════════════════════════════════════════════════════
# ## CELL D — Verification: U-Net Forward Pass
# ════════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_unet = BreastNet3D_UNet().to(device)
dummy = torch.zeros(2, 5, 128, 128, device=device)
# Need a dummy visual hull too
dummy_hull = torch.ones(2, 1, 128, 128, 128, device=device)
with torch.no_grad():
    out = model_unet(dummy, dummy_hull)
assert out.shape == (2, 1, 128, 128, 128), f"Shape mismatch: {out.shape}"
print("U-Net model forward pass OK:", out.shape)
del model_unet, dummy, dummy_hull
torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════
# ## CELL E — Verification: Prior Visualisation
# ════════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
prior_np = build_dual_ellipsoid_prior(128, 'cpu').squeeze().numpy()
plt.imshow(prior_np[64, :, :], cmap='hot', vmin=0, vmax=1)
plt.title('Dual-ellipsoid prior — frontal slice (depth=64)')
plt.colorbar()
plt.show()
