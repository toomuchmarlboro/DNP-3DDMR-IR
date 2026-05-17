###############################################################################
# CELL A — New U-Net model + Shape Prior (self-contained)
###############################################################################

"""BreastNet3D_UNet — U-Net skip connections + Shape Prior definitions."""
import math, torch, torch.nn as nn, torch.nn.functional as F

# ── Building blocks (repeated here for self-containment) ─────────────────
class DoubleConv2D(nn.Module):
    def __init__(self, inc, outc, drop=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc), nn.ReLU(True),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(outc, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc), nn.ReLU(True))
    def forward(self, x): return self.block(x)

class DoubleConv3D(nn.Module):
    def __init__(self, inc, outc, drop=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inc, outc, 3, padding=1, bias=False),
            nn.BatchNorm3d(outc), nn.ReLU(True),
            nn.Dropout3d(drop) if drop > 0 else nn.Identity(),
            nn.Conv3d(outc, outc, 3, padding=1, bias=False),
            nn.BatchNorm3d(outc), nn.ReLU(True))
    def forward(self, x): return self.block(x)

def _init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)

# ── Skip Projection: 2D encoder features → 3D decoder space ──────────────
class SkipProjection(nn.Module):
    def __init__(self, in_ch, out_ch, depth):
        super().__init__()
        self.depth = depth
        self.proj = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_2d):
        # feat_2d: (B, C, H, W) → expand to (B, C, depth, H, W)
        feat_3d = feat_2d.unsqueeze(2).expand(-1, -1, self.depth, -1, -1)
        return self.proj(feat_3d)


# ── Decoder block with optional skip concatenation ───────────────────────
class DecoderBlock3D(nn.Module):
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


# ── BreastNet3D_UNet ─────────────────────────────────────────────────────
# 6-stage 2D encoder → 1000-d bottleneck → 6-stage 3D decoder
# Skip connections on enc2–enc5 (4 levels):
#   enc5 (512ch, 8²)  → 32ch @ 8³  → dec stage 2
#   enc4 (256ch,16²)  → 16ch @16³  → dec stage 3
#   enc3 (128ch,32²)  →  8ch @32³  → dec stage 4
#   enc2 ( 64ch,64²)  →  4ch @64³  → dec stage 5
# enc1 skipped (128³ too large for memory), enc6 skipped (bottleneck-adjacent)

class BreastNet3D_UNet(nn.Module):
    def __init__(self, drop=0.25):
        super().__init__()
        # ── 2D Encoder ──
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv2D(5, 32, 0)
        self.enc2 = DoubleConv2D(32, 64, 0)
        self.enc3 = DoubleConv2D(64, 128, drop)
        self.enc4 = DoubleConv2D(128, 256, drop)
        self.enc5 = DoubleConv2D(256, 512, drop)
        self.enc6 = DoubleConv2D(512, 512, drop)
        self.enc_fc = nn.Sequential(nn.Dropout(drop), nn.Linear(512 * 2 * 2, 1000))

        # ── Skip Projections ──
        self.proj2 = SkipProjection(64,   4, depth=64)
        self.proj3 = SkipProjection(128,  8, depth=32)
        self.proj4 = SkipProjection(256, 16, depth=16)
        self.proj5 = SkipProjection(512, 32, depth=8)

        # ── Bottleneck ──
        self.dec_fc = nn.Linear(1000, 512 * 2 * 2 * 2)

        # ── 3D Decoder ──
        self.up1 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.d1  = DoubleConv3D(256, 256, drop)          # 2³→4³, no skip
        self.dec2 = DecoderBlock3D(256, 32, 128, drop)    # 4³→8³,  skip enc5
        self.dec3 = DecoderBlock3D(128, 16, 64,  drop)    # 8³→16³, skip enc4
        self.dec4 = DecoderBlock3D(64,   8, 32,  0)       # 16³→32³, skip enc3
        self.dec5 = DecoderBlock3D(32,   4, 16,  0)       # 32³→64³, skip enc2
        self.up6 = nn.ConvTranspose3d(16, 8, 2, stride=2)
        self.d6  = DoubleConv3D(8, 8, 0)                  # 64³→128³, no skip

        # ── Visual hull fusion (same as Decoder3D) ──
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
        # Encoder — store skip features before pool
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        s5 = self.enc5(self.pool(s4))
        s6 = self.enc6(self.pool(s5))
        z  = self.enc_fc(self.pool(s6).view(x.size(0), -1))

        # Bottleneck → 3D seed
        x3 = self.dec_fc(z).view(x.size(0), 512, 2, 2, 2)

        # Decoder with skip injection
        x3 = self.d1(self.up1(x3))
        x3 = self.dec2(x3, self.proj5(s5))
        x3 = self.dec3(x3, self.proj4(s4))
        x3 = self.dec4(x3, self.proj3(s3))
        x3 = self.dec5(x3, self.proj2(s2))
        x3 = self.d6(self.up6(x3))

        # Visual hull fusion
        if visual_hull is None:
            visual_hull = torch.zeros(
                x3.size(0), 1, x3.size(2), x3.size(3), x3.size(4),
                device=x3.device, dtype=x3.dtype)
        x3 = torch.cat([x3, visual_hull], dim=1)
        return self.fusion(x3) * visual_hull


# ── Shape Prior (defined, NOT wired into training yet) ───────────────────
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
    def ell(cx):
        return ((xx - cx)**2 / ax**2 + (yy - cy)**2 / ay**2 + (zz - cz)**2 / az**2)
    prior = torch.sigmoid(8.0 * (1.0 - torch.minimum(ell(cx_L), ell(cx_R))))
    return prior.unsqueeze(0).unsqueeze(0)

class ShapePriorLoss(nn.Module):
    def __init__(self, volume_size=128, device='cuda', weight=0.3):
        super().__init__()
        self.weight = weight
        self.register_buffer('prior', build_dual_ellipsoid_prior(volume_size, device).detach())

    def forward(self, V_pred):
        return self.weight * (V_pred * (1.0 - self.prior)).mean()


USE_SHAPE_PRIOR = False  # set True to activate shape prior


###############################################################################
# CELL B — Paste as NEXT cell: Verification
###############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Verify U-Net forward pass
_test_model = BreastNet3D_UNet().to(device)
_test_in = torch.zeros(2, 5, 128, 128, device=device)
_test_hull = torch.ones(2, 1, 128, 128, 128, device=device)
with torch.no_grad():
    _test_out = _test_model(_test_in, _test_hull)
assert _test_out.shape == (2, 1, 128, 128, 128), f"Shape mismatch: {_test_out.shape}"
print("U-Net model forward pass OK:", _test_out.shape)
del _test_model, _test_in, _test_hull, _test_out
torch.cuda.empty_cache()

# Verify prior visualisation
import matplotlib.pyplot as plt
_prior_np = build_dual_ellipsoid_prior(128, 'cpu').squeeze().numpy()
plt.imshow(_prior_np[64, :, :], cmap='hot', vmin=0, vmax=1)
plt.title('Dual-ellipsoid prior — frontal slice (depth=64)')
plt.colorbar(); plt.show()
del _prior_np
