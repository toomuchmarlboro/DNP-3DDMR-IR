"""
models_v7.py — 3DBreastNet V7 Architecture
============================================
Changes from V5/V6:
  1. Encoder2D bottleneck: 1000 → 8192 (8× capacity for curvature memory)
  2. Decoder3D fusion: 2-layer sequential → 4-layer residual block (sculpting power)
  3. Decoder3D bias: -4.0 → +4.0 (start full, carve down — eliminates flat cutoffs)
  4. compute_visual_hull() exported here for clean imports
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Weight initializer ──────────────────────────────────────────────
def _init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ── Visual Hull (differentiable, used in training + inference) ──────
def compute_visual_hull(m5, device):
    """Intersect 5 silhouette cones to form a 128³ binary hull."""
    B = m5.shape[0]
    D = H = W = 128
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    grid_pts = torch.stack([x, y, z], dim=0).view(3, -1)

    angles = [-90.0, -45.0, 0.0, 45.0, 90.0]
    hull = torch.ones((B, 1, D * H * W), device=device)

    for i, angle in enumerate(angles):
        rad = angle * math.pi / 180.0
        c, s = math.cos(rad), math.sin(rad)
        X_cam = grid_pts[0] * c - grid_pts[2] * s
        Y_cam = grid_pts[1]
        sample_coords = (
            torch.stack([X_cam, Y_cam], dim=-1)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(B, -1, -1, -1)
        )
        mask_view = m5[:, i : i + 1, :, :]
        sampled = F.grid_sample(
            mask_view, sample_coords, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )
        hull = hull * sampled.squeeze(3)

    return hull.view(B, 1, D, H, W)


# ── 2D building blocks ──────────────────────────────────────────────
class DoubleConv2D(nn.Module):
    def __init__(self, inc, outc, drop=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc), nn.ReLU(True),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(outc, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc), nn.ReLU(True),
        )
    def forward(self, x):
        return self.block(x)


# ── 3D building blocks ──────────────────────────────────────────────
class DoubleConv3D(nn.Module):
    def __init__(self, inc, outc, drop=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inc, outc, 3, padding=1, bias=False),
            nn.BatchNorm3d(outc), nn.ReLU(True),
            nn.Dropout3d(drop) if drop > 0 else nn.Identity(),
            nn.Conv3d(outc, outc, 3, padding=1, bias=False),
            nn.BatchNorm3d(outc), nn.ReLU(True),
        )
    def forward(self, x):
        return self.block(x)


# ── U-Net (2D segmentation, frozen during 3D training) ──────────────
class DoubleConv(nn.Module):
    def __init__(self, inc, outc, drop=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc), nn.ReLU(True),
            nn.Conv2d(outc, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc), nn.ReLU(True),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
        )
    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, b=64, drop=0.2):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_c, b, 0.0)
        self.enc2 = DoubleConv(b, b * 2, 0.0)
        self.enc3 = DoubleConv(b * 2, b * 4, 0.1)
        self.enc4 = DoubleConv(b * 4, b * 8, 0.1)
        self.bottleneck = DoubleConv(b * 8, b * 16, drop)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8, 0.1)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4, 0.1)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2, 0.0)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = DoubleConv(b * 2, b, 0.0)
        self.out = nn.Conv2d(b, out_c, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


# ── Encoder2D  (5×128×128 → 8192-d latent) ──────────────────────────
class Encoder2D(nn.Module):
    def __init__(self, drop=0.25):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv2D(5, 32, 0)
        self.enc2 = DoubleConv2D(32, 64, 0)
        self.enc3 = DoubleConv2D(64, 128, drop)
        self.enc4 = DoubleConv2D(128, 256, drop)
        self.enc5 = DoubleConv2D(256, 512, drop)
        self.enc6 = DoubleConv2D(512, 512, drop)
        # V7: 8192-d bottleneck (was 1000 in V5)
        self.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(512 * 2 * 2, 8192))
        self.apply(_init)

    def forward(self, x):
        for enc in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.enc6]:
            x = enc(x)
            x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))


# ── Decoder3D  (8192-d → 1×128×128×128) with visual-hull sculpting ──
class Decoder3D(nn.Module):
    def __init__(self, drop=0.25):
        super().__init__()
        # V7: accepts 8192-d input (was 1000 in V5)
        self.fc = nn.Linear(8192, 512 * 2 * 2 * 2)

        self.up1 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.d1 = DoubleConv3D(256, 256, drop)
        self.up2 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.d2 = DoubleConv3D(128, 128, drop)
        self.up3 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.d3 = DoubleConv3D(64, 64, drop)
        self.up4 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.d4 = DoubleConv3D(32, 32, 0)
        self.up5 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.d5 = DoubleConv3D(16, 16, 0)
        self.up6 = nn.ConvTranspose3d(16, 8, 2, stride=2)
        self.d6 = DoubleConv3D(8, 8, 0)

        # V7: Deep residual fusion block (was 2-layer sequential in V5)
        self.fusion_conv1 = nn.Conv3d(8 + 1, 16, 3, padding=1, bias=False)
        self.fusion_bn1 = nn.BatchNorm3d(16)
        self.fusion_res1 = nn.Conv3d(16, 16, 3, padding=1, bias=False)
        self.fusion_res_bn1 = nn.BatchNorm3d(16)
        self.fusion_res2 = nn.Conv3d(16, 16, 3, padding=1, bias=False)
        self.fusion_res_bn2 = nn.BatchNorm3d(16)
        self.fusion_out = nn.Conv3d(16, 1, 1)

        self.apply(_init)
        # V7: +4.0 bias → network starts FULL (was -4.0 in V5)
        # Combined with visual hull, epoch-1 output ≈ exact visual hull
        if self.fusion_out.bias is not None:
            nn.init.constant_(self.fusion_out.bias, 4.0)

    # Gradient-checkpointed stages (saves ~40% VRAM)
    def _s4(self, x): return self.d4(self.up4(x))
    def _s5(self, x): return self.d5(self.up5(x))
    def _s6(self, x): return self.d6(self.up6(x))

    def forward(self, x, visual_hull=None):
        x = self.fc(x).view(x.size(0), 512, 2, 2, 2)
        x = self.d1(self.up1(x))
        x = self.d2(self.up2(x))
        x = self.d3(self.up3(x))
        if x.requires_grad:
            x = torch.utils.checkpoint.checkpoint(self._s4, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._s5, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._s6, x, use_reentrant=False)
        else:
            x = self._s4(x)
            x = self._s5(x)
            x = self._s6(x)

        if visual_hull is None:
            visual_hull = torch.ones(
                x.size(0), 1, x.size(2), x.size(3), x.size(4),
                device=x.device, dtype=x.dtype,
            )

        # Concatenate learned features + hull, pass through residual fusion
        x = torch.cat([x, visual_hull], dim=1)
        f = F.relu(self.fusion_bn1(self.fusion_conv1(x)))
        res = f
        f = F.relu(self.fusion_res_bn1(self.fusion_res1(f)))
        f = self.fusion_res_bn2(self.fusion_res2(f))
        f = F.relu(f + res)  # residual connection
        out = torch.sigmoid(self.fusion_out(f))
        # Multiply by hull: hard geometric constraint (nothing outside hull)
        return out * visual_hull


# ── Differentiable projection (Eq. 1-3 from paper) ──────────────────
def render_projection(volume, theta_deg):
    """Render a 2D silhouette from a 3D volume at a given angle."""
    B, C, D, H, W = volume.shape
    dev = volume.device
    if not isinstance(theta_deg, torch.Tensor):
        theta_deg = torch.full((B,), float(theta_deg), device=dev, dtype=torch.float32)
    theta_deg = theta_deg.float()
    rad = theta_deg * math.pi / 180.0
    c, s = torch.cos(rad), torch.sin(rad)
    z, o = torch.zeros_like(rad), torch.ones_like(rad)
    mat = torch.stack(
        [
            torch.stack([c, z, s, z], -1),
            torch.stack([z, o, z, z], -1),
            torch.stack([-s, z, c, z], -1),
        ],
        -2,
    )
    grid = F.affine_grid(mat, volume.shape, align_corners=False)
    Vr = F.grid_sample(volume, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return 1.0 - torch.exp(-Vr.squeeze(1).sum(dim=1, keepdim=True))


# ── Dice loss (float32-safe) ────────────────────────────────────────
def dice_loss(pred, target, eps=1e-6):
    p, t = pred.float(), target.float()
    num = 2 * (p * t).sum()
    den = p.pow(2).sum() + t.pow(2).sum() + eps
    return 1 - num / den


# ── Boundary loss (Laplacian edge) ──────────────────────────────────
def boundary_loss(pred, target):
    weight = torch.tensor(
        [[[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]]],
        device=pred.device,
    )
    pred_edge = F.conv2d(pred, weight, padding=1)
    target_edge = F.conv2d(target, weight, padding=1)
    return F.l1_loss(torch.abs(pred_edge), torch.abs(target_edge))


# ── View windows for jittered training ──────────────────────────────
VIEW_WINDOWS = [
    (-90.0, -67.5),
    (-67.5, -22.5),
    (-22.5, 22.5),
    (22.5, 67.5),
    (67.5, 90.0),
]
