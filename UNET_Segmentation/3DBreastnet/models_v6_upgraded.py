\"\"\"3DBreastNet Upgraded Models & Helpers Library.
Contains all architectures (original & U-Net upgraded) and rendering/loss functions
for 3D Breast Thermography Reconstruction.
\"\"\"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Building Blocks ─────────────────────────────────────────────────────────
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

# ── Original Frozen U-Net Segmentor ──────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, inc, outc, drop=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc), nn.ReLU(True),
            nn.Conv2d(outc, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc), nn.ReLU(True),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity())
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, b=64, drop=0.2):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_c, b, 0.0)
        self.enc2 = DoubleConv(b, b*2, 0.0)
        self.enc3 = DoubleConv(b*2, b*4, 0.1)
        self.enc4 = DoubleConv(b*4, b*8, 0.1)
        self.bottleneck = DoubleConv(b*8, b*16, drop)
        self.up4 = nn.ConvTranspose2d(b*16, b*8, 2, stride=2)
        self.dec4 = DoubleConv(b*16, b*8, 0.1)
        self.up3 = nn.ConvTranspose2d(b*8, b*4, 2, stride=2)
        self.dec3 = DoubleConv(b*8, b*4, 0.1)
        self.up2 = nn.ConvTranspose2d(b*4, b*2, 2, stride=2)
        self.dec2 = DoubleConv(b*4, b*2, 0.0)
        self.up1 = nn.ConvTranspose2d(b*2, b, 2, stride=2)
        self.dec1 = DoubleConv(b*2, b, 0.0)
        self.out = nn.Conv2d(b, out_c, 1)
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        b=self.bottleneck(self.pool(e4))
        d4=self.dec4(torch.cat([self.up4(b),e4],1))
        d3=self.dec3(torch.cat([self.up3(d4),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return self.out(d1)

# ── Original Encoder 2D ──────────────────────────────────────────────────────
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
        self.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(512*2*2, 1000))
        self.apply(_init)
    def forward(self, x):
        for enc in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.enc6]:
            x = enc(x); x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))

# ── Original Decoder 3D ──────────────────────────────────────────────────────
class Decoder3D(nn.Module):
    def __init__(self, drop=0.25):
        super().__init__()
        self.fc = nn.Linear(1000, 512*2*2*2)
        self.up1=nn.ConvTranspose3d(512,256,2,stride=2); self.d1=DoubleConv3D(256,256,drop)
        self.up2=nn.ConvTranspose3d(256,128,2,stride=2); self.d2=DoubleConv3D(128,128,drop)
        self.up3=nn.ConvTranspose3d(128,64,2,stride=2);  self.d3=DoubleConv3D(64,64,drop)
        self.up4=nn.ConvTranspose3d(64,32,2,stride=2);   self.d4=DoubleConv3D(32,32,0)
        self.up5=nn.ConvTranspose3d(32,16,2,stride=2);   self.d5=DoubleConv3D(16,16,0)
        self.up6=nn.ConvTranspose3d(16,8,2,stride=2);    self.d6=DoubleConv3D(8,8,0)
        
        self.fusion = nn.Sequential(
            nn.Conv3d(8 + 1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(8), nn.ReLU(True),
            nn.Conv3d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.apply(_init)
        if self.fusion[3].bias is not None:
            nn.init.constant_(self.fusion[3].bias, -4.0)

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
            x = self._s4(x); x = self._s5(x); x = self._s6(x)
            
        if visual_hull is None:
            visual_hull = torch.zeros(
                x.size(0), 1, x.size(2), x.size(3), x.size(4),
                device=x.device, dtype=x.dtype
            )
        x = torch.cat([x, visual_hull], dim=1)
        return self.fusion(x) * visual_hull

# ── Skip Projection: bridges 2D encoder features to 3D decoder space ──────────
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
        feat_3d = feat_2d.unsqueeze(2).expand(-1, -1, self.depth, -1, -1)
        return self.proj(feat_3d)

# ── Decoder block with skip concatenation ──
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

# ── Upgraded U-Net Model (BreastNet3D_UNet) ───────────────────────────────────
class BreastNet3D_UNet(nn.Module):
    def __init__(self, drop=0.25):
        super().__init__()
        # Encoder
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv2D(5, 32, 0)
        self.enc2 = DoubleConv2D(32, 64, 0)
        self.enc3 = DoubleConv2D(64, 128, drop)
        self.enc4 = DoubleConv2D(128, 256, drop)
        self.enc5 = DoubleConv2D(256, 512, drop)
        self.enc6 = DoubleConv2D(512, 512, drop)
        self.enc_fc = nn.Sequential(nn.Dropout(drop), nn.Linear(512*2*2, 1000))

        # Skip Projections
        self.proj2 = SkipProjection(64,   4, depth=64)
        self.proj3 = SkipProjection(128,  8, depth=32)
        self.proj4 = SkipProjection(256, 16, depth=16)
        self.proj5 = SkipProjection(512, 32, depth=8)

        # Bottleneck
        self.dec_fc = nn.Linear(1000, 512*2*2*2)

        # 3D Decoder
        self.up1 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.d1  = DoubleConv3D(256, 256, drop)
        self.dec2 = DecoderBlock3D(256, 32, 128, drop)
        self.dec3 = DecoderBlock3D(128, 16, 64, drop)
        self.dec4 = DecoderBlock3D(64, 8, 32, 0)
        self.dec5 = DecoderBlock3D(32, 4, 16, 0)
        self.up6 = nn.ConvTranspose3d(16, 8, 2, stride=2)
        self.d6  = DoubleConv3D(8, 8, 0)

        # Fusion
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
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        s5 = self.enc5(self.pool(s4))
        s6 = self.enc6(self.pool(s5))
        z  = self.enc_fc(self.pool(s6).view(x.size(0), -1))

        x3 = self.dec_fc(z).view(x.size(0), 512, 2, 2, 2)
        x3 = self.d1(self.up1(x3))
        x3 = self.dec2(x3, self.proj5(s5))
        x3 = self.dec3(x3, self.proj4(s4))
        x3 = self.dec4(x3, self.proj3(s3))
        x3 = self.dec5(x3, self.proj2(s2))
        x3 = self.d6(self.up6(x3))

        if visual_hull is None:
            visual_hull = torch.zeros(
                x3.size(0), 1, x3.size(2), x3.size(3), x3.size(4),
                device=x3.device, dtype=x3.dtype)
        x3 = torch.cat([x3, visual_hull], dim=1)
        return self.fusion(x3) * visual_hull

# ── Differentiable Projection ─────────────────────────────────────────────────
def render_projection(volume, theta_deg):
    B,C,D,H,W = volume.shape; dev = volume.device
    if not isinstance(theta_deg, torch.Tensor):
        theta_deg = torch.full((B,), float(theta_deg), device=dev, dtype=torch.float32)
    theta_deg = theta_deg.float()
    rad = theta_deg * math.pi / 180.0
    c, s = torch.cos(rad), torch.sin(rad)
    z, o = torch.zeros_like(rad), torch.ones_like(rad)
    mat = torch.stack([torch.stack([c,z,s,z],-1),
                       torch.stack([z,o,z,z],-1),
                       torch.stack([-s,z,c,z],-1)], -2)
    grid = F.affine_grid(mat, volume.shape, align_corners=False)
    Vr = F.grid_sample(volume, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return 1.0 - torch.exp(-Vr.squeeze(1).sum(dim=1, keepdim=True))

# ── Dice Loss ─────────────────────────────────────────────────────────────────
def dice_loss(pred, target, eps=1e-6):
    p, t = pred.float(), target.float()
    num = 2*(p*t).sum()
    den = p.pow(2).sum() + t.pow(2).sum() + eps
    return 1 - num/den

VIEW_WINDOWS = [(-90.,-67.5),(-67.5,-22.5),(-22.5,22.5),(22.5,67.5),(67.5,90.)]

# ── Visual Hull ───────────────────────────────────────────────────────────────
def compute_visual_hull(m5, device):
    B = m5.shape[0]
    D = H = W = 128
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid_pts = torch.stack([x, y, z], dim=0).view(3, -1)
    
    angles = [-90., -45., 0., 45., 90.]
    hull = torch.ones((B, 1, D*H*W), device=device)
    
    for i, angle in enumerate(angles):
        rad = angle * math.pi / 180.0
        c, s = math.cos(rad), math.sin(rad)
        X_cam = grid_pts[0]*c - grid_pts[2]*s
        Y_cam = grid_pts[1]
        
        sample_coords = torch.stack([X_cam, Y_cam], dim=-1).unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1)
        mask_view = m5[:, i:i+1, :, :]
        sampled = F.grid_sample(mask_view, sample_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        hull = hull * sampled.squeeze(3)
        
    return hull.view(B, 1, D, H, W)

# ── Boundary Loss ─────────────────────────────────────────────────────────────
def boundary_loss(pred, target):
    weight = torch.tensor([[[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]]], device=pred.device)
    pred_edge = F.conv2d(pred, weight, padding=1)
    target_edge = F.conv2d(target, weight, padding=1)
    return F.l1_loss(torch.abs(pred_edge), torch.abs(target_edge))

# ── Shape Prior ───────────────────────────────────────────────────────────────
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
        return ((xx-cx)**2/ax**2 + (yy-cy)**2/ay**2 + (zz-cz)**2/az**2)
    prior = torch.sigmoid(8.0 * (1.0 - torch.minimum(ell(cx_L), ell(cx_R))))
    return prior.unsqueeze(0).unsqueeze(0)

class ShapePriorLoss(nn.Module):
    def __init__(self, volume_size=128, device='cuda', weight=0.3):
        super().__init__()
        self.weight = weight
        self.register_buffer('prior', build_dual_ellipsoid_prior(volume_size, device).detach())
    def forward(self, V_pred):
        return self.weight * (V_pred * (1.0 - self.prior)).mean()
