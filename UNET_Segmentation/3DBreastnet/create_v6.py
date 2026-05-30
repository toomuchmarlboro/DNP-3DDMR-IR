import json

NOTEBOOK_PATH = r"c:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\DNP-3DDMR-IR\UNET_Segmentation\3DBreastnet\breastnet3d_v5.ipynb"
OUTPUT_PATH = r"c:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\DNP-3DDMR-IR\UNET_Segmentation\3DBreastnet\breastnet3d_v6.ipynb"

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

model_code = """import math, torch, torch.nn as nn, torch.nn.functional as F

# ── Building blocks ──────────────────────────────────────────
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

class DecoderBlock3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, drop=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.block = DoubleConv3D(in_ch + skip_ch, out_ch, drop)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class Encoder2D_UNet(nn.Module):
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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.enc6(self.pool(e5))
        latent = self.fc(self.pool(e6).view(x.size(0), -1))
        return latent, (e1, e2, e3, e4, e5, e6)

class Decoder3D_UNet(nn.Module):
    def __init__(self, drop=0.25):
        super().__init__()
        self.fc = nn.Linear(1000, 512*2*2*2)
        self.proj6 = SkipProjection(512, 128, depth=4)
        self.proj5 = SkipProjection(512, 128, depth=8)
        self.proj4 = SkipProjection(256, 64, depth=16)
        self.proj3 = SkipProjection(128, 32, depth=32)
        self.proj2 = SkipProjection(64, 16, depth=64)
        self.proj1 = SkipProjection(32, 8, depth=128)
        self.d1 = DecoderBlock3D(512, 128, 256, drop)
        self.d2 = DecoderBlock3D(256, 128, 128, drop)
        self.d3 = DecoderBlock3D(128, 64, 64, drop)
        self.d4 = DecoderBlock3D(64, 32, 32, 0)
        self.d5 = DecoderBlock3D(32, 16, 16, 0)
        self.d6 = DecoderBlock3D(16, 8, 8, 0)
        self.out = nn.Sequential(nn.Conv3d(8, 1, 1), nn.Sigmoid())
        self.apply(_init)
        nn.init.constant_(self.out[0].bias, -4.0)

    def _s4(self, x, s): return self.d4(x, s)
    def _s5(self, x, s): return self.d5(x, s)
    def _s6(self, x, s): return self.d6(x, s)

    def forward(self, latent, skips):
        e1, e2, e3, e4, e5, e6 = skips
        x = self.fc(latent).view(latent.size(0), 512, 2, 2, 2)
        x = self.d1(x, self.proj6(e6))
        x = self.d2(x, self.proj5(e5))
        x = self.d3(x, self.proj4(e4))
        if x.requires_grad:
            x = torch.utils.checkpoint.checkpoint(self._s4, x, self.proj3(e3), use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._s5, x, self.proj2(e2), use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._s6, x, self.proj1(e1), use_reentrant=False)
        else:
            x = self.d4(x, self.proj3(e3))
            x = self.d5(x, self.proj2(e2))
            x = self.d6(x, self.proj1(e1))
        return self.out(x)

class BreastNet3D_UNet(nn.Module):
    def __init__(self, drop=0.25):
        super().__init__()
        self.encoder = Encoder2D_UNet(drop)
        self.decoder = Decoder3D_UNet(drop)
    def forward(self, x):
        latent, skips = self.encoder(x)
        return self.decoder(latent, skips)

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

def dice_loss(pred, target, eps=1e-6):
    p, t = pred.float(), target.float()
    num = 2*(p*t).sum()
    den = p.pow(2).sum() + t.pow(2).sum() + eps
    return 1 - num/den

VIEW_WINDOWS = [(-90.,-67.5),(-67.5,-22.5),(-22.5,22.5),(22.5,67.5),(67.5,90.)]
"""

prior_code = """def build_dual_ellipsoid_prior(volume_size=128, device='cuda'):
    V = volume_size
    zz, yy, xx = torch.meshgrid(
        torch.linspace(0, 1, V, device=device),
        torch.linspace(0, 1, V, device=device),
        torch.linspace(0, 1, V, device=device),
        indexing='ij'
    )
    cx_L, cx_R = 0.30, 0.70
    cy = 0.55
    cz = 0.50
    ax, ay, az = 0.20, 0.35, 0.28
    def ellipsoid_sdf(cx):
        return ((xx - cx)**2 / ax**2 + (yy - cy)**2 / ay**2 + (zz - cz)**2 / az**2)
    e_L = ellipsoid_sdf(cx_L)
    e_R = ellipsoid_sdf(cx_R)
    prior = torch.sigmoid(8.0 * (1.0 - torch.minimum(e_L, e_R)))
    return prior.unsqueeze(0).unsqueeze(0)

class ShapePriorLoss(nn.Module):
    def __init__(self, volume_size=128, device='cuda', weight=0.3):
        super().__init__()
        self.weight = weight
        prior = build_dual_ellipsoid_prior(volume_size, device).detach()
        self.register_buffer('prior', prior)
    def forward(self, V_pred):
        outside_mass = V_pred * (1.0 - self.prior)
        return self.weight * outside_mass.mean()
"""

cc_code = """from scipy.ndimage import label
import numpy as np

def remove_floating_residuals(V_pred_np, threshold=0.5):
    \"\"\"
    Retains only the largest connected component in the 3D grid.
    This explicitly removes the floating 'red blobs' (visual hull artifacts).
    \"\"\"
    binary_vol = V_pred_np > threshold
    labeled_vol, num_features = label(binary_vol)
    if num_features == 0:
        return V_pred_np
    
    # Find the largest component (ignoring background 0)
    sizes = np.bincount(labeled_vol.ravel())
    sizes[0] = 0
    largest_label = sizes.argmax()
    
    cleaned_vol = V_pred_np.copy()
    cleaned_vol[labeled_vol != largest_label] = 0.0
    return cleaned_vol
"""

val_code = """import numpy as np
from scipy.spatial.distance import directed_hausdorff

def compute_metrics(pred_proj, target_sil, threshold=0.5):
    pred_bin = (pred_proj >= threshold).astype(np.float32)
    tgt = target_sil.astype(np.float32)
    TP = (pred_bin * tgt).sum()
    TN = ((1 - pred_bin) * (1 - tgt)).sum()
    FP = (pred_bin * (1 - tgt)).sum()
    FN = ((1 - pred_bin) * tgt).sum()
    N = pred_bin.size
    accuracy = (TP + TN) / N
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    jaccard = TP / (TP + FP + FN + 1e-6)
    pred_pts = np.argwhere(pred_bin > 0).astype(float)
    tgt_pts = np.argwhere(tgt > 0).astype(float)
    if len(pred_pts) == 0 or len(tgt_pts) == 0:
        hausdorff = float('nan')
    else:
        hausdorff = max(directed_hausdorff(pred_pts, tgt_pts)[0], directed_hausdorff(tgt_pts, pred_pts)[0])
    return {'Accuracy': accuracy, 'Dice Index': dice, 'Jaccard Index': jaccard, 'Hausdorff Distance': hausdorff}

print("Evaluation Metrics Function Ready.")
"""

model_idx = -1
for i, cell in enumerate(cells):
    src = "".join(cell.get('source', []))
    if 'import math, torch' in src and 'DoubleConv' in src:
        model_idx = i

if model_idx != -1:
    cells[model_idx]['source'] = [line + '\\n' for line in model_code.split('\\n')]

insertions = []
insertions.append({"cell_type": "markdown", "metadata": {}, "source": ["## Upgrade: Shape Prior"]})
insertions.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\\n' for line in prior_code.split('\\n')]})
insertions.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["USE_SHAPE_PRIOR = True\\nshape_prior_loss = ShapePriorLoss(volume_size=128, device='cuda', weight=0.3)\\n"]})
insertions.append({"cell_type": "markdown", "metadata": {}, "source": ["## Post-Processing: Remove Floating Residuals\\nRun this on the numpy volume before saving or plotting!"]})
insertions.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\\n' for line in cc_code.split('\\n')]})
insertions.append({"cell_type": "markdown", "metadata": {}, "source": ["## Validation Metrics"]})
insertions.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\\n' for line in val_code.split('\\n')]})

if model_idx != -1:
    cells = cells[:model_idx+1] + insertions + cells[model_idx+1:]

nb['cells'] = cells

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully created {OUTPUT_PATH}")
