###############################################################################
# CELL C — REPLACE the existing Training cell (Cell 3 in original notebook)
# Fully integrated and 100% self-contained training script.
# Contains all model definitions, losses, and helper functions directly
# to ensure zero dependencies and complete seamless execution.
###############################################################################

"""3DBreastNet v6 — Training (U-Net skip-connection edition)."""
import os, sys, time, random, json, math
import numpy as np, cv2, tifffile, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage

# ════════════════════════════════════════════════════════════════════════════
# 1. MODEL DEFINITIONS, LOSSES, AND GEOMETRIC HELPERS
# ════════════════════════════════════════════════════════════════════════════

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

# Frozen U-Net Segmentor
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

# Skip Projection
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

# Decoder block with skip concatenation
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

# Upgraded U-Net Model (BreastNet3D_UNet)
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

# Differentiable Projection
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

# Dice Loss
def dice_loss(pred, target, eps=1e-6):
    p, t = pred.float(), target.float()
    num = 2*(p*t).sum()
    den = p.pow(2).sum() + t.pow(2).sum() + eps
    return 1 - num/den

VIEW_WINDOWS = [(-90.,-67.5),(-67.5,-22.5),(-22.5,22.5),(22.5,67.5),(67.5,90.)]

# Visual Hull
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

# Boundary Loss
def boundary_loss(pred, target):
    weight = torch.tensor([[[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]]], device=pred.device)
    pred_edge = F.conv2d(pred, weight, padding=1)
    target_edge = F.conv2d(target, weight, padding=1)
    return F.l1_loss(torch.abs(pred_edge), torch.abs(target_edge))

# Shape Prior
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

# ════════════════════════════════════════════════════════════════════════════
# 2. CONFIG AND DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

CFG = {
    "epochs":       400,
    "batch_size":   2,
    "lr":           1e-4,
    "betas":        (0.5, 0.9),
    "n_per_view":   2,
    "seed":         42,
    "patience":     50,
    "ckpt_dir":     "checkpoints_3d_v6",
    "tiff_base":    r"../../data/organized_by_patient",
    "unet_ckpt":    r"/mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/UNET_Segmentation/breast_segmentation_unet_best_gpu.pth",
}

USE_SHAPE_PRIOR = False  # Set to True to enable Upgrade 2 prior loss

@dataclass
class PatientGroup:
    patient_id: str
    label: str
    views: Dict[str, Path]

def get_view_key(filename):
    n = filename.lower()
    if "right later" in n: return "RL"
    if "right obli"  in n: return "RO"
    if "frontal" in n or "anterior" in n: return "F"
    if "left obliq"  in n: return "LO"
    if "left later"  in n: return "LL"
    return None

def build_patient_groups(tiff_base):
    tb = Path(tiff_base)
    pd_ = {}
    for tp in tb.rglob("*.tiff"):
        parts = tp.relative_to(tb).parts
        if len(parts) < 3: continue
        pid, lab, fn = parts[0], parts[1], parts[-1]
        vk = get_view_key(fn)
        if not vk: continue
        key = (pid, lab)
        if key not in pd_: pd_[key] = {"views": {}}
        pd_[key]["views"][vk] = tp
    groups, skip, nb_, nm = [], 0, 0, 0
    for (pid, lab), d in pd_.items():
        if len(d["views"]) == 5:
            groups.append(PatientGroup(pid, lab, d["views"]))
            nb_ += lab.lower() == "benign"; nm += lab.lower() != "benign"
        else:
            print(f"  Skip {pid} ({lab}): {len(d['views'])}/5 views"); skip += 1
    groups.sort(key=lambda g: g.patient_id)
    print(f"Patients: {len(pd_)} | Complete: {len(groups)} | Skipped: {skip} | B={nb_} M={nm}")
    return groups

class PatientDataset(Dataset):
    def __init__(self, groups, unet, device, img_sz=256):
        self.groups, self.unet, self.device = groups, unet, device
        self.img_sz = img_sz
        self.views = ["RL","RO","F","LO","LL"]
    def __len__(self): return len(self.groups)
    def __getitem__(self, idx):
        g = self.groups[idx]; thermals, masks = [], []
        for v in self.views:
            raw = tifffile.imread(str(g.views[v])).astype(np.float32)
            raw = cv2.resize(raw, (self.img_sz, self.img_sz))
            mn, mx = raw.min(), raw.max()
            norm = (raw - mn) / (mx - mn + 1e-8)
            thermals.append(norm)
            with torch.no_grad():
                inp = torch.tensor(norm).unsqueeze(0).unsqueeze(0).to(self.device)
                m = (torch.sigmoid(self.unet(inp)).squeeze().cpu().numpy() > 0.5).astype(np.float32)
            masks.append(cv2.resize(m, (128,128), interpolation=cv2.INTER_NEAREST))
        return {
            "masks_5ch": torch.tensor(np.stack(masks), dtype=torch.float32),
            "thermals_5ch": torch.tensor(np.stack(thermals), dtype=torch.float32),
            "patient_id": g.patient_id, "label": g.label,
        }

# ════════════════════════════════════════════════════════════════════════════
# 3. TRAINING ENGINE
# ════════════════════════════════════════════════════════════════════════════

def hd95(p, t):
    if p.sum()==0 or t.sum()==0: return 128.0
    pe = p ^ scipy.ndimage.binary_erosion(p)
    te = t ^ scipy.ndimage.binary_erosion(t)
    dtp = scipy.ndimage.distance_transform_edt(~pe)
    dtt = scipy.ndimage.distance_transform_edt(~te)
    d1 = np.percentile(dtt[pe], 95) if pe.sum()>0 else 128.0
    d2 = np.percentile(dtp[te], 95) if te.sum()>0 else 128.0
    return max(d1, d2)

def train(cfg):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Frozen U-Net segmentor
    unet = UNet().to(device)
    unet.load_state_dict(torch.load(cfg["unet_ckpt"], map_location=device))
    unet.eval()
    for p in unet.parameters(): p.requires_grad = False

    groups = build_patient_groups(cfg["tiff_base"])
    rng = random.Random(cfg["seed"])
    ben = [g for g in groups if g.label.lower()=="benign"]
    mal = [g for g in groups if g.label.lower()!="benign"]
    rng.shuffle(ben); rng.shuffle(mal)
    s = 0.78
    trn = ben[:int(len(ben)*s)] + mal[:int(len(mal)*s)]
    val = ben[int(len(ben)*s):] + mal[int(len(mal)*s):]
    print(f"Train: {len(trn)} | Val: {len(val)}")
    trn_dl = DataLoader(PatientDataset(trn, unet, device),
                        batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_dl = DataLoader(PatientDataset(val, unet, device),
                        batch_size=cfg["batch_size"], shuffle=False)

    # ── BreastNet3D_UNet ──
    model = BreastNet3D_UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=cfg["betas"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=30, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

    shape_prior = ShapePriorLoss(128, device) if USE_SHAPE_PRIOR else None

    best_dice, no_imp = 0.0, 0
    hist = {"epoch":[], "train_dice_loss":[], "val_dice_loss":[],
            "val_dice_score":[], "val_hd95":[], "prior_loss":[]}

    for epoch in range(1, cfg["epochs"]+1):
        t0 = time.time()
        model.train()
        ep_loss, ep_prior = 0.0, 0.0

        for batch in tqdm(trn_dl, desc=f"E{epoch:03d} train", leave=False):
            m5 = batch["masks_5ch"].to(device); B = m5.size(0)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                hull = compute_visual_hull(m5, device)
                vol = model(m5, hull)
            vol = vol.float()
            loss = torch.tensor(0.0, device=device)
            for i in range(5):
                lo, hi = VIEW_WINDOWS[i]
                for _ in range(cfg["n_per_view"]):
                    th = torch.rand(B, device=device)*(hi-lo)+lo
                    proj = render_projection(vol, th)
                    gt = m5[:, i:i+1]
                    loss = loss + dice_loss(proj, gt) + 2.0 * boundary_loss(proj, gt)
            loss = loss / (5 * cfg["n_per_view"])

            p_val = 0.0
            if shape_prior is not None:
                pl = shape_prior(vol); loss = loss + pl; p_val = pl.item()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if torch.isfinite(loss):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
            else:
                print(f"  ⚠ NaN loss @ epoch {epoch}")
            scaler.update(); opt.zero_grad(set_to_none=True)
            ep_loss += loss.item() if torch.isfinite(loss) else 0.0
            ep_prior += p_val
        ep_loss /= max(len(trn_dl), 1); ep_prior /= max(len(trn_dl), 1)

        # ── Validation ──
        model.eval()
        vl, vd, vh, cnt = 0., 0., 0., 0
        val_angles = [-90., -45., 0., 45., 90.]
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"E{epoch:03d} val", leave=False):
                m5 = batch["masks_5ch"].to(device); B = m5.size(0)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    hull = compute_visual_hull(m5, device)
                vol = model(m5, hull).float()
                for i in range(5):
                    th = torch.full((B,), val_angles[i], device=device)
                    proj = render_projection(vol, th)
                    dl = dice_loss(proj, m5[:, i:i+1])
                    vl += dl.item(); vd += (1 - dl).item()
                    pb = (proj > 0.5).cpu().numpy(); mb = (m5[:, i:i+1] > 0.5).cpu().numpy()
                    for b in range(B): vh += hd95(pb[b,0], mb[b,0]); cnt += 1
        vl /= max(len(val_dl)*5,1); vd /= max(len(val_dl)*5,1); vh /= max(cnt,1)

        hist["epoch"].append(epoch); hist["train_dice_loss"].append(ep_loss)
        hist["val_dice_loss"].append(vl); hist["val_dice_score"].append(vd)
        hist["val_hd95"].append(vh); hist["prior_loss"].append(ep_prior)
        sched.step(vd)
        lr_now = opt.param_groups[0]["lr"]
        print(f"E{epoch:03d} | loss={ep_loss:.4f} prior={ep_prior:.4f} | "
              f"vl={vl:.4f} vd={vd:.4f} hd={vh:.2f} | lr={lr_now:.6f} | {time.time()-t0:.1f}s")

        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "opt": opt.state_dict(), "best_dice": max(best_dice, vd),
                "cfg": cfg, "hist": hist}
        torch.save(ckpt, Path(cfg["ckpt_dir"]) / "3dbreastnet_unet_last.pth")
        if vd > best_dice:
            best_dice = vd; no_imp = 0
            torch.save(ckpt, Path(cfg["ckpt_dir"]) / "3dbreastnet_unet_best.pth")
            print(f"  ★ new best dice={best_dice:.4f}")
        else:
            no_imp += 1
            if no_imp >= cfg["patience"]:
                print(f"Early stopping @ epoch {epoch}"); break

    fig, ax = plt.subplots(1, 4, figsize=(24, 5))
    ax[0].plot(hist["epoch"], hist["train_dice_loss"], label="train")
    ax[0].plot(hist["epoch"], hist["val_dice_loss"], label="val")
    ax[0].set_title("Dice Loss"); ax[0].legend()
    ax[1].plot(hist["epoch"], hist["val_dice_score"], color="green"); ax[1].set_title("Val Dice")
    ax[2].plot(hist["epoch"], hist["val_hd95"], color="red"); ax[2].set_title("Val HD95")
    ax[3].plot(hist["epoch"], hist["prior_loss"], color="purple"); ax[3].set_title("Prior Loss")
    for a in ax: a.set_xlabel("Epoch")
    plt.tight_layout(); plt.savefig(Path(cfg["ckpt_dir"])/"training_history.png", dpi=150)
    print(f"Plot saved. Best dice={best_dice:.4f}")

if __name__ == "__main__":
    train(CFG)
