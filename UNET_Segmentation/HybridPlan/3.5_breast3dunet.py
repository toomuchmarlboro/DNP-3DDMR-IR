"""3DBreastNet - 128x128x128 Voxel Reconstruction.

Standalone script extracted from notebook cells 1-8 so it can be run from a
terminal or tmux session.
"""

# -----------------------------------------------------------------------------
# 1. Install Dependencies
# -----------------------------------------------------------------------------
# From a shell, install the notebook dependencies with:
#   uv pip install torch torchvision tifffile opencv-python numpy scipy scikit-image matplotlib tqdm pandas
#   uv pip install tqdm ipywidgets


# -----------------------------------------------------------------------------
# 2. Model Definitions
# -----------------------------------------------------------------------------
import math
import os
import sys
import time
import random
import glob

import numpy as np
import cv2
import scipy.ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class DoubleConv2D(nn.Module):
    def __init__(self, inc, outc, drop=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc), nn.ReLU(True),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(outc, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class DoubleConv3D(nn.Module):
    def __init__(self, inc, outc, drop=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(inc, outc, 3, padding=1, bias=False),
            nn.BatchNorm3d(outc), nn.ReLU(True),
            nn.Dropout3d(drop) if drop > 0 else nn.Identity(),
            nn.Conv3d(outc, outc, 3, padding=1, bias=False),
            nn.BatchNorm3d(outc), nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


def _init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Encoder2D(nn.Module):
    def __init__(self, drop=0.25):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv2D(10, 32, 0)  # 10 Channels for Otsu+UNet
        self.enc2 = DoubleConv2D(32, 64, 0)
        self.enc3 = DoubleConv2D(64, 128, drop)
        self.enc4 = DoubleConv2D(128, 256, drop)
        self.enc5 = DoubleConv2D(256, 512, drop)
        self.enc6 = DoubleConv2D(512, 512, drop)
        self.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(512 * 2 * 2, 1000))
        self.apply(_init)

    def forward(self, x):
        for enc in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.enc6]:
            x = enc(x)
            x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))


class Decoder3D(nn.Module):
    def __init__(self, drop=0.25):
        super().__init__()
        self.fc = nn.Linear(1000, 512 * 2 * 2 * 2)
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
        self.out = nn.Sequential(nn.Conv3d(8, 1, 1), nn.Sigmoid())
        self.apply(_init)
        nn.init.constant_(self.out[0].bias, -4.0)

    # REMOVED GRADIENT CHECKPOINTING: Runs entirely in robust FP32
    def forward(self, x):
        x = self.fc(x).view(x.size(0), 512, 2, 2, 2)
        x = self.d1(self.up1(x))
        x = self.d2(self.up2(x))
        x = self.d3(self.up3(x))
        x = self.d4(self.up4(x))
        x = self.d5(self.up5(x))
        x = self.d6(self.up6(x))
        return self.out(x)


def render_projection(volume, theta_deg):
    B, C, D, H, W = volume.shape
    dev = volume.device
    if not isinstance(theta_deg, torch.Tensor):
        theta_deg = torch.full((B,), float(theta_deg), device=dev, dtype=torch.float32)
    theta_deg = theta_deg.float()
    rad = theta_deg * math.pi / 180.0
    c, s = torch.cos(rad), torch.sin(rad)
    z, o = torch.zeros_like(rad), torch.ones_like(rad)
    mat = torch.stack([
        torch.stack([c, z, s, z], -1),
        torch.stack([z, o, z, z], -1),
        torch.stack([-s, z, c, z], -1)
    ], -2)
    grid = F.affine_grid(mat, volume.shape, align_corners=False)
    Vr = F.grid_sample(volume, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return 1.0 - torch.exp(-Vr.squeeze(1).sum(dim=1, keepdim=True))


def dice_loss(pred, target, eps=1e-6):
    p, t = pred.float(), target.float()
    num = 2 * (p * t).sum()
    den = p.pow(2).sum() + t.pow(2).sum() + eps
    return 1 - num / den


VIEW_WINDOWS = [(-90., -67.5), (-67.5, -22.5), (-22.5, 22.5), (22.5, 67.5), (67.5, 90.)]


# -----------------------------------------------------------------------------
# 3. Training
# -----------------------------------------------------------------------------
CFG = {
    "epochs": 400,
    "batch_size": 2,
    "lr": 1e-4,
    "betas": (0.9, 0.999), 
    "n_per_view": 2,
    "seed": 42,
    "patience": 50,
    "n_splits": 3,
    "ckpt_dir": "checkpoints_3d_kfold",
    "otsu_dir": r"/mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/data/organized_by_patient_otsu",
    "unet_dir": r"/mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/data/organized_by_patient_unet",
}


def build_hybrid_patients(otsu_dir, unet_dir):
    otsu_p = set([f for f in os.listdir(otsu_dir) if f.startswith('Patient_')])
    unet_p = set([f for f in os.listdir(unet_dir) if f.startswith('Patient_')])
    valid_patients = sorted(list(otsu_p.intersection(unet_p)))

    patients = []
    for p in valid_patients:
        subs = [d for d in os.listdir(os.path.join(otsu_dir, p)) if os.path.isdir(os.path.join(otsu_dir, p, d))]
        if not subs:
            continue
        patients.append((p, subs[0]))
    return patients


class HybridPatientDataset(Dataset):
    def __init__(self, patient_tuples, otsu_dir, unet_dir, img_sz=128):
        self.patients = patient_tuples
        self.otsu_dir = otsu_dir
        self.unet_dir = unet_dir
        self.img_sz = img_sz
        self.views = ['Right Lateral', 'Right Oblique', 'Anterior', 'Left Oblique', 'Left Lateral']

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pid, cat = self.patients[idx]
        otsu_full = os.path.join(self.otsu_dir, pid, cat)
        unet_full = os.path.join(self.unet_dir, pid, cat)

        otsu_tensors, unet_tensors = [], []
        for view in self.views:
            of = [f for f in glob.glob(os.path.join(otsu_full, "*.png")) if view in os.path.basename(f)]
            if of:
                m = cv2.imdecode(np.fromfile(of[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                otsu_tensors.append(torch.from_numpy(cv2.resize(m, (self.img_sz, self.img_sz), interpolation=cv2.INTER_NEAREST)).float() / 255.0)
            else:
                otsu_tensors.append(torch.zeros((self.img_sz, self.img_sz), dtype=torch.float32))

            uf = [f for f in glob.glob(os.path.join(unet_full, "*.png")) if view in os.path.basename(f)]
            if uf:
                m = cv2.imdecode(np.fromfile(uf[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                unet_tensors.append(torch.from_numpy(cv2.resize(m, (self.img_sz, self.img_sz), interpolation=cv2.INTER_NEAREST)).float() / 255.0)
            else:
                unet_tensors.append(torch.zeros((self.img_sz, self.img_sz), dtype=torch.float32))

        input_10ch = torch.cat([torch.stack(otsu_tensors, dim=0), torch.stack(unet_tensors, dim=0)], dim=0)
        return {"input_10ch": input_10ch, "target_5ch": torch.stack(unet_tensors, dim=0)}


def hd95(p, t):
    if p.sum() == 0 or t.sum() == 0:
        return 128.0
    pe = p ^ scipy.ndimage.binary_erosion(p)
    te = t ^ scipy.ndimage.binary_erosion(t)
    dtp = scipy.ndimage.distance_transform_edt(~pe)
    dtt = scipy.ndimage.distance_transform_edt(~te)
    return max(np.percentile(dtt[pe], 95) if pe.sum() > 0 else 128.0,
               np.percentile(dtp[te], 95) if te.sum() > 0 else 128.0)


def train(cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_patients = build_hybrid_patients(cfg["otsu_dir"], cfg["unet_dir"])
    X = np.arange(len(all_patients))
    y = np.array([1 if "malignant" in cat.lower() else 0 for _, cat in all_patients])

    skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'=' * 40}\n========== FOLD {fold + 1}/{cfg['n_splits']} ==========\n{'=' * 40}")

        trn = [all_patients[i] for i in train_idx]
        val = [all_patients[i] for i in val_idx]
        trn_dl = DataLoader(HybridPatientDataset(trn, cfg["otsu_dir"], cfg["unet_dir"]), batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
        val_dl = DataLoader(HybridPatientDataset(val, cfg["otsu_dir"], cfg["unet_dir"]), batch_size=cfg["batch_size"], shuffle=False)

        enc = Encoder2D().to(device)
        dec = Decoder3D().to(device)
        
        opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=cfg["lr"], betas=cfg["betas"], weight_decay=1e-4)
        
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=30, min_lr=1e-6)
        
        # REMOVED SCALER (We are now in mathematically pure FP32)

        fold_dir = Path(cfg["ckpt_dir"]) / f"fold_{fold + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        best_dice, no_imp = 0.0, 0
        hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": [], "val_hd": []}
        val_angles = [-90., -45., 0., 45., 90.]

        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            enc.train()
            dec.train()
            ep_loss = 0.0

            for batch in tqdm(trn_dl, desc=f"F{fold + 1} E{epoch:03d} train", leave=False):
                inp10 = batch["input_10ch"].to(device)
                tgt5 = batch["target_5ch"].to(device)
                B = inp10.size(0)

                opt.zero_grad(set_to_none=True)
                
                # REMOVED AUTOCAST: Forward pass directly in FP32
                vol = dec(enc(inp10)).float()

                loss = torch.tensor(0.0, device=device)
                for i in range(5):
                    lo, hi = VIEW_WINDOWS[i]
                    for _ in range(cfg["n_per_view"]):
                        loss += dice_loss(render_projection(vol, torch.rand(B, device=device) * (hi - lo) + lo), tgt5[:, i:i + 1])
                loss /= (5 * cfg["n_per_view"])

                # PURE FP32 BACKWARD
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
                opt.step()
                
                ep_loss += loss.item()

            ep_loss /= max(len(trn_dl), 1)

            enc.eval()
            dec.eval()
            vl, vd, vh, cnt = 0., 0., 0., 0
            with torch.no_grad():
                for batch in tqdm(val_dl, desc=f"F{fold + 1} E{epoch:03d} val", leave=False):
                    inp10 = batch["input_10ch"].to(device)
                    tgt5 = batch["target_5ch"].to(device)
                    B = inp10.size(0)
                    
                    # REMOVED AUTOCAST: Forward pass directly in FP32
                    vol = dec(enc(inp10)).float()

                    for i in range(5):
                        proj = render_projection(vol, torch.full((B,), val_angles[i], device=device))
                        dl = dice_loss(proj, tgt5[:, i:i + 1])
                        vl += dl.item()
                        vd += (1 - dl).item()
                        pb, mb = (proj > 0.5).cpu().numpy(), (tgt5[:, i:i + 1] > 0.5).cpu().numpy()
                        for b in range(B):
                            vh += hd95(pb[b, 0], mb[b, 0])
                            cnt += 1

            vl /= max(len(val_dl) * 5, 1)
            vd /= max(len(val_dl) * 5, 1)
            vh /= max(cnt, 1)
            hist["epoch"].append(epoch)
            hist["train_loss"].append(ep_loss)
            hist["val_loss"].append(vl)
            hist["val_dice"].append(vd)
            hist["val_hd"].append(vh)
            sched.step(vd)

            print(f"F{fold + 1} E{epoch:03d} | loss={ep_loss:.4f} | vl={vl:.4f} vd={vd:.4f} hd={vh:.2f} | lr={opt.param_groups[0]['lr']:.6f} | {time.time() - t0:.1f}s")

            ckpt = {"epoch": epoch, "enc": enc.state_dict(), "dec": dec.state_dict(), "best_dice": max(best_dice, vd)}
            torch.save(ckpt, fold_dir / "3dbreastnet_last.pth")
            if vd > best_dice:
                best_dice = vd
                no_imp = 0
                torch.save(ckpt, fold_dir / "3dbreastnet_best.pth")
            else:
                no_imp += 1
                if no_imp >= cfg["patience"]:
                    print(f"Early stopping @ epoch {epoch}")
                    break

        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
        ax[0].plot(hist["epoch"], hist["val_loss"], label="val")
        ax[0].set_title(f"Fold {fold + 1} Dice Loss")
        ax[0].legend()
        ax[1].plot(hist["epoch"], hist["val_dice"], color="green")
        ax[1].set_title("Val Dice")
        ax[2].plot(hist["epoch"], hist["val_hd"], color="red")
        ax[2].set_title("Val HD95")
        plt.tight_layout()
        plt.savefig(fold_dir / f"history_fold{fold + 1}.png", dpi=150)
        plt.close()

        fold_results.append(best_dice)
        print(f"Fold {fold + 1} Best Dice: {best_dice:.4f}")

    print(f"\n{'=' * 40}\nFINAL K-FOLD RESULTS\n{'=' * 40}")
    for i, res in enumerate(fold_results):
        print(f"Fold {i + 1}: {res:.4f}")
    print(f"Average Best Dice: {np.mean(fold_results):.4f}")

if __name__ == "__main__":
    train(CFG)