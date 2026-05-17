# ════════════════════════════════════════════════════════════════════════════
# ## CELL F — Modified Training Loop (REPLACES existing training cell)
# ════════════════════════════════════════════════════════════════════════════
# Key changes vs original:
#   - Uses BreastNet3D_UNet (single model) instead of separate enc/dec
#   - Saves to 3dbreastnet_unet_best.pth / 3dbreastnet_unet_last.pth
#   - History keys: train_dice_loss, val_dice_loss, val_dice_score, val_hd95, prior_loss
#   - Conditional shape prior integration via USE_SHAPE_PRIOR flag
# ════════════════════════════════════════════════════════════════════════════

"""3DBreastNet — Training script for 128³ voxel reconstruction (U-Net edition)."""
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

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════════
# DATA (unchanged from v6)
# ════════════════════════════════════════════════════════════════
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
    print(f"Patients: {len(pd_)} | Complete: {len(groups)} | "
          f"Skipped: {skip} | B={nb_} M={nm}")
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

# ════════════════════════════════════════════════════════════════
# METRICS (unchanged)
# ════════════════════════════════════════════════════════════════
def hd95(p, t):
    if p.sum()==0 or t.sum()==0: return 128.0
    pe = p ^ scipy.ndimage.binary_erosion(p)
    te = t ^ scipy.ndimage.binary_erosion(t)
    dtp = scipy.ndimage.distance_transform_edt(~pe)
    dtt = scipy.ndimage.distance_transform_edt(~te)
    d1 = np.percentile(dtt[pe], 95) if pe.sum()>0 else 128.0
    d2 = np.percentile(dtp[te], 95) if te.sum()>0 else 128.0
    return max(d1, d2)

# ════════════════════════════════════════════════════════════════
# TRAIN (modified for BreastNet3D_UNet)
# ════════════════════════════════════════════════════════════════
def train(cfg):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # U-Net (frozen segmentor)
    unet = UNet().to(device)
    unet.load_state_dict(torch.load(cfg["unet_ckpt"], map_location=device))
    unet.eval()
    for p in unet.parameters(): p.requires_grad = False

    # Data
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

    # ── Model: BreastNet3D_UNet (single combined model) ──
    model = BreastNet3D_UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=cfg["betas"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=30, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Optional shape prior ──
    shape_prior_loss = None
    if USE_SHAPE_PRIOR:
        shape_prior_loss = ShapePriorLoss(volume_size=128, device=device)
        print("Shape prior ENABLED (weight={})".format(shape_prior_loss.weight))

    best_dice, no_imp = 0.0, 0
    hist = {"epoch":[], "train_dice_loss":[], "val_dice_loss":[],
            "val_dice_score":[], "val_hd95":[], "prior_loss":[]}
    val_angles = [-90., -45., 0., 45., 90.]

    for epoch in range(1, cfg["epochs"]+1):
        t0 = time.time()
        # ── train ──
        model.train()
        ep_loss, ep_prior = 0.0, 0.0
        for batch in tqdm(trn_dl, desc=f"E{epoch:03d} train", leave=False):
            m5 = batch["masks_5ch"].to(device)
            B = m5.size(0)
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
                    gt_mask = m5[:, i:i+1]
                    dl = dice_loss(proj, gt_mask)
                    bl = boundary_loss(proj, gt_mask)
                    loss = loss + dl + (2.0 * bl)

            loss = loss / (5*cfg["n_per_view"])

            # Conditional shape prior
            prior_val = 0.0
            if USE_SHAPE_PRIOR and shape_prior_loss is not None:
                pl = shape_prior_loss(vol)
                loss = loss + pl
                prior_val = pl.item()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if torch.isfinite(loss):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
            else:
                print(f"  ⚠ NaN loss in epoch {epoch}, skipping batch")
            scaler.update()
            opt.zero_grad(set_to_none=True)
            ep_loss += loss.item() if torch.isfinite(loss) else 0.0
            ep_prior += prior_val
        ep_loss /= max(len(trn_dl), 1)
        ep_prior /= max(len(trn_dl), 1)

        # ── val ──
        model.eval()
        vl, vd, vh, cnt = 0., 0., 0., 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"E{epoch:03d} val", leave=False):
                m5 = batch["masks_5ch"].to(device); B = m5.size(0)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    hull = compute_visual_hull(m5, device)
                vol = model(m5, hull)

                vol = vol.float()
                for i in range(5):
                    th = torch.full((B,), val_angles[i], device=device)
                    proj = render_projection(vol, th)
                    dl = dice_loss(proj, m5[:, i:i+1])
                    vl += dl.item(); vd += (1-dl).item()
                    pb = (proj>0.5).cpu().numpy()
                    mb = (m5[:, i:i+1]>0.5).cpu().numpy()
                    for b in range(B): vh += hd95(pb[b,0], mb[b,0]); cnt += 1
        vl /= max(len(val_dl)*5,1); vd /= max(len(val_dl)*5,1)
        vh /= max(cnt,1)
        elapsed = time.time()-t0

        hist["epoch"].append(epoch)
        hist["train_dice_loss"].append(ep_loss)
        hist["val_dice_loss"].append(vl)
        hist["val_dice_score"].append(vd)
        hist["val_hd95"].append(vh)
        hist["prior_loss"].append(ep_prior)
        sched.step(vd)

        lr_now = opt.param_groups[0]["lr"]
        print(f"E{epoch:03d} | loss={ep_loss:.4f} prior={ep_prior:.4f} | "
              f"vl={vl:.4f} vd={vd:.4f} hd={vh:.2f} | lr={lr_now:.6f} | {elapsed:.1f}s")

        # ── Checkpoint (new filenames for U-Net model) ──
        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "opt": opt.state_dict(), "best_dice": max(best_dice,vd),
                "cfg": cfg, "hist": hist}
        torch.save(ckpt, Path(cfg["ckpt_dir"])/"3dbreastnet_unet_last.pth")
        if vd > best_dice:
            best_dice = vd; no_imp = 0
            torch.save(ckpt, Path(cfg["ckpt_dir"])/"3dbreastnet_unet_best.pth")
            print(f"  ★ new best dice={best_dice:.4f}")
        else:
            no_imp += 1
            if no_imp >= cfg["patience"]:
                print(f"Early stopping @ epoch {epoch}"); break

    # ── plot ──
    fig, ax = plt.subplots(1, 4, figsize=(24, 5))
    ax[0].plot(hist["epoch"], hist["train_dice_loss"], label="train")
    ax[0].plot(hist["epoch"], hist["val_dice_loss"], label="val")
    ax[0].set_title("Dice Loss"); ax[0].legend()
    ax[1].plot(hist["epoch"], hist["val_dice_score"], color="green")
    ax[1].set_title("Val Dice Score")
    ax[2].plot(hist["epoch"], hist["val_hd95"], color="red")
    ax[2].set_title("Val HD95")
    ax[3].plot(hist["epoch"], hist["prior_loss"], color="purple")
    ax[3].set_title("Shape Prior Loss")
    for a in ax: a.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(Path(cfg["ckpt_dir"])/"training_history.png", dpi=150)
    print(f"Plot saved. Best dice={best_dice:.4f}")

if __name__ == "__main__":
    train(CFG)


# ════════════════════════════════════════════════════════════════════════════
# ## CELL G — Evaluation Metrics (NEW — after training/plotting cells)
# ════════════════════════════════════════════════════════════════════════════

import numpy as np
from scipy.spatial.distance import directed_hausdorff

def compute_metrics(pred_proj, target_sil, threshold=0.5):
    """
    pred_proj:   (H, W) float in [0, 1]  — rendered 2D projection
    target_sil:  (H, W) binary           — input silhouette
    Returns dict with Accuracy, Dice, Jaccard, Hausdorff.
    """
    pred_bin = (pred_proj >= threshold).astype(np.float32)
    tgt      = target_sil.astype(np.float32)

    TP = (pred_bin * tgt).sum()
    TN = ((1 - pred_bin) * (1 - tgt)).sum()
    FP = (pred_bin * (1 - tgt)).sum()
    FN = ((1 - pred_bin) * tgt).sum()
    N  = pred_bin.size

    accuracy = (TP + TN) / N
    dice     = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    jaccard  = TP / (TP + FP + FN + 1e-6)

    pred_pts = np.argwhere(pred_bin > 0).astype(float)
    tgt_pts  = np.argwhere(tgt  > 0).astype(float)
    if len(pred_pts) == 0 or len(tgt_pts) == 0:
        hausdorff = float('nan')
    else:
        hausdorff = max(
            directed_hausdorff(pred_pts, tgt_pts)[0],
            directed_hausdorff(tgt_pts,  pred_pts)[0]
        )

    return {
        'Accuracy':           accuracy,
        'Dice Index':         dice,
        'Jaccard Index':      jaccard,
        'Hausdorff Distance': hausdorff,
    }


# ── Evaluation loop ──
model.eval()
view_names   = ['RL (−90°)', 'RO (−45°)', 'F (0°)', 'LO (+45°)', 'LL (+90°)']
std_angles   = [-90, -45, 0, 45, 90]
results_per_view = {v: [] for v in view_names}

with torch.no_grad():
    for batch in val_dl:          # uses val_dl from training; swap to test_loader if available
        masks = batch["masks_5ch"].to(device)     # (B, 5, H, W)
        hull  = compute_visual_hull(masks, device)
        V_pred = model(masks, hull)               # (B, 1, 128, 128, 128)

        for view_idx, (vname, angle) in enumerate(zip(view_names, std_angles)):
            proj = render_projection(V_pred, angle)
            proj_np = proj.squeeze(1).cpu().float().numpy()  # (B, H, W)
            sil_np  = masks[:, view_idx].cpu().numpy()       # (B, H, W)

            for b in range(proj_np.shape[0]):
                m = compute_metrics(proj_np[b], sil_np[b])
                results_per_view[vname].append(m)

# ── Print table ──
metric_keys = ['Accuracy', 'Dice Index', 'Jaccard Index', 'Hausdorff Distance']
print(f"{'View':<15} {'Accuracy':>10} {'Dice':>10} {'Jaccard':>10} {'Hausdorff':>12}")
print("-" * 60)
all_vals = {k: [] for k in metric_keys}

for vname in view_names:
    vals = results_per_view[vname]
    row  = {k: np.nanmean([v[k] for v in vals]) for k in metric_keys}
    for k in metric_keys:
        all_vals[k].extend([v[k] for v in vals])
    print(f"{vname:<15} {row['Accuracy']:>10.4f} {row['Dice Index']:>10.4f} "
          f"{row['Jaccard Index']:>10.4f} {row['Hausdorff Distance']:>12.4f}")

print("-" * 60)
overall = {k: np.nanmean(all_vals[k]) for k in metric_keys}
print(f"{'Overall':<15} {overall['Accuracy']:>10.4f} {overall['Dice Index']:>10.4f} "
      f"{overall['Jaccard Index']:>10.4f} {overall['Hausdorff Distance']:>12.4f}")


# ════════════════════════════════════════════════════════════════════════════
# ## CELL H — Thermal Projection (NEW — after evaluation metrics)
# ════════════════════════════════════════════════════════════════════════════

import torch
import numpy as np
from scipy.spatial import cKDTree

def estimate_view_angle(V_pred_np, sil_np, angle_range=(-90, 90), step=1):
    """Find angle θ̂ minimising Dice loss between silhouette and projection."""
    best_angle, best_loss = 0, float('inf')
    angles = np.arange(angle_range[0], angle_range[1] + step, step)
    V_t = torch.from_numpy(V_pred_np).unsqueeze(0).unsqueeze(0).float()
    for angle in angles:
        proj = render_projection(V_t, angle).squeeze().numpy()
        inter = (proj * sil_np).sum()
        loss  = 1 - (2 * inter) / (proj.sum() + sil_np.sum() + 1e-6)
        if loss < best_loss:
            best_loss, best_angle = loss, angle
    return best_angle


def overlay_temperatures_on_volume(V_pred_np, thermal_images, estimated_angles,
                                   volume_size=128):
    """Maps 2D temperatures onto 3D silhouette surface via ray-casting."""
    D, H, W   = V_pred_np.shape
    temp_vol  = np.full((D, H, W), np.nan, dtype=np.float32)
    count_vol = np.zeros((D, H, W), dtype=np.float32)
    vox_coords = np.argwhere(V_pred_np > 0.5).astype(np.float32)

    for angle, temp_2d in zip(estimated_angles, thermal_images):
        if temp_2d is None:
            continue
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        centre = np.array([D / 2, H / 2, W / 2])
        shifted = vox_coords - centre
        d, h, w = shifted[:, 0], shifted[:, 1], shifted[:, 2]
        w_rot =  w * cos_t + d * sin_t
        d_rot = -w * sin_t + d * cos_t
        front_mask = d_rot > 0
        if front_mask.sum() == 0:
            continue
        h_pix = (h[front_mask] + centre[1]).astype(int)
        w_pix = (w_rot[front_mask] + centre[2]).astype(int)
        temp_h, temp_w = temp_2d.shape
        scale_h, scale_w = temp_h / H, temp_w / W
        h_img = np.clip((h_pix * scale_h).astype(int), 0, temp_h - 1)
        w_img = np.clip((w_pix * scale_w).astype(int), 0, temp_w - 1)
        orig_coords = np.argwhere(V_pred_np > 0.5)[front_mask]
        for i, (vd, vh, vw) in enumerate(orig_coords):
            t = temp_2d[h_img[i], w_img[i]]
            if not np.isnan(t):
                if np.isnan(temp_vol[vd, vh, vw]):
                    temp_vol[vd, vh, vw]  = t
                    count_vol[vd, vh, vw] = 1
                else:
                    temp_vol[vd, vh, vw] = (
                        temp_vol[vd, vh, vw] * count_vol[vd, vh, vw] + t)
                    count_vol[vd, vh, vw] += 1
                    temp_vol[vd, vh, vw] /= count_vol[vd, vh, vw]

    # KNN interpolation for occluded voxels
    occupied_mask   = ~np.isnan(temp_vol) & (V_pred_np > 0.5)
    unoccupied_mask =  np.isnan(temp_vol) & (V_pred_np > 0.5)
    if occupied_mask.sum() > 0 and unoccupied_mask.sum() > 0:
        filled_coords = np.argwhere(occupied_mask).astype(np.float32)
        filled_temps  = temp_vol[occupied_mask]
        query_coords  = np.argwhere(unoccupied_mask).astype(np.float32)
        tree  = cKDTree(filled_coords)
        _, idx = tree.query(query_coords, k=1)
        temp_vol[unoccupied_mask] = filled_temps[idx]

    return temp_vol


# ── Run for one patient (adapt paths/variables to your setup) ──
# patient_id  = 'example_patient_id'  # replace with actual ID variable
# views_order = ['RL', 'RO', 'F', 'LO', 'LL']
# std_angles  = [-90, -45, 0, 45, 90]
#
# V_pred_np = np.load(f'outputs/{patient_id}_volume_soft.npy')
# V_bin     = (V_pred_np > 0.5).astype(np.float32)
#
# silhouettes = [...]        # list of 5 binary (128, 128) arrays
# thermal_abs = [...]        # list of 5 float (H_orig, W_orig) arrays in °C
#
# estimated_angles = []
# for view_idx in range(5):
#     angle_est = estimate_view_angle(V_bin, silhouettes[view_idx],
#                                     angle_range=(std_angles[view_idx] - 20,
#                                                  std_angles[view_idx] + 20))
#     estimated_angles.append(angle_est)
#     print(f"View {views_order[view_idx]}: estimated angle = {angle_est:.1f}°")
#
# temp_volume = overlay_temperatures_on_volume(V_bin, thermal_abs, estimated_angles)
# np.save(f'outputs/{patient_id}_thermal_overlay.npy', temp_volume)
# print(f"Thermal overlay saved: {temp_volume.shape}, "
#       f"temp range [{np.nanmin(temp_volume):.1f}, {np.nanmax(temp_volume):.1f}] °C")
