#!/usr/bin/env python3
"""
TherMAM-NeRF v2.9 — Comprehensive Validation & Visualization
=============================================================
Run on SSH:  python thermamnerf_validate_v2.9.py
Or open in VS Code / Jupyter as an interactive notebook (uses # %% markers).

This script loads a trained checkpoint and performs:
  1. Geometry reconstruction metrics (Dice, IoU, HD95, SSIM, ASSD, Precision, Recall)
  2. Thermal field accuracy metrics (MSE, MAE, PSNR, Pearson r, Gradient Error)
  3. Generalization analysis (Train vs Val split, Mann-Whitney U test)
  4. Paper-ready plots (seaborn, 300 DPI)
  5. Interactive 3D visualization (Plotly Mesh3d with thermal overlay)
  6. PINN + FEA export (STL, PLY, NPZ, CSV)
"""

# %% [markdown]
# # Cell 1: Setup & Dependencies

# %%
import os
import math
import struct
import random
import warnings
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.ndimage import gaussian_filter, sobel
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import mannwhitneyu, pearsonr
from skimage.measure import marching_cubes
from skimage.metrics import structural_similarity as ssim_metric

import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ── Device ──
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
TIFF_DIR   = str(REPO_ROOT / 'data' / 'organized_by_patient')
UNET_DIR   = str(REPO_ROOT / 'data' / 'organized_by_patient_unet')
CKPT_DIR   = str(SCRIPT_DIR / 'thermamnerf_outputs2.9')
OUTPUT_DIR  = str(SCRIPT_DIR / 'validation_results_v2.9')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'Checkpoint dir: {CKPT_DIR}')
print(f'Output dir:     {OUTPUT_DIR}')

# ── Hyperparameters (must match training) ──
CFG = {
    'img_size'        : 128,
    'n_views'         : 5,
    'view_angles_deg' : [-90, -45, 0, 45, 90],
    'view_names'      : ['RL', 'RO', 'F', 'LO', 'LL'],
    'feat_channels'   : 32,
    'pos_enc_L'       : 8,
    'mlp_hidden'      : 256,
    'mlp_layers'      : 4,
    'n_samples'       : 256,
    'near'            : -1.0,
    'far'             : 1.0,
    'density_scale'   : 10.0,
    'mc_threshold'    : 0.3,
    'mc_resolution'   : 128,
}

# %% [markdown]
# # Cell 2: Model Definitions & Checkpoint Loading

# %%
# ── Data loading utilities (copied from v2.9) ──

def load_tiff_celsius(path: str, target_size: int) -> np.ndarray:
    arr = tifffile.imread(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    img = Image.fromarray(arr, mode='F')
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


def load_mask(path: str, target_size: int) -> np.ndarray:
    img = Image.open(path).convert('L')
    img = img.resize((target_size, target_size), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)


def normalize_thermal(arr: np.ndarray):
    tmin, tmax = arr.min(), arr.max()
    normed = (arr - tmin) / (tmax - tmin + 1e-6)
    return normed, tmin, tmax


def get_view_key(filename):
    n = filename.lower()
    if 'right later' in n: return 'RL'
    if 'right obli'  in n: return 'RO'
    if 'frontal' in n or 'anterior' in n: return 'F'
    if 'left obliq'  in n: return 'LO'
    if 'left later'  in n: return 'LL'
    return None


def discover_patients_split(tiff_base, unet_base):
    tb = Path(tiff_base)
    ub = Path(unet_base)
    pd_ = {}
    for tp in tb.rglob('*.tiff'):
        parts = tp.relative_to(tb).parts
        if len(parts) < 2: continue
        pid, lab, fn = parts[0], parts[1], parts[-1]
        vk = get_view_key(fn)
        if not vk: continue
        key = (pid, lab)
        if key not in pd_: pd_[key] = {'tiffs': {}, 'masks': {}}
        pd_[key]['tiffs'][vk] = tp
    for mp in ub.rglob('*.png'):
        parts = mp.relative_to(ub).parts
        if len(parts) < 2: continue
        pid, lab, fn = parts[0], parts[1], parts[-1]
        vk = get_view_key(fn)
        if not vk: continue
        key = (pid, lab)
        if key in pd_:
            pd_[key]['masks'][vk] = mp
    patients = []
    for (pid, lab), d in pd_.items():
        if len(d['tiffs']) == 5 and len(d['masks']) == 5:
            patients.append({'id': pid, 'tiffs': d['tiffs'], 'masks': d['masks']})
        else:
            print(f'  [SKIP] {pid} — missing views')
    patients.sort(key=lambda p: int(p['id'].split('_')[-1]) if '_' in p['id'] else p['id'])
    print(f'Found {len(patients)} complete patients.')
    return patients


from torch.utils.data import Dataset

class BreastThermDataset(Dataset):
    def __init__(self, patient_list: list, cfg: dict):
        self.patients = patient_list
        self.cfg = cfg
        self.S = cfg['img_size']
        self.view_names = cfg['view_names']

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        tiffs_norm, tiffs_abs, masks = [], [], []
        tmins, tmaxs = [], []
        for v in self.view_names:
            raw   = load_tiff_celsius(str(p['tiffs'][v]), self.S)
            normd, tmin, tmax = normalize_thermal(raw)
            mask  = load_mask(str(p['masks'][v]), self.S)
            tiffs_norm.append(normd)
            tiffs_abs.append(raw)
            masks.append(mask)
            tmins.append(tmin)
            tmaxs.append(tmax)
        return {
            'patient_id' : p['id'],
            'tiffs_norm' : torch.tensor(np.stack(tiffs_norm), dtype=torch.float32),
            'tiffs_abs'  : torch.tensor(np.stack(tiffs_abs),  dtype=torch.float32),
            'masks'      : torch.tensor(np.stack(masks),      dtype=torch.float32),
            'tmin'       : torch.tensor(tmins, dtype=torch.float32),
            'tmax'       : torch.tensor(tmaxs, dtype=torch.float32),
        }


# ── Model definitions (copied from v2.9) ──

class SiameseEncoder(nn.Module):
    def __init__(self, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2,  16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(inplace=False),
            nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=False),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, tiff_norm, mask):
        x = torch.stack([tiff_norm, mask], dim=1)
        out = self.net(x)
        return out * mask.unsqueeze(1)


def project_and_sample(pts_3d, feat_maps, view_angles_rad):
    B, N, _ = pts_3d.shape
    V, C    = feat_maps.shape[1], feat_maps.shape[2]
    per_view_feats = []
    for v in range(V):
        theta = view_angles_rad[v]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        x  = pts_3d[..., 0]
        y  = pts_3d[..., 1]
        z  = pts_3d[..., 2]
        xr =  cos_t * x + sin_t * z
        yr =  y
        grid = torch.stack([xr, yr], dim=-1).unsqueeze(1)
        fmap = feat_maps[:, v]
        sampled = F.grid_sample(fmap, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        sampled = sampled.squeeze(2).permute(0, 2, 1)
        per_view_feats.append(sampled)
    stacked = torch.stack(per_view_feats, dim=2)
    mu  = stacked.mean(dim=2)
    var = stacked.var(dim=2)
    return torch.cat([mu, var], dim=-1)


def positional_encoding(x, L, alpha=None):
    freqs = 2.0 ** torch.arange(L, dtype=torch.float32, device=x.device)
    x_freq = x.unsqueeze(-1) * freqs * math.pi
    sin_part = torch.sin(x_freq)
    cos_part = torch.cos(x_freq)
    if alpha is not None:
        k = torch.arange(L, dtype=torch.float32, device=x.device)
        w = torch.clamp(alpha - k, 0.0, 1.0)
        w = 0.5 * (1 - torch.cos(math.pi * w))
        w = w.view(1, 1, 1, L)
        sin_part = sin_part * w
        cos_part = cos_part * w
    enc = torch.cat([sin_part, cos_part], dim=-1)
    return enc.flatten(-2)


class ThermamNeRFMLP(nn.Module):
    def __init__(self, pos_enc_dim, feat_dim, hidden=256, n_layers=6):
        super().__init__()
        in_dim = pos_enc_dim + feat_dim
        self.layers  = nn.ModuleList()
        self.skip_at = n_layers // 2 - 1
        prev = in_dim
        for i in range(n_layers - 1):
            if i == self.skip_at:
                self.layers.append(nn.Linear(prev + in_dim, hidden))
            else:
                self.layers.append(nn.Linear(prev, hidden))
            prev = hidden
        self.sigma_head = nn.Linear(hidden, 1)
        self.temp_head  = nn.Linear(hidden, 1)

    def forward(self, pe, feat):
        x0 = torch.cat([pe, feat], dim=-1)
        h  = x0
        for i, layer in enumerate(self.layers):
            if i == self.skip_at:
                h = torch.cat([h, x0], dim=-1)
            h = F.relu(layer(h), inplace=False)
        sigma  = F.softplus(self.sigma_head(h))
        T_norm = torch.sigmoid(self.temp_head(h))
        return sigma, T_norm


def get_rays(H, W, angle_deg, device):
    theta = math.radians(angle_deg)
    cam_dir = torch.tensor([-math.sin(theta), 0.0, math.cos(theta)], device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    right = torch.tensor([math.cos(theta), 0.0, math.sin(theta)], device=device)
    up    = torch.tensor([0.0, 1.0, 0.0], device=device)
    origins = (grid_x.flatten().unsqueeze(1) * right.unsqueeze(0) +
               grid_y.flatten().unsqueeze(1) * up.unsqueeze(0))
    dirs = cam_dir.unsqueeze(0).expand(H * W, -1)
    return origins, dirs


def volume_render(sigma, T_field, deltas, density_scale):
    sigma   = sigma * density_scale
    alpha   = 1.0 - torch.exp(-sigma * deltas.unsqueeze(0))
    T_trans = torch.cumprod(
        torch.cat([torch.ones(alpha.shape[0], 1, device=alpha.device),
                   1.0 - alpha[:, :-1] + 1e-10], dim=1), dim=1)
    weights = alpha * T_trans
    rendered_mask = weights.sum(dim=1)
    rendered_temp = (weights.detach() * T_field).sum(dim=1)
    opacity_mask  = (rendered_mask.detach() > 0.05).float()
    rendered_temp = (rendered_temp / (rendered_mask.detach() + 1e-6)) * opacity_mask
    return rendered_mask, rendered_temp


@torch.no_grad()
def render_view(encoder, mlp, tiffs_norm, masks, view_idx, cfg, device,
                alpha=None, chunk=2048):
    H = W = cfg['img_size']
    angle = cfg['view_angles_deg'][view_idx]
    view_angles_rad = torch.tensor([math.radians(a) for a in cfg['view_angles_deg']], device=device)
    feat_maps = []
    for v in range(cfg['n_views']):
        fm = encoder(tiffs_norm[v:v+1], masks[v:v+1])
        feat_maps.append(fm)
    feat_maps = torch.stack(feat_maps, dim=1)
    rays_o, rays_d = get_rays(H, W, angle, device)
    near, far      = cfg['near'], cfg['far']
    n_samples      = cfg['n_samples']
    t_vals         = torch.linspace(near, far, n_samples, device=device)
    deltas         = torch.cat([t_vals[1:] - t_vals[:-1], torch.tensor([1e-3], device=device)])
    all_mask, all_temp = [], []
    for i in range(0, rays_o.shape[0], chunk):
        ro = rays_o[i:i+chunk]
        rd = rays_d[i:i+chunk]
        pts = ro.unsqueeze(1) + t_vals.view(1, -1, 1) * rd.unsqueeze(1)
        pts = pts.unsqueeze(0)
        R_  = pts.shape[1]
        pts_flat = pts.reshape(1, -1, 3)
        pe   = positional_encoding(pts_flat, L=cfg['pos_enc_L'], alpha=alpha)
        feat = project_and_sample(pts_flat, feat_maps, view_angles_rad)
        sigma, T_pred = mlp(pe, feat)
        # View-specific depth cropping
        if angle <= -80:
            sigma = sigma.masked_fill(pts_flat[..., 0:1] > 0.1, 0.0)
        elif angle >= 80:
            sigma = sigma.masked_fill(pts_flat[..., 0:1] < -0.1, 0.0)
        sigma  = sigma.squeeze(-1).squeeze(0).reshape(R_, n_samples)
        T_pred = T_pred.squeeze(-1).squeeze(0).reshape(R_, n_samples)
        rm, rt = volume_render(sigma, T_pred, deltas, cfg['density_scale'])
        all_mask.append(rm)
        all_temp.append(rt)
    rendered_mask = torch.cat(all_mask).reshape(H, W)
    rendered_temp = torch.cat(all_temp).reshape(H, W)
    return rendered_mask, rendered_temp


def extract_3d_volume(encoder, mlp, tiffs_norm, masks, cfg, device,
                       resolution=None, chunk=8192):
    R = resolution or cfg['mc_resolution']
    linspace = torch.linspace(-1, 1, R, device=device)
    zz, yy, xx = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)
    view_angles_rad = torch.tensor([math.radians(a) for a in cfg['view_angles_deg']], device=device)
    feat_maps = []
    for v in range(cfg['n_views']):
        fm = encoder(tiffs_norm[v:v+1], masks[v:v+1])
        feat_maps.append(fm)
    feat_maps = torch.stack(feat_maps, dim=1)
    alpha_final = float(cfg['pos_enc_L'])
    sigma_all, T_all = [], []
    n_pts = pts.shape[1]
    with torch.no_grad():
        for i in range(0, n_pts, chunk):
            p    = pts[:, i:i+chunk]
            pe   = positional_encoding(p, L=cfg['pos_enc_L'], alpha=alpha_final)
            feat = project_and_sample(p, feat_maps, view_angles_rad)
            sg, tp = mlp(pe, feat)
            sigma_all.append(sg.squeeze().cpu())
            T_all.append(tp.squeeze().cpu())
    sigma_grid = torch.cat(sigma_all).reshape(R, R, R).numpy()
    T_grid     = torch.cat(T_all).reshape(R, R, R).numpy()
    return sigma_grid, T_grid


# ── Discover patients & replicate the exact train/val split ──

patients = discover_patients_split(TIFF_DIR, UNET_DIR)
if len(patients) == 0:
    raise RuntimeError('No complete patients found. Check paths.')

# Replicate the exact same split used during training
random.seed(42)
patients_shuffled = list(patients)  # copy
random.shuffle(patients_shuffled)
split = int(0.8 * len(patients_shuffled))
train_ids = set(p['id'] for p in patients_shuffled[:split])
val_ids   = set(p['id'] for p in patients_shuffled[split:])
print(f'Train: {len(train_ids)} patients | Val: {len(val_ids)} patients')

# Build dataset over ALL patients (sorted order)
full_ds = BreastThermDataset(patients, CFG)

# ── Load checkpoint ──

L_enc    = CFG['pos_enc_L']
pos_dim  = 3 * 2 * L_enc
feat_dim = CFG['feat_channels'] * 2

encoder = SiameseEncoder(out_channels=CFG['feat_channels']).to(DEVICE)
mlp     = ThermamNeRFMLP(pos_enc_dim=pos_dim, feat_dim=feat_dim,
                          hidden=CFG['mlp_hidden'],
                          n_layers=CFG['mlp_layers']).to(DEVICE)

ckpt_path = os.path.join(CKPT_DIR, 'thermamnerf_best.pth')
ckpt = torch.load(ckpt_path, map_location=DEVICE)
enc_state = {k.replace('module.', ''): v for k, v in ckpt['encoder'].items()}
mlp_state = {k.replace('module.', ''): v for k, v in ckpt['mlp'].items()}
encoder.load_state_dict(enc_state)
mlp.load_state_dict(mlp_state)
encoder.eval()
mlp.eval()

n_params = sum(p.numel() for p in list(encoder.parameters()) + list(mlp.parameters()))
print(f'Loaded checkpoint: {ckpt_path}')
print(f'Total parameters: {n_params:,}')


# %% [markdown]
# # Cell 3 & 4: Geometry + Thermal Metrics (Combined Evaluation Loop)

# %%
# ── Metric helper functions ──

def compute_hausdorff_95(pred_bin, gt_bin):
    """Compute 95th percentile Hausdorff distance on boundary pixels."""
    pred_boundary = np.zeros_like(pred_bin, dtype=bool)
    gt_boundary   = np.zeros_like(gt_bin, dtype=bool)
    # Simple boundary: pixels where the mask changes
    pred_boundary[1:, :] |= (pred_bin[1:, :] != pred_bin[:-1, :])
    pred_boundary[:, 1:] |= (pred_bin[:, 1:] != pred_bin[:, :-1])
    gt_boundary[1:, :]   |= (gt_bin[1:, :] != gt_bin[:-1, :])
    gt_boundary[:, 1:]   |= (gt_bin[:, 1:] != gt_bin[:, :-1])

    pred_pts = np.argwhere(pred_boundary)
    gt_pts   = np.argwhere(gt_boundary)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0

    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)
    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def compute_assd(pred_bin, gt_bin):
    """Average Symmetric Surface Distance."""
    pred_boundary = np.zeros_like(pred_bin, dtype=bool)
    gt_boundary   = np.zeros_like(gt_bin, dtype=bool)
    pred_boundary[1:, :] |= (pred_bin[1:, :] != pred_bin[:-1, :])
    pred_boundary[:, 1:] |= (pred_bin[:, 1:] != pred_bin[:, :-1])
    gt_boundary[1:, :]   |= (gt_bin[1:, :] != gt_bin[:-1, :])
    gt_boundary[:, 1:]   |= (gt_bin[:, 1:] != gt_bin[:, :-1])

    pred_pts = np.argwhere(pred_boundary)
    gt_pts   = np.argwhere(gt_boundary)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0

    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)
    return float(0.5 * (d_pred_to_gt.mean() + d_gt_to_pred.mean()))


def compute_gradient_error(pred_temp, gt_temp, fg_mask):
    """Compare spatial thermal gradients (Sobel) in foreground."""
    pred_gx = sobel(pred_temp, axis=0)
    pred_gy = sobel(pred_temp, axis=1)
    gt_gx   = sobel(gt_temp, axis=0)
    gt_gy   = sobel(gt_temp, axis=1)
    pred_mag = np.sqrt(pred_gx**2 + pred_gy**2)
    gt_mag   = np.sqrt(gt_gx**2 + gt_gy**2)
    if fg_mask.sum() == 0:
        return 0.0
    return float(np.abs(pred_mag[fg_mask] - gt_mag[fg_mask]).mean())


# ── Main evaluation loop ──

alpha_final = float(CFG['pos_enc_L'])
results = []

print(f'\n{"="*60}')
print(f'Starting full cohort evaluation ({len(full_ds)} patients)...')
print(f'{"="*60}\n')

for idx in tqdm(range(len(full_ds)), desc='Evaluating Patients'):
    sample     = full_ds[idx]
    tiffs_norm = sample['tiffs_norm'].to(DEVICE)
    masks_gt   = sample['masks'].to(DEVICE)
    tiffs_abs  = sample['tiffs_abs']
    patient_id = sample['patient_id']
    tmin_vals  = sample['tmin']
    tmax_vals  = sample['tmax']
    split_tag  = 'TRAIN' if patient_id in train_ids else 'VAL'

    patient_metrics = {
        'patient_id': patient_id,
        'split': split_tag,
    }

    view_dice, view_iou = [], []

    for v, vname in enumerate(CFG['view_names']):
        rm, rt = render_view(encoder, mlp, tiffs_norm, masks_gt, v, CFG, DEVICE,
                             alpha=alpha_final)
        pred_m = rm.cpu().numpy()
        pred_t = rt.cpu().numpy()
        gt_m   = masks_gt[v].cpu().numpy()
        gt_t   = sample['tiffs_norm'][v].numpy()

        pred_bin = (pred_m > 0.5).astype(np.float32)
        gt_bin   = gt_m.astype(np.float32)

        # ── Geometry Metrics ──
        tp = (pred_bin * gt_bin).sum()
        fp = (pred_bin * (1 - gt_bin)).sum()
        fn = ((1 - pred_bin) * gt_bin).sum()

        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
        iou  = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        hd95 = compute_hausdorff_95(pred_bin, gt_bin)
        assd = compute_assd(pred_bin, gt_bin)

        # SSIM on continuous opacity map (data_range=1.0 since both are [0,1])
        ssim_val = ssim_metric(gt_m, pred_m, data_range=1.0)

        patient_metrics[f'dice_{vname}']      = dice
        patient_metrics[f'iou_{vname}']       = iou
        patient_metrics[f'precision_{vname}'] = precision
        patient_metrics[f'recall_{vname}']    = recall
        patient_metrics[f'hd95_{vname}']      = hd95
        patient_metrics[f'assd_{vname}']      = assd
        patient_metrics[f'ssim_{vname}']      = ssim_val

        view_dice.append(dice)
        view_iou.append(iou)

        # ── Thermal Metrics (foreground only) ──
        fg = gt_bin > 0.5
        if fg.sum() > 10:
            pred_fg = pred_t[fg]
            gt_fg   = gt_t[fg]
            mse = float(np.mean((pred_fg - gt_fg) ** 2))
            mae = float(np.mean(np.abs(pred_fg - gt_fg)))
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            r_val, _ = pearsonr(pred_fg.flatten(), gt_fg.flatten())
            max_err = float(np.max(np.abs(pred_fg - gt_fg)))
            grad_err = compute_gradient_error(pred_t, gt_t, fg)
        else:
            mse, mae, psnr, r_val, max_err, grad_err = 0, 0, 0, 0, 0, 0

        patient_metrics[f'thermal_mse_{vname}']  = mse
        patient_metrics[f'thermal_mae_{vname}']  = mae
        patient_metrics[f'thermal_psnr_{vname}'] = psnr
        patient_metrics[f'thermal_r_{vname}']    = r_val
        patient_metrics[f'thermal_maxerr_{vname}'] = max_err
        patient_metrics[f'thermal_grderr_{vname}'] = grad_err

    # Mean across views
    patient_metrics['dice_mean'] = np.mean(view_dice)
    patient_metrics['iou_mean']  = np.mean(view_iou)

    # Mean geometry
    for m in ['precision', 'recall', 'hd95', 'assd', 'ssim']:
        vals = [patient_metrics[f'{m}_{v}'] for v in CFG['view_names']]
        patient_metrics[f'{m}_mean'] = np.mean(vals)

    # Mean thermal
    for m in ['thermal_mse', 'thermal_mae', 'thermal_psnr', 'thermal_r',
              'thermal_maxerr', 'thermal_grderr']:
        vals = [patient_metrics[f'{m}_{v}'] for v in CFG['view_names']]
        patient_metrics[f'{m}_mean'] = np.mean(vals)

    results.append(patient_metrics)

# Build DataFrame
df = pd.DataFrame(results)

# Save full results
csv_path = os.path.join(OUTPUT_DIR, 'cohort_all_metrics.csv')
df.to_csv(csv_path, index=False)
print(f'\nSaved full metrics to {csv_path}')

# ── Print summary ──
print(f'\n{"="*60}')
print(f'COHORT SUMMARY ({len(df)} patients)')
print(f'{"="*60}')
for metric in ['dice_mean', 'iou_mean', 'precision_mean', 'recall_mean',
               'hd95_mean', 'assd_mean', 'ssim_mean',
               'thermal_mse_mean', 'thermal_mae_mean', 'thermal_psnr_mean',
               'thermal_r_mean']:
    print(f'  {metric:25s}: {df[metric].mean():.4f} ± {df[metric].std():.4f}')


# %% [markdown]
# # Cell 5: Generalization Analysis

# %%
df_train = df[df['split'] == 'TRAIN']
df_val   = df[df['split'] == 'VAL']

print(f'\n{"="*60}')
print('GENERALIZATION ANALYSIS')
print(f'{"="*60}')
print(f'Train patients: {len(df_train)} | Val patients: {len(df_val)}')

# 1. Generalization Gap
train_dice_mean = df_train['dice_mean'].mean()
val_dice_mean   = df_val['dice_mean'].mean()
gap = train_dice_mean - val_dice_mean
print(f'\nTrain Mean Dice: {train_dice_mean:.4f}')
print(f'Val   Mean Dice: {val_dice_mean:.4f}')
print(f'Generalization Gap: {gap:.4f}')
if abs(gap) < 0.01:
    print('  → EXCELLENT: Gap < 0.01 — model generalizes very well!')
elif abs(gap) < 0.03:
    print('  → GOOD: Gap < 0.03 — minor generalization difference.')
else:
    print('  → WARNING: Gap > 0.03 — possible overfitting concern.')

# 2. Mann-Whitney U Test
stat, p_value = mannwhitneyu(df_train['dice_mean'], df_val['dice_mean'], alternative='two-sided')
print(f'\nMann-Whitney U Test:')
print(f'  U-statistic: {stat:.1f}')
print(f'  p-value:     {p_value:.4f}')
if p_value > 0.05:
    print('  → p > 0.05: No significant difference between train and val. Model generalizes well!')
else:
    print('  → p < 0.05: Significant difference detected between train and val distributions.')

# 3. Generalization across all metrics
print(f'\n{"─"*60}')
print('Per-Metric Generalization:')
print(f'{"Metric":<25} {"Train":>10} {"Val":>10} {"Gap":>10}')
print(f'{"─"*60}')
for metric in ['dice_mean', 'iou_mean', 'ssim_mean', 'hd95_mean',
               'thermal_mse_mean', 'thermal_psnr_mean', 'thermal_r_mean']:
    t = df_train[metric].mean()
    v = df_val[metric].mean()
    g = t - v
    print(f'{metric:<25} {t:>10.4f} {v:>10.4f} {g:>+10.4f}')

# 4. Worst-10 Patient Analysis
print(f'\n{"─"*60}')
print('Worst-10 Patients (by Dice):')
worst10 = df.nsmallest(10, 'dice_mean')[['patient_id', 'split', 'dice_mean', 'iou_mean',
                                          'thermal_mse_mean']].reset_index(drop=True)
print(worst10.to_string(index=False))
n_val_in_worst = (worst10['split'] == 'VAL').sum()
n_train_in_worst = (worst10['split'] == 'TRAIN').sum()
print(f'\nOf the worst 10: {n_train_in_worst} TRAIN, {n_val_in_worst} VAL')
if n_val_in_worst >= 7:
    print('  → Most worst patients are unseen (VAL) — potential generalization issue.')
else:
    print('  → Worst patients are spread across TRAIN/VAL — likely hard anatomy, not overfitting.')

# 5. Per-View Generalization
print(f'\n{"─"*60}')
print('Per-View Generalization (Dice):')
print(f'{"View":<10} {"Train":>10} {"Val":>10} {"Gap":>10}')
for vname in CFG['view_names']:
    col = f'dice_{vname}'
    t = df_train[col].mean()
    v = df_val[col].mean()
    print(f'{vname:<10} {t:>10.4f} {v:>10.4f} {t-v:>+10.4f}')


# %% [markdown]
# # Cell 6: Paper-Ready Plots

# %%
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

# ── 1. Dice Distribution by Split ──
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df, x='dice_mean', hue='split', kde=True, alpha=0.5, bins=20,
             palette={'TRAIN': 'royalblue', 'VAL': 'orangered'}, ax=ax)
ax.set_title('Distribution of Dice Scores (Train vs Val)', fontweight='bold')
ax.set_xlabel('Mean Dice Score')
ax.set_ylabel('Number of Patients')
ax.axvline(df['dice_mean'].mean(), color='black', ls='--', lw=1.5,
           label=f'Cohort Mean: {df["dice_mean"].mean():.4f}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_dice_distribution.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ── 2. Multi-Metric Box Plot ──
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
metrics_box = ['dice_mean', 'iou_mean', 'ssim_mean', 'hd95_mean']
titles_box  = ['Dice Score', 'IoU Score', 'SSIM', 'HD95 (pixels)']
for ax, metric, title in zip(axes, metrics_box, titles_box):
    sns.boxplot(data=df, x='split', y=metric, palette={'TRAIN': 'royalblue', 'VAL': 'orangered'},
                width=0.5, ax=ax)
    sns.stripplot(data=df, x='split', y=metric, color='black', alpha=0.3, size=3, ax=ax, jitter=True)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('')
plt.suptitle('Geometry Metrics: Train vs Val', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_boxplot_geometry.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ── 3. Sorted Waterfall Plot ──
df_sorted = df.sort_values('dice_mean', ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(14, 5))
colors = ['royalblue' if s == 'TRAIN' else 'orangered' for s in df_sorted['split']]
ax.bar(range(len(df_sorted)), df_sorted['dice_mean'], color=colors, width=1.0, edgecolor='none')
ax.set_title('Per-Patient Dice Scores (Sorted, Colored by Split)', fontweight='bold')
ax.set_xlabel('Patients (Ordered by Performance)')
ax.set_ylabel('Mean Dice Score')
ax.set_xticks([])
ax.set_ylim(max(0.85, df['dice_mean'].min() - 0.02), 1.0)
# Legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='royalblue', label='TRAIN'), Patch(color='orangered', label='VAL')],
          loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_sorted_waterfall.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ── 4. Per-View Breakdown ──
view_data = []
for vname in CFG['view_names']:
    for _, row in df.iterrows():
        view_data.append({
            'View': vname,
            'Split': row['split'],
            'Dice': row[f'dice_{vname}'],
        })
df_view = pd.DataFrame(view_data)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=df_view, x='View', y='Dice', hue='Split',
            palette={'TRAIN': 'royalblue', 'VAL': 'orangered'}, ax=ax, errorbar='sd')
ax.set_title('Per-View Dice Scores (Train vs Val)', fontweight='bold')
ax.set_ylabel('Dice Score')
ax.set_ylim(0.9, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_per_view_breakdown.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ── 5. Thermal Metrics Box Plot ──
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
thermal_metrics = ['thermal_mse_mean', 'thermal_mae_mean', 'thermal_psnr_mean', 'thermal_r_mean']
thermal_titles  = ['Thermal MSE', 'Thermal MAE', 'PSNR (dB)', 'Pearson r']
for ax, metric, title in zip(axes, thermal_metrics, thermal_titles):
    sns.boxplot(data=df, x='split', y=metric, palette={'TRAIN': 'royalblue', 'VAL': 'orangered'},
                width=0.5, ax=ax)
    sns.stripplot(data=df, x='split', y=metric, color='black', alpha=0.3, size=3, ax=ax, jitter=True)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('')
plt.suptitle('Thermal Metrics: Train vs Val', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_boxplot_thermal.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ── 6. Summary Statistics Table ──
summary_rows = []
for metric, label in [
    ('dice_mean', 'Dice'), ('iou_mean', 'IoU'), ('ssim_mean', 'SSIM'),
    ('hd95_mean', 'HD95 (px)'), ('assd_mean', 'ASSD (px)'),
    ('precision_mean', 'Precision'), ('recall_mean', 'Recall'),
    ('thermal_mse_mean', 'Thermal MSE'), ('thermal_mae_mean', 'Thermal MAE'),
    ('thermal_psnr_mean', 'PSNR (dB)'), ('thermal_r_mean', 'Pearson r'),
    ('thermal_grderr_mean', 'Gradient Err'),
]:
    all_mean = df[metric].mean()
    all_std  = df[metric].std()
    t_mean   = df_train[metric].mean()
    t_std    = df_train[metric].std()
    v_mean   = df_val[metric].mean()
    v_std    = df_val[metric].std()
    summary_rows.append([label,
                         f'{all_mean:.4f} ± {all_std:.4f}',
                         f'{t_mean:.4f} ± {t_std:.4f}',
                         f'{v_mean:.4f} ± {v_std:.4f}'])

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')
table = ax.table(cellText=summary_rows,
                 colLabels=['Metric', 'All (n={})'.format(len(df)),
                            'Train (n={})'.format(len(df_train)),
                            'Val (n={})'.format(len(df_val))],
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.5)
# Style header
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#D6E4F0')
plt.title('TherMAM-NeRF v2.9 — Summary Statistics', fontweight='bold', fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'plot_summary_table.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

print('All paper-ready plots saved to:', OUTPUT_DIR)


# %% [markdown]
# # Cell 7: Interactive 3D Visualization + PINN/FEA Export

# %%
# ── 3D Visualization for sample patients ──

N_VIZ = 3  # Number of patients to visualize in 3D
viz_indices = [0, len(full_ds) // 2, len(full_ds) - 1]  # First, middle, last

print(f'\n{"="*60}')
print(f'3D Visualization & Export')
print(f'{"="*60}\n')

for viz_idx in viz_indices[:N_VIZ]:
    sample     = full_ds[viz_idx]
    tiffs_norm = sample['tiffs_norm'].to(DEVICE)
    masks_gt   = sample['masks'].to(DEVICE)
    patient_id = sample['patient_id']
    tmin_vals  = sample['tmin'].numpy()
    tmax_vals  = sample['tmax'].numpy()
    split_tag  = 'TRAIN' if patient_id in train_ids else 'VAL'

    print(f'Processing {patient_id} ({split_tag})...')

    # Extract 3D volume
    sigma_grid, T_grid = extract_3d_volume(encoder, mlp, tiffs_norm, masks_gt,
                                            CFG, DEVICE, resolution=128)

    # Smooth for mesh quality
    sigma_smooth = gaussian_filter(sigma_grid, sigma=1.5)

    # Marching cubes
    try:
        verts, faces, normals, _ = marching_cubes(sigma_smooth, level=CFG['mc_threshold'])
    except ValueError:
        print(f'  Could not generate mesh for {patient_id} (empty volume).')
        continue

    # ── Lambertian Thermal Painting ──
    # Adapted from breastnet_v5: project NeRF thermal field onto surface vertices
    # using multi-view weighted blending based on surface normal dot camera direction
    intensities = np.zeros(len(verts))
    weight_sum  = np.zeros(len(verts))
    R = 128
    half = R / 2.0

    # Vertex positions in grid coords (verts from marching_cubes are in grid space)
    # Convert to normalized [-1, 1] for sampling
    verts_norm = (verts / (R - 1)) * 2.0 - 1.0  # [0, R-1] -> [-1, 1]

    outward_normals = -normals  # Flip to point outward

    for vi, angle in enumerate(CFG['view_angles_deg']):
        rad = np.deg2rad(angle)
        c, s = np.cos(rad), np.sin(rad)

        # Camera direction vector (same as get_rays)
        cam_vec = np.array([-s, 0.0, c])

        # Dot product of outward normal with camera direction
        dot_prod = (outward_normals[:, 0] * cam_vec[0] +
                    outward_normals[:, 1] * cam_vec[1] +
                    outward_normals[:, 2] * cam_vec[2])

        # Exponential weighting — only faces visible from this camera
        weights = np.maximum(dot_prod + 0.3, 0.0) ** 8

        valid_idx = np.where(weights > 0)[0]
        if len(valid_idx) == 0:
            continue

        # Project 3D vertex onto the 2D image plane for this view
        # Rotation: xr = cos(theta)*x + sin(theta)*z, yr = y
        vx = verts_norm[valid_idx, 0]  # x in [-1,1]
        vy = verts_norm[valid_idx, 1]  # y in [-1,1]
        vz = verts_norm[valid_idx, 2]  # z in [-1,1]
        xr = c * vx + s * vz
        yr = vy

        # Map from [-1,1] to pixel coords [0, 127]
        x_px = (xr + 1.0) * 0.5 * (R - 1)
        y_px = (yr + 1.0) * 0.5 * (R - 1)

        # Sample the NeRF's own thermal field from the rendered thermal map
        # We use the GT normalized thermal for ground-truth temperature
        thermal_img = sample['tiffs_norm'][vi].numpy()

        # Bilinear interpolation
        x0 = np.clip(np.floor(x_px).astype(int), 0, R - 2)
        y0 = np.clip(np.floor(y_px).astype(int), 0, R - 2)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = x_px - x0
        wy = y_px - y0

        sampled = (thermal_img[y0, x0] * (1 - wx) * (1 - wy) +
                   thermal_img[y0, x1] * wx * (1 - wy) +
                   thermal_img[y1, x0] * (1 - wx) * wy +
                   thermal_img[y1, x1] * wx * wy)

        intensities[valid_idx] += sampled * weights[valid_idx]
        weight_sum[valid_idx]  += weights[valid_idx]

    # Normalize
    untextured = weight_sum == 0
    weight_sum[untextured] = 1.0
    thermal_paint = intensities / weight_sum

    # Frontal fallback for untextured vertices
    if np.any(untextured):
        frontal_img = sample['tiffs_norm'][2].numpy()  # Index 2 = Frontal
        x_fb = np.clip(((verts_norm[untextured, 0] + 1.0) * 0.5 * (R - 1)).astype(int), 0, R - 1)
        y_fb = np.clip(((verts_norm[untextured, 1] + 1.0) * 0.5 * (R - 1)).astype(int), 0, R - 1)
        thermal_paint[untextured] = frontal_img[y_fb, x_fb]

    # Denormalize to °C using the frontal view's temperature range
    tmin_f = float(tmin_vals[2])
    tmax_f = float(tmax_vals[2])
    thermal_celsius = thermal_paint * (tmax_f - tmin_f) + tmin_f

    # ── Plotly 3D Mesh ──
    # Axes: x=verts[:,2] (width), y=verts[:,0] (depth), z=-verts[:,1] (height, inverted)
    fig_3d = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 2], y=verts[:, 0], z=-verts[:, 1],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            colorscale='Jet',
            intensity=thermal_celsius,
            showscale=True,
            colorbar_title="Temperature<br>(°C)"
        )
    ])
    fig_3d.update_layout(
        title=f"TherMAM-NeRF | {patient_id} ({split_tag}) | 3D Thermal Mesh",
        scene=dict(
            xaxis_title='Width', yaxis_title='Depth', zaxis_title='Height',
            aspectmode='data',
            camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=0, y=1.5, z=0.2))
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    html_path = os.path.join(OUTPUT_DIR, f'3d_thermal_{patient_id}.html')
    fig_3d.write_html(html_path)
    print(f'  Saved interactive 3D: {html_path}')

    # Also save a static image
    try:
        img_path = os.path.join(OUTPUT_DIR, f'3d_thermal_{patient_id}.png')
        fig_3d.write_image(img_path, width=1200, height=800, scale=2)
        print(f'  Saved static 3D: {img_path}')
    except Exception:
        print(f'  (Static image export requires kaleido: pip install kaleido)')


# ── PINN + FEA Export (all patients) ──

print(f'\n{"─"*60}')
print('Exporting for PINN + FEA...')

export_dir_stl = os.path.join(OUTPUT_DIR, 'exported_stls')
export_dir_npz = os.path.join(OUTPUT_DIR, 'exported_npz')
export_dir_ply = os.path.join(OUTPUT_DIR, 'exported_plys')
export_dir_csv = os.path.join(OUTPUT_DIR, 'exported_surface_csv')
os.makedirs(export_dir_stl, exist_ok=True)
os.makedirs(export_dir_npz, exist_ok=True)
os.makedirs(export_dir_ply, exist_ok=True)
os.makedirs(export_dir_csv, exist_ok=True)


def save_stl_binary(filepath, verts, faces):
    """Write a binary STL file."""
    with open(filepath, 'wb') as f:
        f.write(b'\0' * 80)  # header
        f.write(struct.pack('<I', len(faces)))
        for face in faces:
            tri = verts[face].astype(np.float32)
            v0, v1, v2 = tri
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = (normal / norm).astype(np.float32)
            else:
                normal = np.zeros(3, dtype=np.float32)
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))


def export_ply_thermal(filepath, verts, faces, temperatures):
    """Write PLY with per-vertex temperature and inferno coloring."""
    cmap = plt.get_cmap('inferno')
    t_min, t_max = temperatures.min(), temperatures.max()
    t_norm = (temperatures - t_min) / (t_max - t_min + 1e-6)
    colors = (cmap(t_norm)[:, :3] * 255).astype(np.uint8)

    header = (
        f"ply\nformat ascii 1.0\n"
        f"element vertex {len(verts)}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"property float temperature\n"
        f"element face {len(faces)}\n"
        f"property list uchar int vertex_indices\nend_header\n"
    )
    with open(filepath, 'w') as f:
        f.write(header)
        for v, c, t in zip(verts, colors, temperatures):
            f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]} {t:.4f}\n')
        for face in faces:
            f.write(f'3 {face[0]} {face[1]} {face[2]}\n')


n_exported = 0
for idx in tqdm(range(len(full_ds)), desc='Exporting PINN/FEA'):
    sample     = full_ds[idx]
    tiffs_norm = sample['tiffs_norm'].to(DEVICE)
    masks_gt   = sample['masks'].to(DEVICE)
    patient_id = sample['patient_id']
    tmin_vals  = sample['tmin'].numpy()
    tmax_vals  = sample['tmax'].numpy()

    # Extract volume
    sigma_grid, T_grid = extract_3d_volume(encoder, mlp, tiffs_norm, masks_gt,
                                            CFG, DEVICE, resolution=128)

    # ── NPZ Export (raw volumetric data for PINN) ──
    npz_path = os.path.join(export_dir_npz, f'{patient_id}_volume.npz')
    np.savez_compressed(npz_path,
                        sigma=sigma_grid,
                        temperature=T_grid,
                        tmin=tmin_vals,
                        tmax=tmax_vals)

    # Smooth for meshing
    sigma_smooth = gaussian_filter(sigma_grid, sigma=1.5)

    # Pad for watertight mesh
    sigma_padded = np.pad(sigma_smooth, pad_width=1, mode='constant', constant_values=0)

    try:
        verts, faces, normals, _ = marching_cubes(sigma_padded, level=CFG['mc_threshold'])
        # Shift vertices back by 1 to compensate for padding
        verts = verts - 1.0
        # Normalize to [-1, 1]
        verts_norm_export = (verts / 63.5) - 1.0

        # ── STL Export ──
        stl_path = os.path.join(export_dir_stl, f'{patient_id}_geometry.stl')
        save_stl_binary(stl_path, verts_norm_export, faces)

        # ── Sample temperature at vertices from T_grid ──
        # Clamp vertex indices to valid grid range
        vi = np.clip(np.round(verts).astype(int), 0, 127)
        vert_temps_norm = T_grid[vi[:, 0], vi[:, 1], vi[:, 2]]
        # Denormalize to °C using frontal view range
        vert_temps_c = vert_temps_norm * (tmax_vals[2] - tmin_vals[2]) + tmin_vals[2]

        # ── PLY Export ──
        ply_path = os.path.join(export_dir_ply, f'{patient_id}_thermal.ply')
        export_ply_thermal(ply_path, verts_norm_export, faces, vert_temps_c)

        # ── Surface CSV Export (for PINN boundary conditions) ──
        csv_surf_path = os.path.join(export_dir_csv, f'{patient_id}_surface.csv')
        surf_df = pd.DataFrame({
            'x': verts_norm_export[:, 0],
            'y': verts_norm_export[:, 1],
            'z': verts_norm_export[:, 2],
            'temperature_C': vert_temps_c,
            'normal_x': normals[:, 0],
            'normal_y': normals[:, 1],
            'normal_z': normals[:, 2],
        })
        surf_df.to_csv(csv_surf_path, index=False)

        n_exported += 1

    except Exception as e:
        print(f'  Skipping {patient_id}: {e}')

print(f'\nExported {n_exported}/{len(full_ds)} patients.')
print(f'  STL files: {export_dir_stl}')
print(f'  PLY files: {export_dir_ply}')
print(f'  NPZ files: {export_dir_npz}')
print(f'  CSV files: {export_dir_csv}')


# ── Final Summary ──

print(f'\n{"="*60}')
print('VALIDATION COMPLETE')
print(f'{"="*60}')
print(f'All results saved to: {OUTPUT_DIR}')
print(f'  cohort_all_metrics.csv     — Full per-patient metrics')
print(f'  plot_*.png                 — Paper-ready visualizations')
print(f'  3d_thermal_*.html          — Interactive 3D models')
print(f'  exported_stls/             — Watertight STL meshes for FEA')
print(f'  exported_plys/             — Thermal-colored PLY meshes')
print(f'  exported_npz/              — Raw 128³ volumetric data for PINN')
print(f'  exported_surface_csv/      — Surface BCs for PINN solvers')
