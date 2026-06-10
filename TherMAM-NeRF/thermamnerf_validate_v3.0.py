#!/usr/bin/env python3
"""
TherMAM-NeRF v3.0 — Out-of-Fold Validation & Ensemble Export
=============================================================
Run on SSH:  python thermamnerf_validate_v3.0.py

This script requires all 5 folds to be trained. It performs:
  1. Out-of-Fold (OOF) Geometry & Thermal Metrics evaluation
  2. Ensemble 3D volume rendering for high-fidelity PINN/FEA export
"""

import os
import math
import json
import struct
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
from scipy.stats import pearsonr
from skimage.measure import marching_cubes
from skimage.metrics import structural_similarity as ssim_metric
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
TIFF_DIR   = str(REPO_ROOT / 'data' / 'organized_by_patient')
UNET_DIR   = str(REPO_ROOT / 'data' / 'organized_by_patient_unet')
OUTPUT_DIR = str(SCRIPT_DIR / 'validation_results_v3.0')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyperparameters (must match v3.0 training) ──
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

# ── Utilities ──
def load_tiff_celsius(path: str, target_size: int) -> np.ndarray:
    arr = tifffile.imread(path).astype(np.float32)
    if arr.ndim == 3: arr = arr[..., 0]
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
    return (arr - tmin) / (tmax - tmin + 1e-6), tmin, tmax

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
        if key in pd_: pd_[key]['masks'][vk] = mp
    patients = []
    for (pid, lab), d in pd_.items():
        if len(d['tiffs']) == 5 and len(d['masks']) == 5:
            patients.append({'id': pid, 'tiffs': d['tiffs'], 'masks': d['masks']})
    patients.sort(key=lambda p: p['id'])
    return patients

patients = discover_patients_split(TIFF_DIR, UNET_DIR)
print(f'Found {len(patients)} complete patients.')

class BreastThermDataset(torch.utils.data.Dataset):
    def __init__(self, patient_list: list, cfg: dict):
        self.patients = patient_list
        self.S = cfg['img_size']
        self.view_names = cfg['view_names']
    def __len__(self): return len(self.patients)
    def __getitem__(self, idx):
        p = self.patients[idx]
        tiffs_norm, tiffs_abs, masks = [], [], []
        tmins, tmaxs = [], []
        for v in self.view_names:
            raw = load_tiff_celsius(str(p['tiffs'][v]), self.S)
            normd, tmin, tmax = normalize_thermal(raw)
            mask = load_mask(str(p['masks'][v]), self.S)
            tiffs_norm.append(normd); tiffs_abs.append(raw); masks.append(mask)
            tmins.append(tmin); tmaxs.append(tmax)
        return {
            'patient_id': p['id'],
            'tiffs_norm': torch.tensor(np.stack(tiffs_norm), dtype=torch.float32),
            'tiffs_abs' : torch.tensor(np.stack(tiffs_abs), dtype=torch.float32),
            'masks'     : torch.tensor(np.stack(masks), dtype=torch.float32),
            'tmin'      : torch.tensor(tmins, dtype=torch.float32),
            'tmax'      : torch.tensor(tmaxs, dtype=torch.float32),
        }

full_ds = BreastThermDataset(patients, CFG)

# ── Models ──
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class SiameseEncoder(nn.Module):
    def __init__(self, in_ch=2, base_ch=16, out_ch=16):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(base_ch * 2 + base_ch, out_ch)
    def forward(self, thermal, mask):
        x = torch.cat([thermal.unsqueeze(1), mask.unsqueeze(1)], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d1 = self.dec1(torch.cat([self.up1(e3), e2], dim=1))
        out = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return out

class ThermamNeRFMLP(nn.Module):
    def __init__(self, pos_ch, feat_ch, hidden_ch=128, num_layers=4):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(pos_ch + feat_ch, hidden_ch)])
        for _ in range(num_layers - 2): self.net.append(nn.Linear(hidden_ch, hidden_ch))
        self.sigma_head = nn.Linear(hidden_ch, 1)
        self.temp_head  = nn.Linear(hidden_ch, 1)
    def forward(self, pe, feat):
        chunk_size = 131072
        if pe.shape[1] <= chunk_size:
            x = torch.cat([pe, feat], dim=-1)
            for layer in self.net:
                x = F.relu(layer(x))
            sigma = F.softplus(self.sigma_head(x))
            temp  = torch.sigmoid(self.temp_head(x))
            return sigma, temp
            
        sigma_list, temp_list = [], []
        for i in range(0, pe.shape[1], chunk_size):
            pe_c = pe[:, i:i+chunk_size]
            feat_c = feat[:, i:i+chunk_size]
            x = torch.cat([pe_c, feat_c], dim=-1)
            for layer in self.net:
                x = F.relu(layer(x))
            sigma_list.append(F.softplus(self.sigma_head(x)))
            temp_list.append(torch.sigmoid(self.temp_head(x)))
        return torch.cat(sigma_list, dim=1), torch.cat(temp_list, dim=1)

def positional_encoding(x, L=6, alpha=None):
    if alpha is None: alpha = float(L)
    pe = [x]
    for i in range(L):
        freq = 2.0 ** i
        w = 1.0
        if i > alpha: w = 0.0
        elif i > alpha - 1: w = alpha - i
        pe.append(torch.sin(x * math.pi * freq) * w)
        pe.append(torch.cos(x * math.pi * freq) * w)
    return torch.cat(pe, dim=-1)

def project_and_sample(pts, feat_maps, view_angles_rad):
    B, N, _ = pts.shape
    V = feat_maps.shape[1]
    sampled_feats = []
    for v in range(V):
        theta = view_angles_rad[v]
        c, s = torch.cos(theta), torch.sin(theta)
        xr = c * pts[..., 0] - s * pts[..., 2]
        yr = pts[..., 1]
        grid = torch.stack([xr, yr], dim=-1).unsqueeze(1)
        sf = F.grid_sample(feat_maps[:, v], grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled_feats.append(sf.squeeze(2).transpose(1, 2))
    return torch.cat(sampled_feats, dim=-1)

def get_rays(H, W, angle_deg, device):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    cam_dir = torch.tensor([-s, 0.0, c], device=device)
    up      = torch.tensor([0.0, 1.0, 0.0], device=device)
    right   = torch.cross(cam_dir, up)
    gy, gx = torch.meshgrid(torch.linspace(1, -1, H, device=device),
                            torch.linspace(-1, 1, W, device=device), indexing='ij')
    origins = (gx.flatten().unsqueeze(1) * right.unsqueeze(0) +
               gy.flatten().unsqueeze(1) * up.unsqueeze(0))
    dirs = cam_dir.unsqueeze(0).expand(H * W, -1)
    return origins, dirs

def volume_render(sigma, T_field, deltas, density_scale):
    sigma = sigma * density_scale
    alpha = 1.0 - torch.exp(-sigma * deltas.unsqueeze(0))
    T_trans = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1, device=alpha.device),
                                       1.0 - alpha[:, :-1] + 1e-10], dim=1), dim=1)
    weights = alpha * T_trans
    rm = weights.sum(dim=1)
    rt = (weights.detach() * T_field).sum(dim=1)
    om = (rm.detach() > 0.05).float()
    rt = (rt / (rm.detach() + 1e-6)) * om
    return rm, rt

@torch.no_grad()
def render_view(encoder, mlp, tiffs_norm, masks, view_idx, cfg, device, alpha=6.0):
    H = W = cfg['img_size']
    angle = cfg['view_angles_deg'][view_idx]
    view_angles_rad = torch.tensor([math.radians(a) for a in cfg['view_angles_deg']], device=device)
    feat_maps = torch.stack([encoder(tiffs_norm[v:v+1], masks[v:v+1]) for v in range(cfg['n_views'])], dim=1)
    rays_o, rays_d = get_rays(H, W, angle, device)
    
    n_s = cfg['n_samples']
    near, far = cfg['near'], cfg['far']
    t_vals = torch.linspace(near, far, n_s, device=device)
    deltas = torch.cat([t_vals[1:] - t_vals[:-1], torch.tensor([1e-3], device=device)])
    
    rm_list, rt_list = [], []
    chunk = 2048
    for i in range(0, H * W, chunk):
        ro = rays_o[i:i+chunk]
        rd = rays_d[i:i+chunk]
        pts = ro.unsqueeze(1) + t_vals.view(1, -1, 1) * rd.unsqueeze(1)
        pf = pts.reshape(1, -1, 3)
        pe = positional_encoding(pf, L=cfg['pos_enc_L'], alpha=alpha)
        feat = project_and_sample(pf, feat_maps, view_angles_rad)
        sigma, T_pred = mlp(pe, feat)
        
        if angle <= -80: sigma = sigma.masked_fill(pf[..., 0:1] > 0.1, 0.0)
        elif angle >= 80: sigma = sigma.masked_fill(pf[..., 0:1] < -0.1, 0.0)
            
        sigma = sigma.squeeze(-1).reshape(ro.shape[0], n_s)
        T_pred = T_pred.squeeze(-1).reshape(ro.shape[0], n_s)
        rm, rt = volume_render(sigma, T_pred, deltas, cfg['density_scale'])
        rm_list.append(rm)
        rt_list.append(rt)
        
    return torch.cat(rm_list).reshape(H, W), torch.cat(rt_list).reshape(H, W)


# ── Load Ensembles ──
encoders = []
mlps = []
patient_to_test_fold = {}

for fold_idx in range(5):
    f_dir = SCRIPT_DIR / f'thermamnerf_outputs3.0_fold{fold_idx}'
    ckpt_path = f_dir / 'thermamnerf_best.pth'
    ids_path = f_dir / f'test_ids_fold{fold_idx}.json'
    
    if not ckpt_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"Missing fold {fold_idx} outputs. Ensure all 5 folds are trained.")
    
    pos_ch = 3 * (2 * CFG['pos_enc_L'] + 1)
    enc = SiameseEncoder(out_ch=CFG['feat_channels']).to(DEVICE)
    m   = ThermamNeRFMLP(pos_ch=pos_ch, feat_ch=CFG['feat_channels'] * CFG['n_views'], hidden_ch=CFG['mlp_hidden'], num_layers=CFG['mlp_layers']).to(DEVICE)
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    enc.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['encoder'].items()})
    m.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['mlp'].items()})
    enc.eval()
    m.eval()
    encoders.append(enc)
    mlps.append(m)
    
    with open(ids_path, 'r') as f:
        test_ids = json.load(f)
        for pid in test_ids:
            patient_to_test_fold[pid] = fold_idx

print("Loaded 5-Fold Ensemble.")


# ── Metrics ──
def compute_hausdorff_95(pred_bin, gt_bin):
    pred_boundary = np.zeros_like(pred_bin, dtype=bool)
    gt_boundary   = np.zeros_like(gt_bin, dtype=bool)
    pred_boundary[1:, :] |= (pred_bin[1:, :] != pred_bin[:-1, :])
    pred_boundary[:, 1:] |= (pred_bin[:, 1:] != pred_bin[:, :-1])
    gt_boundary[1:, :]   |= (gt_bin[1:, :] != gt_bin[:-1, :])
    gt_boundary[:, 1:]   |= (gt_bin[:, 1:] != gt_bin[:, :-1])
    pred_pts = np.argwhere(pred_boundary)
    gt_pts   = np.argwhere(gt_boundary)
    if len(pred_pts) == 0 or len(gt_pts) == 0: return 0.0
    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)
    return float(np.percentile(np.concatenate([d_pred_to_gt, d_gt_to_pred]), 95))

def compute_gradient_error(pred_temp, gt_temp, fg_mask):
    pred_mag = np.sqrt(sobel(pred_temp, axis=0)**2 + sobel(pred_temp, axis=1)**2)
    gt_mag   = np.sqrt(sobel(gt_temp, axis=0)**2 + sobel(gt_temp, axis=1)**2)
    if fg_mask.sum() == 0: return 0.0
    return float(np.abs(pred_mag[fg_mask] - gt_mag[fg_mask]).mean())


# ============================================================================
# PHASE 1: Out-of-Fold (OOF) Metrics (Unseen Performance)
# ============================================================================
results = []
print(f'\n{"="*60}\nPhase 1: Out-of-Fold Evaluation (Generalization Proof)\n{"="*60}')

for idx in tqdm(range(len(full_ds)), desc='OOF Eval'):
    sample     = full_ds[idx]
    patient_id = sample['patient_id']
    tiffs_norm = sample['tiffs_norm'].to(DEVICE)
    masks_gt   = sample['masks'].to(DEVICE)
    
    # Use ONLY the model where this patient was held out
    oof_fold = patient_to_test_fold[patient_id]
    enc, m = encoders[oof_fold], mlps[oof_fold]
    
    patient_metrics = {'patient_id': patient_id, 'oof_fold': oof_fold}
    view_dice, view_iou = [], []

    for v, vname in enumerate(CFG['view_names']):
        rm, rt = render_view(enc, m, tiffs_norm, masks_gt, v, CFG, DEVICE, alpha=float(CFG['pos_enc_L']))
        pred_m, pred_t = rm.cpu().numpy(), rt.cpu().numpy()
        gt_m = masks_gt[v].cpu().numpy()
        gt_t = sample['tiffs_norm'][v].numpy()
        pred_bin = (pred_m > 0.5).astype(np.float32)
        gt_bin   = gt_m.astype(np.float32)

        tp = (pred_bin * gt_bin).sum()
        fp = (pred_bin * (1 - gt_bin)).sum()
        fn = ((1 - pred_bin) * gt_bin).sum()
        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
        iou  = tp / (tp + fp + fn + 1e-6)
        
        patient_metrics[f'dice_{vname}'] = dice
        patient_metrics[f'iou_{vname}']  = iou
        patient_metrics[f'hd95_{vname}'] = compute_hausdorff_95(pred_bin, gt_bin)
        patient_metrics[f'ssim_{vname}'] = ssim_metric(gt_m, pred_m, data_range=1.0)
        view_dice.append(dice); view_iou.append(iou)

        fg = gt_bin > 0.5
        if fg.sum() > 10:
            pred_fg, gt_fg = pred_t[fg], gt_t[fg]
            mse = float(np.mean((pred_fg - gt_fg) ** 2))
            mae = float(np.mean(np.abs(pred_fg - gt_fg)))
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            r_val, _ = pearsonr(pred_fg.flatten(), gt_fg.flatten())
            grad_err = compute_gradient_error(pred_t, gt_t, fg)
        else:
            mse, mae, psnr, r_val, grad_err = 0, 0, 0, 0, 0

        patient_metrics[f'thermal_mse_{vname}']  = mse
        patient_metrics[f'thermal_mae_{vname}']  = mae
        patient_metrics[f'thermal_psnr_{vname}'] = psnr
        patient_metrics[f'thermal_r_{vname}']    = r_val
        patient_metrics[f'thermal_grderr_{vname}'] = grad_err

    patient_metrics['dice_mean'] = np.mean(view_dice)
    patient_metrics['iou_mean']  = np.mean(view_iou)
    for m_name in ['hd95', 'ssim', 'thermal_mse', 'thermal_mae', 'thermal_psnr', 'thermal_r', 'thermal_grderr']:
        patient_metrics[f'{m_name}_mean'] = np.mean([patient_metrics[f'{m_name}_{v}'] for v in CFG['view_names']])
    results.append(patient_metrics)

df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, 'cohort_oof_metrics.csv'), index=False)

# Summary
print(f'\n{"="*60}')
print(f'OOF COHORT SUMMARY ({len(df)} patients)')
print(f'{"="*60}')
for metric in ['dice_mean', 'iou_mean', 'hd95_mean', 'ssim_mean', 'thermal_mse_mean', 'thermal_mae_mean', 'thermal_psnr_mean']:
    print(f"{metric:>20}: {df[metric].mean():.4f} ± {df[metric].std():.4f}")


# ============================================================================
# PHASE 2: Ensemble 3D Export (Highest Fidelity)
# ============================================================================
print(f'\n{"="*60}\nPhase 2: Ensemble 3D Extraction (FEA/PINN Exports)\n{"="*60}')
export_dir_stl = os.path.join(OUTPUT_DIR, 'exported_stls')
export_dir_ply = os.path.join(OUTPUT_DIR, 'exported_plys')
export_dir_npz = os.path.join(OUTPUT_DIR, 'exported_npz')
export_dir_csv = os.path.join(OUTPUT_DIR, 'exported_surface_csv')
for d in [export_dir_stl, export_dir_ply, export_dir_npz, export_dir_csv]: os.makedirs(d, exist_ok=True)

def save_stl_binary(filepath, verts, faces):
    with open(filepath, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(faces)))
        for face in faces:
            tri = verts[face].astype(np.float32)
            v0, v1, v2 = tri
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            normal = (normal / norm).astype(np.float32) if norm > 0 else np.zeros(3, dtype=np.float32)
            f.write(struct.pack('<3f', *normal)); f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1)); f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))

def export_ply_thermal(filepath, verts, faces, temperatures):
    cmap = plt.get_cmap('inferno')
    t_min, t_max = temperatures.min(), temperatures.max()
    t_norm = (temperatures - t_min) / (t_max - t_min + 1e-6)
    colors = (cmap(t_norm)[:, :3] * 255).astype(np.uint8)
    with open(filepath, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(verts)}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                f"property float temperature\nelement face {len(faces)}\n"
                f"property list uchar int vertex_indices\nend_header\n")
        for v, c, t in zip(verts, colors, temperatures):
            f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]} {t:.4f}\n')
        for face in faces: f.write(f'3 {face[0]} {face[1]} {face[2]}\n')

for idx in tqdm(range(len(full_ds)), desc='Ensemble Export'):
    sample = full_ds[idx]
    patient_id = sample['patient_id']
    tiffs_norm = sample['tiffs_norm'].to(DEVICE)
    masks_gt   = sample['masks'].to(DEVICE)
    tmin_vals, tmax_vals = sample['tmin'].numpy(), sample['tmax'].numpy()
    
    R = CFG['mc_resolution']
    linspace = torch.linspace(-1, 1, R, device=DEVICE)
    zz, yy, xx = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)
    view_angles_rad = torch.tensor([math.radians(a) for a in CFG['view_angles_deg']], device=DEVICE)
    
    all_sigma, all_T = [], []
    with torch.no_grad():
        for i in range(5):
            enc, m = encoders[i], mlps[i]
            feat_maps = torch.stack([enc(tiffs_norm[v:v+1], masks_gt[v:v+1]) for v in range(5)], dim=1)
            sigma_all, T_all = [], []
            for j in range(0, pts.shape[1], 8192):
                p = pts[:, j:j+8192]
                pe = positional_encoding(p, L=CFG['pos_enc_L'], alpha=float(CFG['pos_enc_L']))
                feat = project_and_sample(p, feat_maps, view_angles_rad)
                sg, tp = m(pe, feat)
                sigma_all.append(sg.squeeze().cpu()); T_all.append(tp.squeeze().cpu())
            all_sigma.append(torch.cat(sigma_all).reshape(R, R, R).numpy())
            all_T.append(torch.cat(T_all).reshape(R, R, R).numpy())
            
    # Ensemble Average
    sigma_grid = np.mean(all_sigma, axis=0)
    T_grid = np.mean(all_T, axis=0)
    
    np.savez_compressed(os.path.join(export_dir_npz, f'{patient_id}_volume.npz'),
                        sigma=sigma_grid, temperature=T_grid, tmin=tmin_vals, tmax=tmax_vals)

    sigma_smooth = gaussian_filter(sigma_grid, sigma=1.5)
    sigma_padded = np.pad(sigma_smooth, pad_width=1, mode='constant', constant_values=0)
    
    try:
        verts, faces, normals, _ = marching_cubes(sigma_padded, level=CFG['mc_threshold'])
        verts = verts - 1.0
        verts_norm_export = (verts / 63.5) - 1.0
        
        save_stl_binary(os.path.join(export_dir_stl, f'{patient_id}_geometry.stl'), verts_norm_export, faces)
        
        vi = np.clip(np.round(verts).astype(int), 0, R-1)
        vert_temps_norm = T_grid[vi[:, 0], vi[:, 1], vi[:, 2]]
        vert_temps_c = vert_temps_norm * (tmax_vals[2] - tmin_vals[2]) + tmin_vals[2]
        
        export_ply_thermal(os.path.join(export_dir_ply, f'{patient_id}_thermal.ply'), verts_norm_export, faces, vert_temps_c)
        
        surf_df = pd.DataFrame({
            'x': verts_norm_export[:, 0], 'y': verts_norm_export[:, 1], 'z': verts_norm_export[:, 2],
            'temperature_C': vert_temps_c,
            'normal_x': normals[:, 0], 'normal_y': normals[:, 1], 'normal_z': normals[:, 2],
        })
        surf_df.to_csv(os.path.join(export_dir_csv, f'{patient_id}_surface.csv'), index=False)
    except Exception as e:
        print(f'  Skipping mesh for {patient_id}: {e}')

print("\nDONE! All exports and metrics are saved to `validation_results_v3.0`")
