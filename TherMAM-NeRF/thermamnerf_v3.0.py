#!/usr/bin/env python3
"""
DDP usage: `torchrun --nproc_per_node=2 thermamnerf_v3.0.py --fold 0`
"""

import os
import copy
import json
import math
import glob
import argparse
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from skimage.measure import marching_cubes
import matplotlib
# Use non-interactive backend for headless environments (tmux/SSH)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import plotly.graph_objects as go
from pathlib import Path
from tqdm import tqdm
import warnings
from sklearn.model_selection import KFold, train_test_split
warnings.filterwarnings('ignore')

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    return rank, world_size, local_rank, device


RANK, WORLD_SIZE, LOCAL_RANK, DEVICE = setup_distributed()
IS_MAIN = (RANK == 0)

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0, help='Which fold to train (0-4)')
args = parser.parse_args()
FOLD_IDX = args.fold

if IS_MAIN:
    print(f'Using device {DEVICE} | world_size={WORLD_SIZE} | rank={RANK}')
    print(f'TRAINING FOLD: {FOLD_IDX}')

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).resolve().parent
REPO_ROOT        = SCRIPT_DIR.parent
TIFF_DIR         = str(REPO_ROOT / 'data' / 'organized_by_patient')
UNET_DIR         = str(REPO_ROOT / 'data' / 'organized_by_patient_unet')
OUTPUT_DIR       = str(SCRIPT_DIR / f'thermamnerf_outputs3.0_fold{FOLD_IDX}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────
CFG = {
    'img_size'        : 128,
    'n_views'         : 5,
    'view_angles_deg' : [-90, -45, 0, 45, 90],
    'view_names'      : ['RL', 'RO', 'F', 'LO', 'LL'],

    # --- NETWORK CAPACITY ---
    'feat_channels'   : 32,      
    'pos_enc_L'       : 8,       
    'mlp_hidden'      : 256,     
    'mlp_layers'      : 4,
    # -----------------------------------------

    'n_samples'       : 256,
    'near'            : -1.0,
    'far'             : 1.0,
    'density_scale'   : 10.0,

    'freq_warmup_epochs': 50,

    'batch_size'      : 1,
    'n_epochs'        : 1000,
    'lr'              : 5e-4,
    'lambda_dice'     : 1.0,
    'lambda_bg'       : 2.0,
    'lambda_thermal'  : 20.0,
    'lambda_tv'       : 0.01,
    'lambda_entropy'  : 0.1,
    'eval_every'      : 5,
    'n_rays'          : 3072,

    'mc_threshold'    : 0.3,
    'mc_resolution'   : 128,

    'use_amp'         : True,
    'use_grad_checkpoint': True,
    
    # Early stopping configs
    'eval_every'      : 5,
    'patience'        : 80
}
if not torch.cuda.is_available():
    CFG['use_amp'] = False
if IS_MAIN:
    print('Config loaded.')

## -------------------- Data loading utilities --------------------------------
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

def normalize_thermal(arr: np.ndarray) -> np.ndarray:
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
        if len(parts) < 2:
            continue
        pid, lab, fn = parts[0], parts[1], parts[-1]
        vk = get_view_key(fn)
        if not vk: continue
        key = (pid, lab)
        if key not in pd_: pd_[key] = {'tiffs': {}, 'masks': {}}
        pd_[key]['tiffs'][vk] = tp

    for mp in ub.rglob('*.png'):
        parts = mp.relative_to(ub).parts
        if len(parts) < 2:
            continue
        pid, lab, fn = parts[0], parts[1], parts[-1]
        vk = get_view_key(fn)
        if not vk: continue
        key = (pid, lab)
        if key in pd_:
            pd_[key]['masks'][vk] = mp

    patients = []
    for (pid, lab), d in pd_.items():
        if len(d['tiffs']) == 5 and len(d['masks']) == 5:
            patients.append({
                'id'   : pid,
                'tiffs': d['tiffs'],
                'masks': d['masks'],
            })
    
    # Sort strictly for determinism before KFold
    patients.sort(key=lambda p: p['id'])
    if IS_MAIN:
        print(f'Found {len(patients)} complete patients.')
    return patients


patients = discover_patients_split(TIFF_DIR, UNET_DIR)
if len(patients) == 0:
    raise RuntimeError(
        f'No complete patients found under TIFF_DIR={TIFF_DIR!r} and UNET_DIR={UNET_DIR!r}. '
    )

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

# --- 5-Fold Cross Validation Split Logic ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
patient_indices = list(range(len(patients)))
folds = list(kf.split(patient_indices))

# For the current FOLD_IDX, we have train_pool and test
train_pool_idx, test_idx = folds[FOLD_IDX]

# From the train_pool, extract 20% for early stopping (VAL), 80% for actual TRAIN
train_idx, val_idx = train_test_split(train_pool_idx, test_size=0.20, random_state=42)

train_patients = [patients[i] for i in train_idx]
val_patients   = [patients[i] for i in val_idx]
test_patients  = [patients[i] for i in test_idx]

if IS_MAIN:
    print(f'--- Fold {FOLD_IDX} Data Split ---')
    print(f'Train: {len(train_patients)} | Val (Early Stop): {len(val_patients)} | Test (Held-Out): {len(test_patients)}')
    
    # Save test IDs so the validation script knows who was held out for this fold
    test_ids = [p['id'] for p in test_patients]
    with open(os.path.join(OUTPUT_DIR, f'test_ids_fold{FOLD_IDX}.json'), 'w') as f:
        json.dump(test_ids, f)

train_ds = BreastThermDataset(train_patients, CFG)
val_ds   = BreastThermDataset(val_patients,   CFG)

train_sampler = None
if WORLD_SIZE > 1:
    train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
    train_dl = DataLoader(train_ds, batch_size=CFG['batch_size'], sampler=train_sampler, num_workers=4, pin_memory=True)
else:
    train_dl = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

# Validation is usually single-GPU for simplicity, but can be on rank 0
val_dl = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False, num_workers=2)


## -------------------- Model Definitions --------------------------------------
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
    def forward(self, x):
        return self.net(x)

class SiameseEncoder(nn.Module):
    def __init__(self, in_ch=2, base_ch=16, out_ch=32):
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
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(pos_ch + feat_ch, hidden_ch))
        for _ in range(num_layers - 2):
            self.net.append(nn.Linear(hidden_ch, hidden_ch))
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

def get_alpha(epoch, warmup_epochs, max_L):
    if epoch >= warmup_epochs: return float(max_L)
    return (epoch / warmup_epochs) * max_L

def project_and_sample(pts, feat_maps, view_angles_rad):
    B, N, _ = pts.shape
    V = feat_maps.shape[1]
    C = feat_maps.shape[2]
    sampled_feats = []
    for v in range(V):
        theta = view_angles_rad[v]
        c, s = torch.cos(theta), torch.sin(theta)
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        xr = c * x - s * z
        yr = y
        grid = torch.stack([xr, yr], dim=-1).unsqueeze(1)
        fm = feat_maps[:, v]
        sf = F.grid_sample(fm, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled_feats.append(sf.squeeze(2).transpose(1, 2))
    return torch.cat(sampled_feats, dim=-1)

def tv_loss_3d(field):
    dx = (field[1:, :, :] - field[:-1, :, :]).abs().mean()
    dy = (field[:, 1:, :] - field[:, :-1, :]).abs().mean()
    dz = (field[:, :, 1:] - field[:, :, :-1]).abs().mean()
    return dx + dy + dz

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

def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1.0 - (2. * intersection + smooth) / (union + smooth)

def compute_loss(rendered_masks, rendered_temps,
                 gt_masks, gt_tiffs_norm, cfg, loss_tv=0.0) -> dict:
    loss_dice    = torch.tensor(0., device=gt_masks.device)
    loss_thermal = torch.tensor(0., device=gt_masks.device)
    loss_bg      = torch.tensor(0., device=gt_masks.device)
    loss_entropy = torch.tensor(0., device=gt_masks.device)
    for v in range(cfg['n_views']):
        pred_m  = rendered_masks[v]
        pred_t  = rendered_temps[v]
        gt_m    = gt_masks[:, v]
        gt_t    = gt_tiffs_norm[:, v]
        fg      = (gt_m > 0.5)
        bg      = (gt_m <= 0.5)
        loss_dice    = loss_dice + dice_loss(pred_m, gt_m)
        if fg.any():
            loss_thermal = loss_thermal + torch.nn.functional.mse_loss(pred_t[fg], gt_t[fg])
        if bg.any():
            loss_bg = loss_bg + pred_m[bg].pow(2).mean()
        
        # Ray opacity penalty
        loss_entropy = loss_entropy + (pred_m * (1.0 - pred_m)).mean()

    loss_dice    = loss_dice / cfg['n_views']
    loss_thermal = loss_thermal / cfg['n_views']
    loss_bg      = loss_bg / cfg['n_views']
    loss_entropy = loss_entropy / cfg['n_views']

    total = (cfg['lambda_dice']    * loss_dice +
             cfg['lambda_thermal'] * loss_thermal +
             cfg.get('lambda_tv', 0.0) * loss_tv +
             cfg.get('lambda_bg', 1.0) * loss_bg +
             cfg.get('lambda_entropy', 0.0) * loss_entropy)
    return {'total': total, 'dice': loss_dice, 'thermal': loss_thermal, 'tv': loss_tv, 'bg': loss_bg, 'entropy': loss_entropy}

def mlp_forward(model, pe, feat, use_grad_checkpoint=False):
    if use_grad_checkpoint and torch.is_grad_enabled():
        def _forward(pe_, feat_):
            return model(pe_, feat_)
        return checkpoint(_forward, pe, feat, use_reentrant=False)
    return model(pe, feat)

def run_one_batch(batch, encoder, mlp, cfg, alpha, device,
                  use_amp=False, use_grad_checkpoint=False):
    tiffs_norm = batch['tiffs_norm'].to(device)
    masks      = batch['masks'].to(device)
    B, V, H, W = masks.shape
    view_angles_rad = torch.tensor([math.radians(a) for a in cfg['view_angles_deg']], device=device)
    
    with autocast(enabled=use_amp):
        tiffs_in = tiffs_norm.reshape(B*V, H, W)
        masks_in = masks.reshape(B*V, H, W)
        fm = encoder(tiffs_in, masks_in)
        feat_maps = fm.reshape(B, V, fm.shape[1], fm.shape[2], fm.shape[3])

    n_s      = cfg['n_samples']
    near     = cfg['near']
    far      = cfg['far']
    N_rays   = cfg.get('n_rays', 3072)
    t_vals   = torch.linspace(near, far, n_s, device=device)
    noise    = torch.rand_like(t_vals) * (far - near) / n_s
    t_vals   = t_vals + noise
    deltas   = torch.cat([t_vals[1:] - t_vals[:-1], torch.tensor([1e-3], device=device)])
    
    pe_list, feat_list, rays_info = [], [], []
    gt_masks_sampled, gt_tiffs_sampled = [], []
    rendered_masks, rendered_temps = [], []
    
    for v in range(V):
        angle  = cfg['view_angles_deg'][v]
        rays_o, rays_d = get_rays(H, W, angle, device)
        ray_indices = torch.randperm(H * W, device=device)[:N_rays]
        rays_o = rays_o[ray_indices]
        rays_d = rays_d[ray_indices]
        gt_m = masks[:, v].reshape(B, H * W)[:, ray_indices]
        gt_t = tiffs_norm[:, v].reshape(B, H * W)[:, ray_indices]
        gt_masks_sampled.append(gt_m)
        gt_tiffs_sampled.append(gt_t)
        pts    = rays_o.unsqueeze(1) + t_vals.view(1, -1, 1) * rays_d.unsqueeze(1)
        pts    = pts.unsqueeze(0).expand(B, -1, -1, -1)
        pf     = pts.reshape(B, -1, 3)
        
        with autocast(enabled=use_amp):
            pe      = positional_encoding(pf, L=cfg['pos_enc_L'], alpha=alpha)
            feat    = project_and_sample(pf, feat_maps, view_angles_rad)
            sigma, T_pred = mlp_forward(mlp, pe, feat, use_grad_checkpoint=use_grad_checkpoint)
        
        if angle <= -80:
            sigma = sigma.masked_fill(pf[..., 0:1] > 0.1, 0.0)
        elif angle >= 80:
            sigma = sigma.masked_fill(pf[..., 0:1] < -0.1, 0.0)
            
        sigma  = sigma.squeeze(-1).reshape(B, N_rays, n_s)
        T_pred = T_pred.squeeze(-1).reshape(B, N_rays, n_s)
        rm_list, rt_list = [], []
        for b in range(B):
            with autocast(enabled=use_amp):
                rm, rt = volume_render(sigma[b], T_pred[b], deltas, cfg['density_scale'])
            rm_list.append(rm)
            rt_list.append(rt)
        rendered_masks.append(torch.stack(rm_list))
        rendered_temps.append(torch.stack(rt_list))
        
    gt_masks_sampled = torch.stack(gt_masks_sampled, dim=1)
    gt_tiffs_sampled = torch.stack(gt_tiffs_sampled, dim=1)
    
    R_tv = 32
    linspace_tv = torch.linspace(-1, 1, R_tv, device=device)
    zz, yy, xx = torch.meshgrid(linspace_tv, linspace_tv, linspace_tv, indexing='ij')
    pts_tv  = torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3).expand(B, -1, -1)
    with autocast(enabled=use_amp):
        pe_tv   = positional_encoding(pts_tv, L=cfg['pos_enc_L'], alpha=alpha)
        feat_tv = project_and_sample(pts_tv, feat_maps, view_angles_rad)
        sigma_tv, _ = mlp_forward(mlp, pe_tv, feat_tv, use_grad_checkpoint=use_grad_checkpoint)
        
    sigma_tv = sigma_tv.squeeze(-1).reshape(B, R_tv, R_tv, R_tv)
    loss_tv = torch.tensor(0., device=device)
    for b in range(B):
        loss_tv = loss_tv + tv_loss_3d(sigma_tv[b])
    loss_tv = loss_tv / B
    
    loss_dict = compute_loss(rendered_masks, rendered_temps,
                             gt_masks_sampled, gt_tiffs_sampled, cfg, loss_tv=loss_tv)
    return loss_dict, rendered_masks, rendered_temps

# --- Initialize Models ---
encoder_base = SiameseEncoder(in_ch=2, base_ch=16, out_ch=CFG['feat_channels']).to(DEVICE)
pos_ch = 3 * (2 * CFG['pos_enc_L'] + 1)
mlp_base     = ThermamNeRFMLP(pos_ch=pos_ch, feat_ch=CFG['feat_channels'] * CFG['n_views'], hidden_ch=CFG['mlp_hidden'], num_layers=CFG['mlp_layers']).to(DEVICE)

if WORLD_SIZE > 1:
    encoder = DDP(encoder_base, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    mlp     = DDP(mlp_base,     device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
else:
    encoder = encoder_base
    mlp     = mlp_base

params    = list(encoder.parameters()) + list(mlp.parameters())
optimiser = torch.optim.Adam(params, lr=CFG['lr'], betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=CFG['n_epochs'], eta_min=CFG['lr'] * 0.05)
scaler    = GradScaler(enabled=CFG.get('use_amp', False))

history       = {'train_loss': [], 'val_dice': [], 'val_thermal': [], 'val_iou': [], 'val_joint': []}
best_val_dice = float('inf')
epochs_without_improvement = 0

for epoch in range(1, CFG['n_epochs'] + 1):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    encoder.train()
    mlp.train()
    
    alpha = get_alpha(epoch, CFG['freq_warmup_epochs'], CFG['pos_enc_L'])
    cfg_step = dict(CFG)
    
    cfg_step['lambda_tv'] = CFG.get('lambda_tv', 0.01) * max(0.5, math.exp(-0.015 * epoch))
    cfg_step['lambda_bg'] = CFG.get('lambda_bg', 5.0) * min(1.0, epoch / 50.0)
    cfg_step['lambda_entropy'] = CFG.get('lambda_entropy', 0.05) * min(1.0, epoch / 100.0)
    cfg_step['density_scale'] = CFG.get('density_scale', 50.0)
    
    epoch_loss = {k: [] for k in ['total', 'dice', 'thermal', 'tv', 'bg', 'entropy']}
    iterator = train_dl if not IS_MAIN else tqdm(train_dl, desc=f'Epoch {epoch}/{CFG["n_epochs"]}', leave=False)
    for batch in iterator:
        optimiser.zero_grad()
        loss_dict, _, _ = run_one_batch(
            batch, encoder, mlp, cfg_step, alpha, DEVICE,
            use_amp=CFG.get('use_amp', False),
            use_grad_checkpoint=CFG.get('use_grad_checkpoint', False),
        )
        scaler.scale(loss_dict['total']).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()
        for k in epoch_loss.keys():
            epoch_loss[k].append(loss_dict[k].item())

    if WORLD_SIZE > 1:
        loss_tensor = torch.tensor([np.mean(epoch_loss[k]) for k in epoch_loss.keys()], device=DEVICE)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor = loss_tensor / WORLD_SIZE
        mean_losses = {k: loss_tensor[i].item() for i, k in enumerate(epoch_loss.keys())}
    else:
        mean_losses = {k: np.mean(epoch_loss[k]) for k in epoch_loss.keys()}

    scheduler.step()
    history['train_loss'].append(mean_losses['total'])

    if IS_MAIN:
        print(f"[Fold {FOLD_IDX}] Epoch {epoch:03d} | Train Total: {mean_losses['total']:.4f} | "
              f"Dice: {mean_losses['dice']:.4f} | Therm: {mean_losses['thermal']:.4f} | "
              f"TV: {mean_losses['tv']:.4f} | BG: {mean_losses['bg']:.4f} | "
              f"Ent: {mean_losses['entropy']:.4f} | α: {alpha:.2f}")

    # --- Early Stopping Eval Loop ---
    if IS_MAIN and epoch % CFG.get('eval_every', 5) == 0:
        encoder.eval()
        mlp.eval()
        val_dice_losses, val_therm_losses = [], []
        with torch.no_grad():
            for batch in val_dl:
                ld, _, _ = run_one_batch(
                    batch, encoder, mlp, cfg_step, alpha, DEVICE,
                    use_amp=CFG.get('use_amp', False),
                    use_grad_checkpoint=False,
                )
                val_dice_losses.append(ld['dice'].item())
                val_therm_losses.append(ld['thermal'].item())
        vd = np.mean(val_dice_losses)
        vt = np.mean(val_therm_losses)
        dice_score = 1 - vd
        iou_score  = dice_score / (2 - dice_score) if dice_score < 1.0 else 1.0
        
        joint_val = vd + (CFG.get('lambda_thermal', 10.0) / 10.0) * vt
        
        history['val_dice'].append(vd)
        history['val_thermal'].append(vt)
        history['val_iou'].append(iou_score)
        history['val_joint'].append(joint_val)
        
        print(f'  --> [Val] Dice Score: {dice_score:.4f} | Therm MSE: {vt:.4f} | Joint: {joint_val:.4f}')
              
        if joint_val < best_val_dice:
            best_val_dice = joint_val
            epochs_without_improvement = 0
            torch.save({'encoder': encoder_base.state_dict(), 'mlp': mlp_base.state_dict()}, 
                       os.path.join(OUTPUT_DIR, 'thermamnerf_best.pth'))
            print(f'  --> New best model saved! (Joint: {best_val_dice:.4f})')
        else:
            epochs_without_improvement += CFG.get('eval_every', 5)
            print(f'  --> No improvement for {epochs_without_improvement} epochs.')
            
        torch.save({'encoder': encoder_base.state_dict(), 'mlp': mlp_base.state_dict()}, 
                   os.path.join(OUTPUT_DIR, 'thermamnerf_latest.pth'))
                   
        if epochs_without_improvement >= CFG['patience']:
            print(f'\n[Early Stopping] Triggered at epoch {epoch}. Best Joint Metric: {best_val_dice:.4f}')
            break


if WORLD_SIZE > 1:
    dist.barrier()
    dist.destroy_process_group()

if IS_MAIN:
    print(f'Fold {FOLD_IDX} training completed successfully.')
