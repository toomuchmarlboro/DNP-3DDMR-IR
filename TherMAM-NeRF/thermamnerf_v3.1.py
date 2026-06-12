#!/usr/bin/env python3
"""
DDP usage: `torchrun --nproc_per_node=2 thermamnerf_v3.1.py`
"""

import os
import copy
import json
import math
import glob
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
if IS_MAIN:
    print(f'Using device {DEVICE} | world_size={WORLD_SIZE} | rank={RANK}')

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).resolve().parent
REPO_ROOT        = SCRIPT_DIR.parent
TIFF_DIR         = str(REPO_ROOT / 'data' / 'organized_by_patient')
UNET_DIR         = str(REPO_ROOT / 'data' / 'organized_by_patient_unet')
OUTPUT_DIR       = str(SCRIPT_DIR / 'thermamnerf_outputs3.1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────
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
    'density_scale'   : 10.0,    # Reverted to 10.0! 50.0 causes the solid block trap.

    'freq_warmup_epochs': 50,

    'batch_size'      : 1,
    'n_epochs'        : 1000,
    'lr'              : 5e-4,
    'lambda_dice'     : 1.0,
    'lambda_bg'       : 2.0,     # Lowered to 2.0 so Frontal Dice overpowers Lateral BG conflicts
    'lambda_thermal'  : 20.0,    # ACTIVATED! Training thermal & geometry jointly from scratch
    'lambda_tv'       : 0.1,    # Increased to smooth out horizontal striations
    'lambda_entropy'  : 0.0,     # Disabled to stop jagged 'comb' artifacts
    'n_rays'          : 3072,

    'mc_threshold'    : 0.3,
    'mc_resolution'   : 128,

    'use_amp'         : True,
    'use_grad_checkpoint': True,
}
if not torch.cuda.is_available():
    CFG['use_amp'] = False
print('Config loaded.')

## -------------------- Data loading utilities --------------------------------
# Optional: frozen UNet for on-the-fly mask generation
USE_PRECOMPUTED_MASKS = True

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
        else:
            print(f'  [SKIP] {pid} — missing views')

    patients.sort(key=lambda p: int(p['id'].split('_')[-1]) if '_' in p['id'] else p['id'])
    print(f'Found {len(patients)} complete patients.')
    return patients


patients = discover_patients_split(TIFF_DIR, UNET_DIR)
if len(patients) == 0:
    raise RuntimeError(
        f'No complete patients found under TIFF_DIR={TIFF_DIR!r} and UNET_DIR={UNET_DIR!r}. '
        'Check the dataset layout or path resolution before training.'
    )


class BreastThermDataset(Dataset):
    """
    Returns per-patient tensors.
    """
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


import random
random.seed(42)
random.shuffle(patients)
split = int(0.8 * len(patients))
train_patients = patients[:split]
val_patients   = patients[split:]
print(f'Train: {len(train_patients)} | Val: {len(val_patients)}')

train_ds = BreastThermDataset(train_patients, CFG)
val_ds   = BreastThermDataset(val_patients,   CFG)

train_sampler = None
if WORLD_SIZE > 1:
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=True,
        drop_last=False,
    )

train_dl = DataLoader(
    train_ds,
    batch_size=CFG['batch_size'],
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=0,
    pin_memory=False,
)

val_dl = None
if IS_MAIN:
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=False)

# Quick sanity-check (will save figures to OUTPUT_DIR when running headless)
if IS_MAIN and len(train_ds) > 0:
    sample = train_ds[0]
    print('tiffs_norm:', sample['tiffs_norm'].shape,
          '| masks:', sample['masks'].shape,
          '| tiffs_abs range:', sample['tiffs_abs'].min().item(),
          '—', sample['tiffs_abs'].max().item(), '°C')

    try:
        fig, axes = plt.subplots(2, 5, figsize=(16, 6))
        for i, v in enumerate(CFG['view_names']):
            axes[0, i].imshow(sample['tiffs_norm'][i], cmap='inferno')
            axes[0, i].set_title(f'Thermal {v}')
            axes[0, i].axis('off')
            axes[1, i].imshow(sample['masks'][i], cmap='gray')
            axes[1, i].set_title(f'Mask {v}')
            axes[1, i].axis('off')
        plt.suptitle(f'Patient: {sample["patient_id"]}', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'sanity_check_sample.png'), dpi=120)
        plt.close(fig)
    except Exception:
        pass


class SiameseEncoder(nn.Module):
    def __init__(self, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2,  16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(inplace=False),
            nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(inplace=False),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, tiff_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.stack([tiff_norm, mask], dim=1)
        out = self.net(x)
        return out * mask.unsqueeze(1)


def project_and_sample(pts_3d: torch.Tensor,
                        feat_maps: torch.Tensor,
                        view_angles_rad: torch.Tensor) -> torch.Tensor:
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
        grid = torch.stack([xr, yr], dim=-1)
        grid = grid.unsqueeze(1)
        fmap = feat_maps[:, v]
        sampled = F.grid_sample(fmap, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        sampled = sampled.squeeze(2)
        sampled = sampled.permute(0, 2, 1)
        per_view_feats.append(sampled)

    stacked = torch.stack(per_view_feats, dim=2)
    mu  = stacked.mean(dim=2)
    var = stacked.var(dim=2)
    return torch.cat([mu, var], dim=-1)


def positional_encoding(x: torch.Tensor, L: int, alpha: float = None) -> torch.Tensor:
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


def get_alpha(epoch: int, warmup_epochs: int, L: int) -> float:
    return min(L * epoch / max(warmup_epochs, 1), float(L))


class ThermamNeRFMLP(nn.Module):
    def __init__(self, pos_enc_dim: int, feat_dim: int,
                 hidden: int = 256, n_layers: int = 6):
        super().__init__()
        in_dim  = pos_enc_dim + feat_dim
        layers  = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=False)]
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

    def forward(self, pe: torch.Tensor, feat: torch.Tensor) -> tuple:
        x0 = torch.cat([pe, feat], dim=-1)
        h  = x0
        for i, layer in enumerate(self.layers):
            if i == self.skip_at:
                h = torch.cat([h, x0], dim=-1)
            h = F.relu(layer(h), inplace=False)
        sigma  = F.softplus(self.sigma_head(h))
        T_norm = torch.sigmoid(self.temp_head(h))
        return sigma, T_norm


# Instantiate models
L        = CFG['pos_enc_L']
pos_dim  = 3 * 2 * L
feat_dim = CFG['feat_channels'] * 2

encoder  = SiameseEncoder(out_channels=CFG['feat_channels']).to(DEVICE)
mlp      = ThermamNeRFMLP(pos_enc_dim=pos_dim,
                           feat_dim=feat_dim,
                           hidden=CFG['mlp_hidden'],
                           n_layers=CFG['mlp_layers']).to(DEVICE)

if WORLD_SIZE > 1:
    if torch.cuda.is_available():
        encoder = DDP(encoder, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        mlp = DDP(mlp, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    else:
        encoder = DDP(encoder)
        mlp = DDP(mlp)

encoder_base = encoder.module if isinstance(encoder, DDP) else encoder
mlp_base = mlp.module if isinstance(mlp, DDP) else mlp

n_params = sum(p.numel() for p in list(encoder_base.parameters()) + list(mlp_base.parameters()))
if IS_MAIN:
    print(f'Total parameters: {n_params:,}')
    print(f'  encoder -> {DEVICE}')
    print(f'  mlp     -> {DEVICE}')
    if WORLD_SIZE > 1:
        print('  wrapped with DistributedDataParallel')


def get_rays(H: int, W: int, angle_deg: float, device) -> tuple:
    theta = math.radians(angle_deg)
    cam_dir = torch.tensor([-math.sin(theta), 0.0,  math.cos(theta)], device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    right = torch.tensor([math.cos(theta), 0.0, math.sin(theta)], device=device)
    up    = torch.tensor([0.0, 1.0, 0.0], device=device)
    origins = (grid_x.flatten().unsqueeze(1) * right.unsqueeze(0) +
               grid_y.flatten().unsqueeze(1) * up.unsqueeze(0))
    dirs = cam_dir.unsqueeze(0).expand(H * W, -1)
    return origins, dirs


def volume_render(sigma: torch.Tensor, T_field: torch.Tensor,
                  deltas: torch.Tensor, density_scale: float) -> tuple:
    sigma   = sigma * density_scale
    alpha   = 1.0 - torch.exp(-sigma * deltas.unsqueeze(0))
    T_trans = torch.cumprod(
        torch.cat([torch.ones(alpha.shape[0], 1, device=alpha.device),
                   1.0 - alpha[:, :-1] + 1e-10], dim=1), dim=1)
    weights = alpha * T_trans
    rendered_mask = weights.sum(dim=1)
    
    # DETACH weights so thermal loss only updates T_field and doesn't destroy geometry!
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
        
        # --- VIEW-SPECIFIC DEPTH CROPPING ---
        # The ribcage is curved. From a strict 90-degree side view, the far breast physically 
        # sticks out behind the near breast. Because GT masks only include the near breast, 
        # we artificially hide the far breast during lateral rendering so it doesn't get penalized!
        if angle <= -80:  # RL (Right Lateral) - Hide the Left breast (X > 0.1)
            sigma = sigma.masked_fill(pts_flat[..., 0:1] > 0.1, 0.0)
        elif angle >= 80: # LL (Left Lateral) - Hide the Right breast (X < -0.1)
            sigma = sigma.masked_fill(pts_flat[..., 0:1] < -0.1, 0.0)
            
        sigma  = sigma.squeeze(-1).squeeze(0).reshape(R_, n_samples)
        T_pred = T_pred.squeeze(-1).squeeze(0).reshape(R_, n_samples)
        rm, rt = volume_render(sigma, T_pred, deltas, cfg['density_scale'])
        all_mask.append(rm)
        all_temp.append(rt)
    rendered_mask = torch.cat(all_mask).reshape(H, W)
    rendered_temp = torch.cat(all_temp).reshape(H, W)
    return rendered_mask, rendered_temp


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred   = pred.flatten()
    target = target.flatten()
    inter  = (pred * target).sum()
    return 1.0 - (2.0 * inter + eps) / (pred.sum() + target.sum() + eps)


def tv_loss_3d(field: torch.Tensor) -> torch.Tensor:
    dx = (field[1:, :, :] - field[:-1, :, :]).abs().mean()
    dy = (field[:, 1:, :] - field[:, :-1, :]).abs().mean()
    dz = (field[:, :, 1:] - field[:, :, :-1]).abs().mean()
    return dx + dy * 3.0 + dz


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
        
        # Ray opacity penalty to gently snap semi-transparent streaks to 0 or 1
        loss_entropy = loss_entropy + (pred_m * (1.0 - pred_m)).mean()

    loss_dice    /= cfg['n_views']
    loss_thermal /= cfg['n_views']
    loss_bg      /= cfg['n_views']
    loss_entropy /= cfg['n_views']

    total = (cfg['lambda_dice']    * loss_dice +
             cfg['lambda_thermal'] * loss_thermal +
             cfg.get('lambda_tv', 0.0) * loss_tv +
             cfg.get('lambda_bg', 1.0) * loss_bg +
             cfg.get('lambda_entropy', 0.0) * loss_entropy)
    return {'total': total, 'dice': loss_dice, 'thermal': loss_thermal, 'tv': loss_tv, 'bg': loss_bg, 'entropy': loss_entropy}


def background_ray_loss(rendered_masks: list, gt_masks: torch.Tensor) -> torch.Tensor:
    loss = torch.tensor(0.0, device=gt_masks.device)
    V = gt_masks.shape[1]
    for v in range(V):
        bg_pixels = (gt_masks[:, v] < 0.5)
        if bg_pixels.any():
            loss += rendered_masks[v][bg_pixels].pow(2).mean()
    return loss / V


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
    view_angles_rad   = torch.tensor([math.radians(a) for a in cfg['view_angles_deg']], device=device)
    feat_maps = []
    with autocast(enabled=use_amp):
        for v in range(V):
            fm = encoder(tiffs_norm[:, v], masks[:, v])
            feat_maps.append(fm)
    feat_maps   = torch.stack(feat_maps, dim=1)
    n_s      = cfg['n_samples']
    near     = cfg['near']
    far      = cfg['far']
    N_rays   = cfg.get('n_rays', 4096)
    t_vals   = torch.linspace(near, far, n_s, device=device)
    noise    = torch.rand_like(t_vals) * (far - near) / n_s
    t_vals   = t_vals + noise
    deltas   = torch.cat([t_vals[1:] - t_vals[:-1], torch.tensor([1e-3], device=device)])
    rendered_masks, rendered_temps = [], []
    gt_masks_sampled, gt_tiffs_sampled = [], []
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
            
        # --- VIEW-SPECIFIC DEPTH CROPPING ---
        if angle <= -80:  # RL: Hide Left breast
            sigma = sigma.masked_fill(pf[..., 0:1] > 0.1, 0.0)
        elif angle >= 80: # LL: Hide Right breast
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
        loss_tv += tv_loss_3d(sigma_tv[b])
    loss_tv /= B
    loss_dict = compute_loss(rendered_masks, rendered_temps,
                             gt_masks_sampled, gt_tiffs_sampled, cfg, loss_tv=loss_tv)
    return loss_dict, rendered_masks, rendered_temps


params    = list(encoder.parameters()) + list(mlp.parameters())
optimiser = torch.optim.Adam(params, lr=CFG['lr'], betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=CFG['n_epochs'], eta_min=CFG['lr'] * 0.05)
scaler    = GradScaler(enabled=CFG.get('use_amp', False))
if IS_MAIN:
    print(f"AMP enabled: {CFG.get('use_amp', False)} | Gradient checkpointing: {CFG.get('use_grad_checkpoint', False)}")

RESUME      = False
START_EPOCH = 1
ckpt_path   = os.path.join(OUTPUT_DIR, 'thermamnerf_best.pth')
if RESUME and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    enc_state = {k.replace('module.', ''): v for k, v in ckpt['encoder'].items()}
    mlp_state = {k.replace('module.', ''): v for k, v in ckpt['mlp'].items()}
    encoder_base.load_state_dict(enc_state)
    mlp_base.load_state_dict(mlp_state)
    if IS_MAIN:
        print('\nResumed weights')
    START_EPOCH = 41
    for _ in range(START_EPOCH - 1):
        scheduler.step()


history       = {'train_loss': [], 'val_dice': [], 'val_thermal': [], 'val_iou': [], 'val_joint': []}
best_val_dice = float('inf')

for epoch in tqdm(range(START_EPOCH, CFG['n_epochs'] + 1), desc='Training Epochs'):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    encoder.train()
    mlp.train()
    
    alpha = get_alpha(epoch, CFG['freq_warmup_epochs'], CFG['pos_enc_L'])
    cfg_step = dict(CFG)
    
    # TV loss decays but bottoms out at 0.005 to keep boundaries smooth
    cfg_step['lambda_tv'] = CFG.get('lambda_tv', 0.01) * max(0.5, math.exp(-0.015 * epoch))
    
    # Slow ramp-up over 50 epochs: prevents solid block trap AND prevents collapse!
    cfg_step['lambda_bg'] = CFG.get('lambda_bg', 5.0) * min(1.0, epoch / 50.0)
    
    # Entropy loss ramps up to clean streaks gently
    cfg_step['lambda_entropy'] = CFG.get('lambda_entropy', 0.05) * min(1.0, epoch / 100.0)
    
    cfg_step['density_scale'] = CFG.get('density_scale', 50.0)
    
    epoch_loss = []
    for batch in tqdm(train_dl, desc=f'Epoch {epoch}/{CFG["n_epochs"]}', leave=False):
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
        epoch_loss.append(loss_dict['total'].item())

    if WORLD_SIZE > 1:
        loss_tensor = torch.tensor([np.mean(epoch_loss)], device=DEVICE)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        mean_train = (loss_tensor.item() / WORLD_SIZE)
    else:
        mean_train = np.mean(epoch_loss)

    scheduler.step()
    history['train_loss'].append(mean_train)

    if IS_MAIN:
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
        
        # Joint validation metric (lower is better): Dice Loss + scaled Thermal MSE
        # Since thermal loss is MSE (~0.05), we scale it to be comparable to Dice loss (~0.1)
        joint_val = vd + (CFG.get('lambda_thermal', 10.0) / 10.0) * vt
        
        history['val_dice'].append(vd)
        history['val_thermal'].append(vt)
        history['val_iou'].append(iou_score)
        history['val_joint'].append(joint_val)
        
        if joint_val < best_val_dice:
            best_val_dice = joint_val
            torch.save({'encoder': encoder_base.state_dict(), 'mlp': mlp_base.state_dict()}, os.path.join(OUTPUT_DIR, 'thermamnerf_best.pth'))
        
        # Also always save the latest model
        torch.save({'encoder': encoder_base.state_dict(), 'mlp': mlp_base.state_dict()}, os.path.join(OUTPUT_DIR, 'thermamnerf_latest.pth'))
        
        print(f'Epoch {epoch:03d}/{CFG["n_epochs"]} | ' \
              f'Train Loss: {mean_train:.4f} | ' \
              f'Val Dice Loss: {vd:.4f} (Dice: {dice_score:.4f}) | ' \
              f'Val IoU: {iou_score:.4f} | ' \
              f'Val Thermal MSE: {vt:.4f} | ' \
              f'α: {alpha:.1f} | ' \
              f'ds: {cfg_step["density_scale"]:.1f} | ' \
              f'λ_tv: {cfg_step["lambda_tv"]:.4f}')


# ── Post-training visualisation & export (will not open interactive windows)
if IS_MAIN:
    try:
        ckpt = torch.load(os.path.join(OUTPUT_DIR, 'thermamnerf_best.pth'), map_location=DEVICE)
        encoder_base.load_state_dict(ckpt['encoder'])
        mlp_base.load_state_dict(ckpt['mlp'])
        encoder_base.eval(); mlp_base.eval()
        alpha_final = float(CFG['pos_enc_L'])
        
        all_val_dice = []
        all_val_iou  = []
        patient_ids  = []
        
        print('\nStarting full cohort performance review...')
        full_ds = BreastThermDataset(patients, CFG)
        for idx in tqdm(range(len(full_ds)), desc='Auditing Patients'):
            sample      = full_ds[idx]
            tiffs_norm  = sample['tiffs_norm'].to(DEVICE)
            tiffs_abs   = sample['tiffs_abs'].to(DEVICE)
            masks       = sample['masks'].to(DEVICE)
            patient_id  = sample['patient_id']
            patient_ids.append(patient_id)
            
            fig, axes = plt.subplots(3, 5, figsize=(18, 10))
            dice_scores = []
            iou_scores  = []
            for v, vname in enumerate(CFG['view_names']):
                angle = CFG['view_angles_deg'][v]
                rm, rt = render_view(encoder_base, mlp_base, tiffs_norm, masks, v, CFG, DEVICE, alpha=alpha_final)
                gt_mask = masks[v].cpu().numpy()
                gt_temp = tiffs_norm[v].cpu().numpy()
                pred_m  = rm.cpu().numpy()
                pred_t  = rt.cpu().numpy()
                
                pred_bin = (pred_m > 0.5).astype(np.float32)
                inter    = (pred_bin * gt_mask).sum()
                union    = pred_bin.sum() + gt_mask.sum() - inter
                
                dsc      = 2 * inter / (pred_bin.sum() + gt_mask.sum() + 1e-6)
                iou      = inter / (union + 1e-6)
                
                dice_scores.append(dsc)
                iou_scores.append(iou)
                
                axes[0, v].imshow(gt_mask, cmap='gray')
                axes[0, v].set_title(f'GT Mask\n{vname} ({angle}°)', fontsize=9)
                axes[0, v].axis('off')
                
                axes[1, v].imshow(pred_m, cmap='gray', vmin=0, vmax=1)
                axes[1, v].set_title(f'Rendered Mask\nDice={dsc:.3f} | IoU={iou:.3f}', color='green' if dsc > 0.85 else 'red', fontsize=9)
                axes[1, v].axis('off')
                
                axes[2, v].imshow(pred_t, cmap='inferno', vmin=0, vmax=1)
                axes[2, v].set_title('Rendered Thermal', fontsize=9)
                axes[2, v].axis('off')
            
            mean_dsc = np.mean(dice_scores)
            mean_iou = np.mean(iou_scores)
            all_val_dice.append(mean_dsc)
            all_val_iou.append(mean_iou)
            
            plt.suptitle(f'Projection Audit: {patient_id} | Mean Dice: {mean_dsc:.3f} | Mean IoU: {mean_iou:.3f}', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'audit_{patient_id}.png'), dpi=120)
            plt.close(fig)
            
            print(f'Patient {patient_id} | Mean Dice: {mean_dsc:.3f} | Mean IoU: {mean_iou:.3f}')
            
        print(f'\n--- Final Cohort Validation Performance ({len(full_ds)} patients) ---')
        print(f'Overall Average Dice: {np.mean(all_val_dice):.4f}')
        print(f'Overall Average IoU:  {np.mean(all_val_iou):.4f}')
        
        # --- Cohort Performance Plot ---
        fig, ax = plt.subplots(figsize=(12, 6))
        x_indices = np.arange(len(patient_ids))
        width = 0.35
        
        ax.bar(x_indices - width/2, all_val_dice, width, label='Dice Score', color='blue', alpha=0.7)
        ax.bar(x_indices + width/2, all_val_iou, width, label='IoU Score', color='green', alpha=0.7)
        
        ax.set_xlabel('Patients')
        ax.set_ylabel('Score')
        ax.set_title('Cohort Performance Review: Dice & IoU per Patient')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(patient_ids, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0.0, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'cohort_performance_plot.png'), dpi=150)
        plt.close(fig)
        print('Cohort performance plot saved to cohort_performance_plot.png')
        
    except Exception as e:
        print('Projection audit skipped:', e)

def extract_3d_volume(encoder, mlp, tiffs_norm, masks, cfg, device,
                       resolution: int = None, chunk: int = 8192) -> tuple:
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


def export_ply(verts, faces, colors_rgb, filepath, temperatures=None):
    temp_prop = 'property float temperature\n' if temperatures is not None else ''
    header = (
        f"ply\nformat ascii 1.0\n"
        f"element vertex {len(verts)}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"{temp_prop}"
        f"element face {len(faces)}\n"
        f"property list uchar int vertex_indices\nend_header\n"
    )
    with open(filepath, 'w') as f:
        f.write(header)
        if temperatures is not None:
            for v, c, t in zip(verts, colors_rgb, temperatures):
                f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} ' f'{int(c[0])} {int(c[1])} {int(c[2])} {t:.4f}\n')
        else:
            for v, c in zip(verts, colors_rgb):
                f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} ' f'{int(c[0])} {int(c[1])} {int(c[2])}\n')
        for face in faces:
            f.write(f'3 {face[0]} {face[1]} {face[2]}\n')
    print(f'Saved PLY: {filepath}')


# ── Training Curves ──────────────────────────────────────────────────────────
if IS_MAIN:
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Train Loss
        axes[0, 0].plot(history['train_loss'], color='blue', label='Train Loss')
        axes[0, 0].set_title('Train Loss (Total)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].grid(True)

        # Val Dice Score (1 - Dice Loss)
        val_dice_scores = [1 - d for d in history['val_dice']]
        axes[0, 1].plot(val_dice_scores, color='green', marker='o', markersize=3)
        axes[0, 1].set_title('Val Dice Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylim(0.0, 1.0)
        axes[0, 1].grid(True)

        # Val Joint Metric (Checkpoint metric)
        axes[1, 0].plot(history['val_joint'], color='orange', marker='o', markersize=3)
        axes[1, 0].set_title('Val Joint Metric (vd + scaled_vt)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].grid(True)

        # Val Thermal MSE
        axes[1, 1].plot(history['val_thermal'], color='red', marker='o', markersize=3)
        axes[1, 1].set_title('Val Thermal MSE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].grid(True)

        plt.suptitle('TherMAM-NeRF v2.2 Joint Training (λ_bg=2.0, λ_therm=10.0)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=120)
        plt.close(fig)
        print(f"Saved training curves to {os.path.join(OUTPUT_DIR, 'training_curves.png')}")
    except Exception as e:
        print('Training curves plot skipped:', e)

if WORLD_SIZE > 1:
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    print('Script executed directly. Training has already run in top-level scope.')
