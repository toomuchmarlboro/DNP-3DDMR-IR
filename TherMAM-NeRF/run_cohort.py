#!/usr/bin/env python
"""Standalone, resumable cohort runner for the TherMAM-NeRF -> PINN v3
bioheat pipeline. Assembled from Thermamnerf_PINN_v3.ipynb (verbatim defs).
DO NOT hand-edit the definitions below; regenerate via /tmp/build_cohort.py.

Run in tmux (unbuffered, logged):
    conda activate bioheat
    python -u run_cohort.py --synthetic            # validate method first
    python -u run_cohort.py --all | tee cohort.log # full cohort, resumable
    python -u run_cohort.py --subset 30            # balanced subset
See --help for all options.
"""


# ===========================================================================
# Verbatim definitions from Thermamnerf_PINN_v3.ipynb
# ===========================================================================

import os, sys, math, time, random, struct
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh
import plotly.graph_objects as go
import plotly.io as pio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device  : {device}')
print(f'trimesh : {trimesh.__version__}')


REPO_ROOT   = Path(__file__).resolve().parent.parent
TIFF_DIR    = str(REPO_ROOT / 'data' / 'organized_by_patient')
UNET_DIR    = str(REPO_ROOT / 'data' / 'organized_by_patient_unet')
NERF_CKPT   = str(REPO_ROOT / 'TherMAM-NeRF' / 'thermamnerf_outputs2.9' / 'thermamnerf_best.pth')
RESULTS_DIR = Path(REPO_ROOT / 'TherMAM-NeRF' / 'PINNpdeSolver' / 'results')
STL_DIR     = Path(REPO_ROOT / 'TherMAM-NeRF' / 'PINNpdeSolver' / 'exported_stls')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STL_DIR.mkdir(parents=True, exist_ok=True)

print(f'TIFF dir  : {Path(TIFF_DIR).exists()}')
print(f'UNET dir  : {Path(UNET_DIR).exists()}')
print(f'NeRF ckpt : {Path(NERF_CKPT).exists()}')


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
    'far'             :  1.0,
    'density_scale'   : 10.0,
    'mc_threshold'    : 0.3,
    'mc_resolution'   : 128,
    'breast_radius_mm': 70.0,
}

class SiameseEncoder(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2,  16, 3, padding=1), nn.GroupNorm(4, 16),  nn.ReLU(inplace=False),
            nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(8, 32),  nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32),  nn.ReLU(inplace=False),
            nn.Conv2d(32, out_channels, 1),
        )
    def forward(self, tiff_norm, mask):
        return self.net(torch.stack([tiff_norm, mask], dim=1)) * mask.unsqueeze(1)

class ThermamNeRFMLP(nn.Module):
    def __init__(self, pos_enc_dim, feat_dim, hidden=256, n_layers=6):
        super().__init__()
        in_dim = pos_enc_dim + feat_dim
        self.layers  = nn.ModuleList()
        self.skip_at = n_layers // 2 - 1
        prev = in_dim
        for i in range(n_layers - 1):
            self.layers.append(
                nn.Linear(prev + in_dim if i == self.skip_at else prev, hidden))
            prev = hidden
        self.sigma_head = nn.Linear(hidden, 1)
        self.temp_head  = nn.Linear(hidden, 1)
    def forward(self, pe, feat):
        x0 = torch.cat([pe, feat], dim=-1)
        h  = x0
        for i, layer in enumerate(self.layers):
            if i == self.skip_at: h = torch.cat([h, x0], dim=-1)
            h = F.relu(layer(h), inplace=False)
        return F.softplus(self.sigma_head(h)), torch.sigmoid(self.temp_head(h))

def positional_encoding(x, L, alpha=None):
    freqs = 2.0 ** torch.arange(L, dtype=torch.float32, device=x.device)
    x_freq = x.unsqueeze(-1) * freqs * math.pi
    sin_p, cos_p = torch.sin(x_freq), torch.cos(x_freq)
    if alpha is not None:
        k = torch.arange(L, dtype=torch.float32, device=x.device)
        w = 0.5 * (1 - torch.cos(math.pi * torch.clamp(alpha - k, 0., 1.)))
        sin_p = sin_p * w.view(1, 1, 1, L)
        cos_p = cos_p * w.view(1, 1, 1, L)
    return torch.cat([sin_p, cos_p], dim=-1).flatten(-2)

def project_and_sample(pts_3d, feat_maps, view_angles_rad):
    per_view = []
    for v in range(feat_maps.shape[1]):
        theta = view_angles_rad[v]
        xr = torch.cos(theta)*pts_3d[...,0] + torch.sin(theta)*pts_3d[...,2]
        yr = pts_3d[...,1]
        grid = torch.stack([xr, yr], dim=-1).unsqueeze(1)
        sampled = F.grid_sample(feat_maps[:,v], grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        per_view.append(sampled.squeeze(2).permute(0,2,1))
    stacked = torch.stack(per_view, dim=2)
    return torch.cat([stacked.mean(dim=2), stacked.var(dim=2)], dim=-1)

print('Model classes defined.')


def load_tiff_celsius(path, target_size):
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim == 3: arr = arr[..., 0]
    return np.array(Image.fromarray(arr, mode='F').resize(
        (target_size, target_size), Image.BILINEAR), dtype=np.float32)

def load_mask(path, target_size):
    return (np.array(Image.open(str(path)).convert('L').resize(
        (target_size, target_size), Image.NEAREST),
        dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

def normalize_thermal(arr):
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

def discover_patients(tiff_base, unet_base):
    tb, ub = Path(tiff_base), Path(unet_base)
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
            patients.append({'id': pid, 'label': lab,
                             'tiffs': d['tiffs'], 'masks': d['masks']})
    patients.sort(key=lambda p: p['id'])
    print(f'Found {len(patients)} complete patients.')
    return patients

class BreastThermDataset(Dataset):
    def __init__(self, patient_list, cfg):
        self.patients   = patient_list
        self.S          = cfg['img_size']
        self.view_names = cfg['view_names']
    def __len__(self): return len(self.patients)
    def __getitem__(self, idx):
        p = self.patients[idx]
        tiffs_norm, tiffs_abs, masks, tmins, tmaxs = [], [], [], [], []
        for v in self.view_names:
            raw         = load_tiff_celsius(str(p['tiffs'][v]), self.S)
            normd, tmin, tmax = normalize_thermal(raw)
            tiffs_norm.append(normd); tiffs_abs.append(raw)
            masks.append(load_mask(str(p['masks'][v]), self.S))
            tmins.append(tmin); tmaxs.append(tmax)
        return {
            'patient_id' : p['id'],
            'label'      : p.get('label', 'unknown'),
            'tiffs_norm' : torch.tensor(np.stack(tiffs_norm), dtype=torch.float32),
            'tiffs_abs'  : torch.tensor(np.stack(tiffs_abs),  dtype=torch.float32),
            'masks'      : torch.tensor(np.stack(masks),      dtype=torch.float32),
            'tmin'       : torch.tensor(tmins, dtype=torch.float32),
            'tmax'       : torch.tensor(tmaxs, dtype=torch.float32),
        }


@torch.no_grad()
def extract_3d_volume(encoder, mlp, tiffs_norm, masks, cfg, device,
                      resolution=None, chunk=8192):
    """
    tiffs_norm, masks : unbatched [V, H, W] tensors on device.
    Returns sigma_grid, T_grid — each [R, R, R] numpy array.
    """
    R = resolution or cfg['mc_resolution']
    linspace = torch.linspace(-1, 1, R, device=device)
    zz, yy, xx = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)
    view_angles_rad = torch.tensor(
        [math.radians(a) for a in cfg['view_angles_deg']], device=device)
    feat_maps = torch.stack([
        encoder(tiffs_norm[v:v+1], masks[v:v+1])
        for v in range(cfg['n_views'])
    ], dim=1)
    alpha_final = float(cfg['pos_enc_L'])
    sigma_all, T_all = [], []
    for i in range(0, pts.shape[1], chunk):
        p    = pts[:, i:i+chunk]
        pe   = positional_encoding(p, L=cfg['pos_enc_L'], alpha=alpha_final)
        feat = project_and_sample(p, feat_maps, view_angles_rad)
        sg, tp = mlp(pe, feat)
        sigma_all.append(sg.squeeze().cpu())
        T_all.append(tp.squeeze().cpu())
    sigma_grid = torch.cat(sigma_all).reshape(R, R, R).numpy()
    T_grid     = torch.cat(T_all).reshape(R, R, R).numpy()
    return sigma_grid, T_grid

print('extract_3d_volume ready.')


def sample_interior_points(verts_mm, n_points=5000):
    bbox_min = verts_mm.min(axis=0)
    bbox_max = verts_mm.max(axis=0)
    centroid = verts_mm.mean(axis=0)
    extents  = (bbox_max - bbox_min) / 2
    pts = np.random.uniform(bbox_min, bbox_max, size=(n_points * 3, 3))
    inside = np.all(np.abs(pts - centroid) / (extents + 1e-8) < 1.0, axis=1)
    selected = pts[inside][:n_points]
    if len(selected) < n_points:
        extra = np.random.uniform(bbox_min, bbox_max,
                                   size=(n_points - len(selected), 3))
        selected = np.vstack([selected, extra])
    return selected.astype(np.float32)


def process_patient_geometry_from_nerf(sigma_grid, T_grid, tmin, tmax, cfg,
                                        gaussian_sigma=2.0,
                                        mc_threshold_override=None,
                                        n_interior=20000,
                                        repair_holes=True,
                                        laplacian_iters=5,
                                        thermal_sigma=1.0):
    """
    NeRF grids → geometry dict ready for the PINN solver.

    Parameters
    ----------
    gaussian_sigma        : smoothing applied to sigma_grid ONLY (geometry)
    thermal_sigma         : light smoothing on T_grid to blend view-projection
                            seams (0 = off). Keep small (~1.0) to preserve the
                            hot/cold gradient the PINN relies on.
    mc_threshold_override : use this instead of cfg['mc_threshold'] (e.g. 0.2 for
                            patients with apex cavity artifacts)
    repair_holes          : run trimesh hole fill + normal fix
    laplacian_iters       : trimesh Laplacian smoothing iterations (0 = off)
    """
    R         = cfg['mc_resolution']
    BR        = cfg['breast_radius_mm']
    threshold = mc_threshold_override if mc_threshold_override is not None \
                else cfg['mc_threshold']

    # ── Step 1: Smooth sigma only (geometry) ──
    sigma_smooth = gaussian_filter(sigma_grid, sigma=gaussian_sigma)

    # ── Step 2: Marching cubes in voxel space ──
    verts_vox, faces, normals, _ = marching_cubes(sigma_smooth, level=threshold)

    # ── Step 2.5: Axis reorder — marching_cubes returns verts in array-index
    # order (axis0, axis1, axis2). sigma_smooth was built via
    # torch.meshgrid(linspace, linspace, linspace, indexing='ij') as
    # zz, yy, xx -> stacked as [xx, yy, zz], so the ARRAY axes are (Z, Y, X)
    # but verts_vox columns come out as (axis0, axis1, axis2) = (Z, Y, X).
    # Reorder columns to (X, Y, Z) before converting to physical mm so that
    # X = RL<->LL matches the Frontal view's horizontal axis. ──
    verts_vox = verts_vox[:, [2, 1, 0]]  # (Z,Y,X) -> (X,Y,Z)

    # ── Step 3: Voxel → normalised [-1,1] → physical mm ──
    verts_norm = verts_vox / (R - 1) * 2.0 - 1.0
    verts_mm   = verts_norm * BR

    # ── Step 4: Trimesh — largest component + hole repair + smoothing ──
    mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False)

    # Keep only the largest connected component
    components = mesh.split(only_watertight=False)
    mesh = max(components, key=lambda m: len(m.faces))
    print(f'  largest component: {len(mesh.vertices):,} verts, '
          f'{len(mesh.faces):,} faces')

    if repair_holes:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        # Second pass for stubborn holes
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
        print(f'  after repair     : watertight={mesh.is_watertight}')

    if laplacian_iters > 0:
        trimesh.smoothing.filter_laplacian(mesh, lamb=0.3,
                                            iterations=laplacian_iters)
        print(f'  laplacian smooth : {laplacian_iters} iters')

    verts_mm = np.array(mesh.vertices, dtype=np.float32)
    faces    = np.array(mesh.faces)
    # outward unit vertex normals — needed for the Robin skin BC (v2 Heavy)
    vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)

    # ── Step 5: Sample T at surface vertices ──
    # Light smoothing on T_grid blends the view-projection seams (e.g. the
    # cold purple band where RO and Frontal projections meet) without washing
    # out the hot/cold gradient. Geometry (sigma) is unaffected.
    T_grid_used = gaussian_filter(T_grid, sigma=thermal_sigma) \
                  if thermal_sigma > 0 else T_grid
    if thermal_sigma > 0:
        print(f'  thermal smooth   : sigma={thermal_sigma}')

    # Re-map cleaned verts back to voxel space for map_coordinates.
    # T_grid has the same (Z,Y,X) array-axis order as sigma_smooth, so the
    # column order passed to map_coordinates must match — reorder back
    # (X,Y,Z) -> (Z,Y,X) before sampling.
    verts_vox_clean = (verts_mm / BR + 1.0) / 2.0 * (R - 1)
    verts_vox_for_sample = verts_vox_clean[:, [2, 1, 0]]  # (X,Y,Z) -> (Z,Y,X)
    temp_norm  = map_coordinates(T_grid_used, verts_vox_for_sample.T,
                                  order=1, mode='nearest').clip(0.0, 1.0)
    # Denormalize: frontal-view tmin/tmax as anchor
    T_measured = (temp_norm * (tmax - tmin) + tmin).astype(np.float32)

    confidence   = np.ones(len(verts_mm), dtype=np.float32)
    interior_pts = sample_interior_points(verts_mm, n_points=n_interior)
    bbox_min     = verts_mm.min(axis=0)
    bbox_max     = verts_mm.max(axis=0)

    return {
        'surface_pts' : verts_mm,
        'T_measured'  : T_measured,
        'confidence'  : confidence,
        'interior_pts': interior_pts,
        'bbox_min'    : bbox_min.astype(np.float32),
        'bbox_max'    : bbox_max.astype(np.float32),
        'bbox_extents': (bbox_max - bbox_min).astype(np.float32),
        'verts_raw'   : verts_mm,
        'faces'       : faces,
        'vertex_normals': vertex_normals,
    }


def load_patient_geo(dataset, idx, encoder, mlp, cfg, device,
                     gaussian_sigma=2.0,
                     mc_threshold_override=None,
                     repair_holes=True,
                     laplacian_iters=5,
                     thermal_sigma=1.0):
    """
    One-liner helper: dataset index → fully preprocessed geo dict.
    Change only VIZ_IDX and re-run to browse patients.

    mc_threshold_override : set to 0.2 for patients with apex cavity artifacts,
                            None to use cfg default (0.3)
    thermal_sigma         : light T_grid smoothing to blend projection seams
                            (1.0 default, 0 to disable)
    """
    item     = dataset[idx]
    tiffs_n  = item['tiffs_norm'].to(device)
    masks_   = item['masks'].to(device)
    tmin_ref = item['tmin'][2].item()   # frontal view = most reliable anchor
    tmax_ref = item['tmax'][2].item()

    print(f'Extracting volume for {item["patient_id"]} ({item["label"]}) …')
    sg, tg = extract_3d_volume(encoder, mlp, tiffs_n, masks_, cfg, device)

    # ── Per-patient physical scale calibration ────────────────────────────
    # FLIR SC620 at 1.0 m, 24° HFoV → 425 mm scene / 640 px raw sensor.
    # After the 640 → img_size resize: 1 px = 425 / img_size mm.
    # Measure the breast width in the frontal mask (view index 2 = 'F') and
    # use half that as breast_radius_mm, replacing the hardcoded default.
    # This eliminates the Laplacian scale error (K∇²T ∝ 1/extent²) that would
    # otherwise inflate or deflate the PDE loss for every non-average patient.
    _FOV_MM = 425.0
    _frontal = (item['masks'][2] > 0).cpu().numpy()  # [H, W]
    _cols    = np.where(_frontal.any(axis=0))[0]
    if len(_cols) >= 2:
        _w_px  = int(_cols[-1] - _cols[0]) + 1
        _br_mm = _w_px * (_FOV_MM / cfg['img_size']) / 2.0
        patient_cfg = dict(cfg); patient_cfg['breast_radius_mm'] = _br_mm
        print(f'  calibrated breast_radius_mm: {_br_mm:.1f} mm '
              f'(mask {_w_px}px wide, default was {cfg["breast_radius_mm"]:.1f}mm)')
    else:
        patient_cfg = cfg
        print(f'  WARNING: empty frontal mask — using default '
              f'breast_radius_mm={cfg["breast_radius_mm"]:.1f}mm')

    geo = process_patient_geometry_from_nerf(
        sg, tg, tmin_ref, tmax_ref, patient_cfg,
        gaussian_sigma=gaussian_sigma,
        mc_threshold_override=mc_threshold_override,
        repair_holes=repair_holes,
        laplacian_iters=laplacian_iters,
        thermal_sigma=thermal_sigma,
    )
    geo['patient_id'] = item['patient_id']
    geo['label']      = item['label']
    print(f'  T_measured: {geo["T_measured"].min():.1f}–'
          f'{geo["T_measured"].max():.1f}°C')
    return geo


print('Geometry pipeline ready.')


def save_stl_binary(filepath, verts, faces):
    with open(filepath, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(faces)))
        for face in faces:
            tri = verts[face].astype(np.float32)
            v0, v1, v2 = tri
            normal = np.cross(v1-v0, v2-v0)
            norm   = np.linalg.norm(normal)
            normal = (normal/norm).astype(np.float32) if norm > 0 \
                     else np.zeros(3, dtype=np.float32)
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))
    print(f'STL saved → {filepath}')


# ── Pennes bioheat constants — IDENTICAL set shared by PINN and FEA ──────────
# (Bezerra et al. 2013; Mukhmetov et al. 2025)
K_TISSUE   = 0.48      # W/(m·K)   tissue conductivity
OMEGA_B    = 0.0005    # 1/s       blood perfusion rate
C_BLOOD    = 3600.0    # J/(kg·K)  blood specific heat
RHO_BLOOD  = 1.0       # kg/m³ multiplier — set ≈1050 IF OMEGA_B does NOT already
                       # fold in blood density. Kept 1.0 so PINN ≡ FEA exactly.
T_ARTERIAL = 37.0      # °C        core / arterial temperature (chest-wall BC)
Q_METAB    = 450.0     # W/m³      basal (healthy-tissue) metabolic heat
H_CONV     = 10.0      # W/(m²·K)  skin convective film coefficient (FEA fwd only)
T_AIR      = 20.0      # °C        ambient air temperature        (FEA fwd only)
PERFUSION  = OMEGA_B * RHO_BLOOD * C_BLOOD   # W/(m³·K) lumped perfusion coeff

# Chest-wall identification — shared by PINN BC and FEA BC for consistency.
CHEST_WALL_FRAC = 0.10   # back 10 % of the depth (z) range is "chest wall"
CHEST_WALL_SIDE = 'max'  # body/chest wall is at HIGH z in this reconstruction:
                         # the ANTERIOR skin (real IR, breast bulge) is at LOW z,
                         # confirmed visually (chestwall_check.png). Previously 'min'
                         # mislabeled the anterior skin as chest wall -> Dirichlet 37C
                         # on the wrong side. Fixed 2026-06-17.

# ── Tumour heat magnitude is FIXED by physiology, NOT learned ───────────────
# Bezerra (2013) sensitivity analysis proves the tumour heat MAGNITUDE has
# near-zero surface sensitivity → it is NOT identifiable from skin IR.  Mukhmetov
# (2025) therefore fixes it and trains ONLY the geometry (centre + radius).  We
# follow that: the volumetric tumour heat is tied to its size via the Gautherie
# doubling-time law (the same relation Bezerra uses):
#     Qm_tumour · τ = C ,  C = 3.27e6 W·day/m³
#     D = 0.01 · exp[0.002134 (τ − 50)]   (D = diameter [m], τ = doubling time [day])
# Inverting:  τ(D) = 50 + ln(D/0.01)/0.002134 ,  then  Qm = C/τ.
# Sanity (Bezerra Table 1):  D=1cm → Qm≈65400 ;  D=2.2cm → Qm≈7800 W/m³.
DOUBLING_C = 3.27e6      # W·day/m³

def tumor_Qm_from_radius(r_t_mm):
    """Volumetric tumour metabolic heat [W/m³] for radius r_t_mm, via the
    Gautherie doubling-time law. Differentiable (accepts a torch scalar)."""
    D_m = 2.0 * r_t_mm * 1e-3                         # diameter [m]
    tau = 50.0 + torch.log(D_m / 0.01) / 0.002134     # doubling time [day]
    tau = torch.clamp(tau, min=5.0)                   # guard for D<1cm
    return DOUBLING_C / tau                           # W/m³


def chest_wall_mask_from_pts(pts_mm, normals=None, frac=CHEST_WALL_FRAC, side=CHEST_WALL_SIDE):
    """Boolean mask: which surface vertices belong to the chest wall (body side).
    The chest wall is the internal face toward the body — NOT seen by the IR
    camera — anchored to arterial temperature (Dirichlet 37 °C).

    NORMAL-BASED (preferred): pass per-vertex `normals`; the chest wall is the
    posterior-facing surface (outward normal pointing to the body). With
    CHEST_WALL_SIDE='max' the anterior skin faces -Z, so chest wall = n_z > 0.
    This covers the whole back (a thin z-slab cannot, on a curved posterior).
    Falls back to a z-slab only if normals are unavailable (legacy PINN path)."""
    if normals is not None:
        nz = np.asarray(normals)[:, 2]
        nz = nz / (np.linalg.norm(np.asarray(normals), axis=1) + 1e-9)
        return nz > 0.0 if side == 'max' else nz < 0.0
    z = pts_mm[:, 2]
    z_lo, z_hi = float(z.min()), float(z.max())
    span = (z_hi - z_lo) + 1e-8
    if side == 'min':
        return z < (z_lo + frac * span)
    return z > (z_hi - frac * span)


class BioheatPINN(nn.Module):
    """6-layer Tanh MLP, (x,y,z)→T(°C), PLUS learnable tumour GEOMETRY.

    v3 — Mukhmetov (2025) formulation:  only the tumour CENTRE (x_t,y_t,z_t) and
    RADIUS (r_t) are trainable.  The heat MAGNITUDE is fixed by physiology
    (tumor_Qm_from_radius) — this removes the unidentifiable magnitude DOF that
    stalled every earlier version (Bezerra sensitivity analysis)."""
    def __init__(self, hidden=256, depth=6):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        # Trainable tumour GEOMETRY only (physical mm). Magnitude is NOT a param.
        self.x_t = nn.Parameter(torch.tensor([0.0]))   # mm
        self.y_t = nn.Parameter(torch.tensor([0.0]))   # mm
        self.z_t = nn.Parameter(torch.tensor([0.0]))   # mm
        self.r_t = nn.Parameter(torch.tensor([15.0]))  # mm

    def forward(self, xyz):
        return self.net(xyz).squeeze(-1)

    def Q_tumor(self, xyz_mm):
        """Gaussian tumour heat source in PHYSICAL mm. Peak density is fixed by
        the radius via tumor_Qm_from_radius (no free magnitude)."""
        r  = torch.clamp(self.r_t, min=5.0)
        Qd = tumor_Qm_from_radius(r)                   # W/m³  (physiology-fixed)
        d2 = ((xyz_mm[:, 0] - self.x_t) ** 2 +
              (xyz_mm[:, 1] - self.y_t) ** 2 +
              (xyz_mm[:, 2] - self.z_t) ** 2)
        return Q_TUMOR_SCALE * Qd * torch.exp(-d2 / (r ** 2 + 1e-8))


def normalise_coords(pts, bbox_min, bbox_max):
    centre = (bbox_max + bbox_min) / 2
    extent = (bbox_max - bbox_min) / 2 + 1e-8
    return (pts - centre) / extent

def compute_laplacian(T, xyz, extent_meters):
    """∇²T in physical (m⁻²) units. xyz is the normalised network input;
    chain rule divides each 2nd derivative by the per-axis half-extent²."""
    dT = torch.autograd.grad(T, xyz,
             grad_outputs=torch.ones_like(T), create_graph=True)[0]
    lap = sum(
        torch.autograd.grad(dT[:, i], xyz,
            grad_outputs=torch.ones_like(dT[:, i]),
            create_graph=True)[0][:, i] / (extent_meters[i] ** 2)
        for i in range(3)
    )
    return lap

def normal_flux(model, pts_norm, normals_unit, extent_meters):
    """(T, ∂T/∂n) at boundary points — for the convective Robin skin BC in the
    FORWARD solve. ∂T/∂n = (∇_x T)·n, with ∇_x T = ∇_ξ T / extent_meters."""
    sp = pts_norm.clone().detach().requires_grad_(True)
    T  = model(sp)
    g  = torch.autograd.grad(T, sp, grad_outputs=torch.ones_like(T),
                             create_graph=True)[0]
    g_phys = g / extent_meters
    dTdn   = (g_phys * normals_unit).sum(dim=1)
    return T, dTdn

def classify_quadrant(x_mm, y_mm):
    if x_mm > 0 and y_mm > 0: return 'Upper Outer'
    if x_mm < 0 and y_mm > 0: return 'Upper Inner'
    if x_mm > 0 and y_mm < 0: return 'Lower Outer'
    return 'Lower Inner'

print('PINN v3 (fixed-magnitude, geometry-only inverse) ready.')


def _fwd_loss_scales(model, cw_norm, skin_norm, skin_n, int_norm, int_pts, extent_m, n_cw):
    """One forward-loss evaluation → per-term step-0 normalisers."""
    L_cw = ((model(cw_norm) - T_ARTERIAL) ** 2).mean() if n_cw > 0 \
           else torch.tensor(1.0, device=int_pts.device)
    T_sk, dTdn = normal_flux(model, skin_norm, skin_n, extent_m)
    L_rob = ((-K_TISSUE * dTdn - H_CONV * (T_sk - T_AIR)) ** 2).mean()
    ii = int_norm.clone().detach().requires_grad_(True)
    Ti = model(ii); lap = compute_laplacian(Ti, ii, extent_m)
    Qt = model.Q_tumor(int_pts)
    L_pde = ((K_TISSUE * lap + PERFUSION * (T_ARTERIAL - Ti) + Q_METAB + Qt) ** 2).mean()
    return (L_cw.detach() + 1e-6, L_rob.detach() + 1e-6, L_pde.detach() + 1e-6)


def train_pinn_single(geo, device, n_starts=1,
                       depth_grid=6, radius_grid=5,
                       r_min=6.0, r_max=30.0,
                       fwd_steps_first=2500, fwd_steps_warm=800,
                       refine_iters=12, hot_pct=0.90, save_plot=True):
    """v3 — forward-matching inverse (per-candidate PINN forward solve).

    The PINN is a FORWARD bioheat solver: given a PRESCRIBED tumour source it
    solves   k∇²T + perf·(Ta−T) + Qm + Q_tumor = 0   with chest-wall Dirichlet
    37 °C and a convective Robin skin BC.  The skin temperature is PREDICTED
    (never pinned).  We then pick the tumour DEPTH and RADIUS whose predicted
    skin temperature best matches the measured IR; lateral position is pinned to
    the IR hot-spot.  Because the source is prescribed — not free to minimise a
    residual — it cannot escape to the boundary (the failure of every free-source
    formulation).  Magnitude is fixed by physiology (tumor_Qm_from_radius).

    Search = coarse grid over (depth, radius) then a local scipy (Nelder–Mead)
    refine.  A persistent MLP is warm-started across candidates for speed.
    """
    from scipy.optimize import minimize

    surf_pts = torch.tensor(geo['surface_pts'],   dtype=torch.float32, device=device)
    T_ir     = torch.tensor(geo['T_measured'],    dtype=torch.float32, device=device)
    normals  = torch.tensor(geo['vertex_normals'],dtype=torch.float32, device=device)
    int_pts  = torch.tensor(geo['interior_pts'],  dtype=torch.float32, device=device)
    bbox_min = torch.tensor(geo['bbox_min'],      dtype=torch.float32, device=device)
    bbox_max = torch.tensor(geo['bbox_max'],      dtype=torch.float32, device=device)

    extent_m  = ((bbox_max - bbox_min) / 2 + 1e-8) * 1e-3
    surf_norm = normalise_coords(surf_pts, bbox_min, bbox_max)
    int_norm  = normalise_coords(int_pts,  bbox_min, bbox_max)

    cw_np   = chest_wall_mask_from_pts(geo['surface_pts'])
    cw_mask = torch.tensor(cw_np, device=device); sk_mask = ~cw_mask
    cw_norm   = surf_norm[cw_mask]
    skin_norm = surf_norm[sk_mask]; skin_ir = T_ir[sk_mask]
    skin_n    = normals[sk_mask];   skin_pts = surf_pts[sk_mask]
    n_cw, n_sk = int(cw_mask.sum().item()), int(sk_mask.sum().item())
    print(f'  surface split: {n_cw} chest-wall / {n_sk} skin vertices')

    # ── Lateral pin from IR hot-spot; depth-search range along z ────────────
    thr     = torch.quantile(skin_ir, hot_pct); hot = skin_ir >= thr
    hot_xyz = skin_pts[hot]
    x_pin   = float(hot_xyz[:, 0].mean()); y_pin = float(hot_xyz[:, 1].mean())
    z_lo, z_hi = float(bbox_min[2]), float(bbox_max[2]); z_rng = z_hi - z_lo
    z_grid_lo, z_grid_hi = z_lo + 0.15 * z_rng, z_hi - 0.15 * z_rng
    print(f'  hot-spot pin: lateral=({x_pin:.1f}, {y_pin:.1f})mm  '
          f'({int(hot.sum())} verts, top {100*(1-hot_pct):.0f}%)')

    # ── Persistent forward-solver model (geometry FROZEN, prescribed) ───────
    model = BioheatPINN().to(device)
    for p in (model.x_t, model.y_t, model.z_t, model.r_t):
        p.requires_grad_(False)
    with torch.no_grad():
        model.x_t.fill_(x_pin); model.y_t.fill_(y_pin)
    scales = {'v': None}

    def forward_solve(z_mm, r_mm, steps):
        """Solve the forward BVP for a prescribed source; return predicted skin T."""
        with torch.no_grad():
            model.z_t.fill_(float(z_mm)); model.r_t.fill_(float(np.clip(r_mm, 5.0, 40.0)))
        if scales['v'] is None:
            scales['v'] = _fwd_loss_scales(model, cw_norm, skin_norm, skin_n,
                                           int_norm, int_pts, extent_m, n_cw)
        s_cw, s_rob, s_pde = scales['v']
        opt = torch.optim.Adam(model.net.parameters(), lr=1e-3)
        for step in range(steps):
            opt.zero_grad()
            L_cw = ((model(cw_norm) - T_ARTERIAL) ** 2).mean() if n_cw > 0 \
                   else torch.zeros((), device=device)
            T_sk, dTdn = normal_flux(model, skin_norm, skin_n, extent_m)
            L_rob = ((-K_TISSUE * dTdn - H_CONV * (T_sk - T_AIR)) ** 2).mean()
            ii = int_norm.clone().detach().requires_grad_(True)
            Ti = model(ii); lap = compute_laplacian(Ti, ii, extent_m)
            Qt = model.Q_tumor(int_pts)
            L_pde = ((K_TISSUE * lap + PERFUSION * (T_ARTERIAL - Ti)
                      + Q_METAB + Qt) ** 2).mean()
            loss = L_cw / s_cw + L_rob / s_rob + L_pde / s_pde
            loss.backward(); opt.step()
        with torch.no_grad():
            return model(skin_norm)

    # Inverse objective: predicted-vs-measured skin mismatch (pattern-centred,
    # so an offset from imperfect h/T_air does not dominate the depth/size fit).
    ir_c = skin_ir - skin_ir.mean()
    def objective(z_mm, r_mm, steps):
        pred = forward_solve(z_mm, r_mm, steps)
        pred_c = pred - pred.mean()
        return float(((pred_c - ir_c) ** 2).mean())

    # ── Coarse grid (first solve cold, rest warm-started) ───────────────────
    z_vals = np.linspace(z_grid_lo, z_grid_hi, depth_grid)
    r_vals = np.linspace(r_min, r_max, radius_grid)
    cost = np.full((depth_grid, radius_grid), np.nan)
    best = {'J': np.inf}
    first = True
    for iz, zz in enumerate(z_vals):
        for ir, rr in enumerate(r_vals):
            steps = fwd_steps_first if first else fwd_steps_warm
            J = objective(zz, rr, steps); first = False
            cost[iz, ir] = J
            if J < best['J']:
                best = {'J': J, 'z': float(zz), 'r': float(rr),
                        'state': {k: v.cpu().clone() for k, v in model.state_dict().items()}}
            print(f'  grid z={zz:6.1f} r={rr:5.1f} → J={J:.4f}'
                  f'{"  ★" if J == best["J"] else ""}')

    # ── Local refine (scipy Nelder–Mead, warm-started) ──────────────────────
    print('  refining (Nelder–Mead)…')
    def neg(p):
        z, r = p
        r = float(np.clip(r, 5.0, 40.0)); z = float(np.clip(z, z_lo, z_hi))
        return objective(z, r, fwd_steps_warm)
    res = minimize(neg, x0=[best['z'], best['r']], method='Nelder-Mead',
                   options={'maxiter': refine_iters, 'xatol': 1.0, 'fatol': 1e-4})
    z_fin = float(np.clip(res.x[0], z_lo, z_hi))
    r_fin = float(np.clip(res.x[1], 5.0, 40.0))
    J_fin = objective(z_fin, r_fin, fwd_steps_first)   # final clean solve
    if J_fin <= best['J']:
        best.update({'J': J_fin, 'z': z_fin, 'r': r_fin,
                     'state': {k: v.cpu().clone() for k, v in model.state_dict().items()}})
    print(f'  refined → z={best["z"]:.1f}mm r={best["r"]:.1f}mm J={best["J"]:.4f}')

    # ── Assemble results ────────────────────────────────────────────────────
    x_mm, y_mm, z_mm, r_mm = x_pin, y_pin, best['z'], best['r']
    Q_fixed  = float(tumor_Qm_from_radius(torch.tensor(r_mm)).item())
    centre   = torch.tensor([x_mm, y_mm, z_mm])
    depth_mm = float((skin_pts.cpu() - centre).norm(dim=1).min().item())

    if save_plot:
        pid = geo['patient_id']
        fig, ax = plt.subplots(figsize=(6.5, 5))
        im = ax.imshow(cost, origin='lower', aspect='auto', cmap='viridis_r',
                       extent=[r_vals[0], r_vals[-1], z_vals[0], z_vals[-1]])
        ax.scatter([r_mm], [z_mm], c='red', marker='*', s=180, label='best')
        ax.set_xlabel('radius r_t (mm)'); ax.set_ylabel('depth coord z_t (mm)')
        ax.set_title(f'Forward-match cost J(z,r) — {pid}')
        fig.colorbar(im, label='skin pattern mismatch'); ax.legend()
        plt.tight_layout()
        plt.savefig(str(RESULTS_DIR / f'{pid}_loss_convergence.png'), dpi=150)
        plt.close(fig)

    results = {
        'patient_id': geo['patient_id'], 'label': geo['label'],
        'x_t_mm': x_mm, 'y_t_mm': y_mm, 'z_t_mm': z_mm,
        'r_t_mm': r_mm, 'Q_max': Q_fixed, 'depth_mm': depth_mm,
        'volume_mm3': (4.0 / 3.0) * np.pi * r_mm ** 3,
        'quadrant': classify_quadrant(x_mm, y_mm),
        'match_cost': best['J'], 'seed': 0,
    }
    print(f"  → lateral=({x_mm:.1f},{y_mm:.1f})mm[pinned]  z={z_mm:.1f}mm  "
          f"r={r_mm:.1f}mm  depth={depth_mm:.1f}mm  Q={Q_fixed:.0f}W/m³  {results['quadrant']}")
    return results, best['state']

print('train_pinn_single (v3 — forward-matching inverse) ready.')


def generate_synthetic_skin(geo, device, z_true_mm, r_true_mm,
                             x_true_mm=None, y_true_mm=None,
                             noise_pct=0.0, steps=2500, hot_pct=0.90, seed=0):
    """Forward-generate a self-consistent synthetic surface temperature for a
    KNOWN tumour, using the SAME forward BVP the inverse solves (chest-wall
    Dirichlet 37 °C + convective Robin skin + interior Pennes, magnitude fixed by
    tumor_Qm_from_radius).  Returns (T_full, planted): a per-surface-vertex
    temperature array to drop into geo['T_measured'], and the ground-truth dict.

    For the synthetic-recovery validation: the data is generated by the forward
    operator, so a correct inverse must recover the planted depth & radius.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    surf_pts = torch.tensor(geo['surface_pts'],   dtype=torch.float32, device=device)
    T_ir     = torch.tensor(geo['T_measured'],    dtype=torch.float32, device=device)
    normals  = torch.tensor(geo['vertex_normals'],dtype=torch.float32, device=device)
    int_pts  = torch.tensor(geo['interior_pts'],  dtype=torch.float32, device=device)
    bbox_min = torch.tensor(geo['bbox_min'],      dtype=torch.float32, device=device)
    bbox_max = torch.tensor(geo['bbox_max'],      dtype=torch.float32, device=device)
    extent_m  = ((bbox_max - bbox_min) / 2 + 1e-8) * 1e-3
    surf_norm = normalise_coords(surf_pts, bbox_min, bbox_max)
    int_norm  = normalise_coords(int_pts,  bbox_min, bbox_max)

    cw_np   = chest_wall_mask_from_pts(geo['surface_pts'])
    cw_mask = torch.tensor(cw_np, device=device); sk_mask = ~cw_mask
    cw_norm   = surf_norm[cw_mask]; skin_norm = surf_norm[sk_mask]
    skin_n    = normals[sk_mask];   skin_pts  = surf_pts[sk_mask]; skin_ir = T_ir[sk_mask]
    n_cw = int(cw_mask.sum().item())

    # default planted lateral = the real IR hot-spot centroid (anatomically plausible)
    if x_true_mm is None or y_true_mm is None:
        thr = torch.quantile(skin_ir, hot_pct); hot = skin_ir >= thr
        x_true_mm = float(skin_pts[hot][:, 0].mean())
        y_true_mm = float(skin_pts[hot][:, 1].mean())

    # prescribe the tumour and solve the forward BVP (identical to forward_solve)
    model = BioheatPINN().to(device)
    for p in (model.x_t, model.y_t, model.z_t, model.r_t):
        p.requires_grad_(False)
    with torch.no_grad():
        model.x_t.fill_(x_true_mm); model.y_t.fill_(y_true_mm)
        model.z_t.fill_(float(z_true_mm))
        model.r_t.fill_(float(np.clip(r_true_mm, 5.0, 40.0)))
    s_cw, s_rob, s_pde = _fwd_loss_scales(model, cw_norm, skin_norm, skin_n,
                                          int_norm, int_pts, extent_m, n_cw)
    opt = torch.optim.Adam(model.net.parameters(), lr=1e-3)
    for step in range(steps):
        opt.zero_grad()
        L_cw = ((model(cw_norm) - T_ARTERIAL) ** 2).mean() if n_cw > 0 \
               else torch.zeros((), device=device)
        T_sk, dTdn = normal_flux(model, skin_norm, skin_n, extent_m)
        L_rob = ((-K_TISSUE * dTdn - H_CONV * (T_sk - T_AIR)) ** 2).mean()
        ii = int_norm.clone().detach().requires_grad_(True)
        Ti = model(ii); lap = compute_laplacian(Ti, ii, extent_m)
        Qt = model.Q_tumor(int_pts)
        L_pde = ((K_TISSUE * lap + PERFUSION * (T_ARTERIAL - Ti)
                  + Q_METAB + Qt) ** 2).mean()
        (L_cw / s_cw + L_rob / s_rob + L_pde / s_pde).backward(); opt.step()

    with torch.no_grad():
        T_full = model(surf_norm).cpu().numpy().astype(np.float32)

    # optional measurement noise on the skin (Bezerra-style robustness study)
    if noise_pct > 0:
        rng  = np.random.default_rng(seed)
        span = float(T_full.max() - T_full.min())
        sk   = sk_mask.cpu().numpy()
        T_full[sk] += rng.normal(0.0, noise_pct * span, size=int(sk.sum())).astype(np.float32)

    centre  = torch.tensor([x_true_mm, y_true_mm, float(z_true_mm)])
    planted = {'x_t_mm': x_true_mm, 'y_t_mm': y_true_mm, 'z_t_mm': float(z_true_mm),
               'r_t_mm': float(r_true_mm),
               'depth_mm': float((skin_pts.cpu() - centre).norm(dim=1).min().item()),
               'noise_pct': noise_pct}
    print(f"  planted: lateral=({x_true_mm:.1f},{y_true_mm:.1f}) "
          f"z={z_true_mm:.1f} r={r_true_mm:.1f} depth={planted['depth_mm']:.1f} "
          f"noise={noise_pct*100:.0f}%  skin T∈[{T_full[sk_mask.cpu().numpy()].min():.1f},"
          f"{T_full[sk_mask.cpu().numpy()].max():.1f}]°C")
    return T_full, planted

print('generate_synthetic_skin ready.')


def stl_to_tet_mesh(stl_path, out_msh_path, mesh_size_mm=3.0):
    import gmsh, numpy as _np
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)
    try:
        gmsh.model.add('breast')
        gmsh.merge(str(stl_path))
        # classifySurfaces turns raw STL triangles into CAD surface entities;
        # without it the geo kernel has no surfaces to loop over.
        gmsh.model.mesh.classifySurfaces(_np.pi, True, True, 2 * _np.pi)
        gmsh.model.mesh.createGeometry()
        s = gmsh.model.getEntities(2)
        l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
        gmsh.model.geo.addVolume([l])
        gmsh.model.geo.synchronize()
        v_ents = gmsh.model.getEntities(3)
        s_ents = gmsh.model.getEntities(2)
        gmsh.model.addPhysicalGroup(3, [e[1] for e in v_ents], 1)
        gmsh.model.setPhysicalName(3, 1, 'breast_volume')
        gmsh.model.addPhysicalGroup(2, [e[1] for e in s_ents], 2)
        gmsh.model.setPhysicalName(2, 2, 'breast_surface')
        gmsh.option.setNumber('Mesh.CharacteristicLengthMax', mesh_size_mm)
        gmsh.option.setNumber('Mesh.CharacteristicLengthMin', mesh_size_mm * 0.5)
        gmsh.option.setNumber('Mesh.Algorithm3D', 1)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize('Netgen')
        gmsh.write(str(out_msh_path))
    finally:
        gmsh.finalize()

def run_fea_forward(msh_path, pinn_results, geo=None):
    """Independent FEniCSx forward solve of the SAME BVP the PINN now solves:
    chest-wall Dirichlet 37 °C + Robin convective skin + Pennes interior.
    Uses the shared global constants so PINN ≡ FEA (no IR data enters here).

    If `geo` (with 'surface_pts','vertex_normals') is given, the chest wall is
    the NORMAL-BASED posterior-facing surface (covers the whole back); otherwise
    a legacy z-slab is used."""
    from mpi4py import MPI
    import dolfinx, dolfinx.mesh
    from dolfinx.io import gmsh as gmshio   # dolfinx 0.10 path
    from dolfinx import fem
    from dolfinx.fem.petsc import LinearProblem
    import ufl, numpy as _np

    fea_data = gmshio.read_from_msh(str(msh_path), MPI.COMM_WORLD, gdim=3)
    msh = fea_data.mesh                      # MeshData NamedTuple
    msh.geometry.x[:] *= 1e-3                # mm → m

    V = fem.functionspace(msh, ('Lagrange', 1))
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    x_t  = pinn_results['x_t_mm'] * 1e-3
    y_t  = pinn_results['y_t_mm'] * 1e-3
    z_t  = pinn_results['z_t_mm'] * 1e-3
    r_t  = pinn_results['r_t_mm'] * 1e-3
    Qmax = pinn_results['Q_max']

    d2      = (x[0]-x_t)**2 + (x[1]-y_t)**2 + (x[2]-z_t)**2
    Q_tumor = Qmax * ufl.exp(-d2 / (r_t**2 + 1e-16))

    # shared constants (defined in the PINN cell) — keeps PINN and FEA identical
    k, perf = K_TISSUE, PERFUSION
    Ta, Qm  = T_ARTERIAL, Q_METAB
    h, T_air = H_CONV, T_AIR

    fdim = msh.topology.dim - 1
    if geo is not None:
        # NORMAL-BASED chest wall: posterior-facing exterior facets = body side.
        from scipy.spatial import cKDTree
        msh.topology.create_connectivity(fdim, msh.topology.dim)
        ext = dolfinx.mesh.exterior_facet_indices(msh.topology)
        mid = dolfinx.mesh.compute_midpoints(msh, fdim, ext)          # metres
        nrm = _np.asarray(geo['vertex_normals'], dtype=float)
        nrm = nrm / (_np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
        _, gi = cKDTree(_np.asarray(geo['surface_pts'], dtype=float) * 1e-3).query(mid)
        post = (nrm[gi, 2] > 0.0) if CHEST_WALL_SIDE == 'max' else (nrm[gi, 2] < 0.0)
        facets = ext[post]
    else:
        # legacy z-slab
        zc = msh.geometry.x[:, 2]
        z_lo, z_hi = float(_np.min(zc)), float(_np.max(zc))
        span = (z_hi - z_lo) + 1e-12
        if CHEST_WALL_SIDE == 'min':
            cut = z_lo + CHEST_WALL_FRAC * span
            chest_wall = lambda pt: pt[2] < cut
        else:
            cut = z_hi - CHEST_WALL_FRAC * span
            chest_wall = lambda pt: pt[2] > cut
        facets = dolfinx.mesh.locate_entities_boundary(msh, fdim, chest_wall)
    dofs   = fem.locate_dofs_topological(V, fdim, facets)
    bc     = fem.dirichletbc(_np.float64(Ta), dofs, V)   # np scalar in 0.10

    a = (k*ufl.inner(ufl.grad(T), ufl.grad(v)) + perf*T*v)*ufl.dx + h*T*v*ufl.ds
    L = (perf*Ta + Qm + Q_tumor)*v*ufl.dx + h*T_air*v*ufl.ds

    try:
        prob = LinearProblem(a, L, bcs=[bc],
                     petsc_options_prefix='bioheat_',
                     petsc_options={'ksp_type':'cg', 'pc_type':'gamg'})
    except TypeError:
        prob = LinearProblem(a, L, bcs=[bc],
                     petsc_options={'ksp_type':'cg', 'pc_type':'gamg'})
    return prob.solve(), msh

def compute_fea_residual(T_fea_sol, msh, surface_pts_mm, T_measured):
    fea_coords = msh.geometry.x * 1e3
    fea_vals   = T_fea_sol.x.array
    _, idx     = cKDTree(fea_coords).query(surface_pts_mm, k=3)
    T_interp   = fea_vals[idx].mean(axis=1)
    residuals  = np.abs(T_interp - T_measured)
    return residuals, residuals.mean(), residuals.max(), T_interp

def plot_fea_vs_pinn(geo, pinn_results, T_fea_interp, save_html=True):
    """§10.8 — IR-measured (PINN target) | FEA forward | |FEA−IR| residual."""
    from plotly.subplots import make_subplots
    verts    = geo['surface_pts']
    faces    = geo['faces']
    T_meas   = geo['T_measured']
    T_fea    = np.array(T_fea_interp, dtype=np.float32)
    residual = np.abs(T_fea - T_meas)
    pid      = geo['patient_id']
    fea_res  = pinn_results.get('fea_mean_residual', float('nan'))

    mesh_kw = dict(x=verts[:,0], y=verts[:,1], z=verts[:,2],
                   i=faces[:,0], j=faces[:,1], k=faces[:,2],
                   intensitymode='vertex', flatshading=False, opacity=1.0,
                   lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2))
    t_lo, t_hi = float(T_meas.min()), float(T_meas.max())

    fig = make_subplots(rows=1, cols=3,
                        specs=[[{'type':'mesh3d'}]*3],
                        subplot_titles=['IR-Measured (PINN target)',
                                        'FEA Forward solve',
                                        '|FEA − IR| Residual'])
    fig.add_trace(go.Mesh3d(**mesh_kw, intensity=T_meas,
                             colorscale='Inferno', cmin=t_lo, cmax=t_hi,
                             colorbar=dict(title='°C', x=0.30, len=0.75, thickness=12)),
                  row=1, col=1)
    fig.add_trace(go.Mesh3d(**mesh_kw, intensity=T_fea,
                             colorscale='Inferno', cmin=t_lo, cmax=t_hi,
                             colorbar=dict(title='°C', x=0.64, len=0.75, thickness=12)),
                  row=1, col=2)
    fig.add_trace(go.Mesh3d(**mesh_kw, intensity=residual,
                             colorscale='Hot', cmin=0, cmax=float(residual.max()),
                             colorbar=dict(title='Δ°C', x=1.00, len=0.75, thickness=12)),
                  row=1, col=3)
    fig.update_layout(
        title=dict(
            text=(f'FEA vs IR — {pid} ({geo["label"]})<br>'
                  f'<sup>mean|FEA−IR|={fea_res:.2f}°C | '
                  f'{"OK ✓" if fea_res < 1.5 else "HIGH RESIDUAL ✗"}</sup>'),
            font=dict(size=13)),
        height=520, width=1500,
        margin=dict(l=0, r=0, b=0, t=70),
    )
    if save_html:
        html_path = RESULTS_DIR / f'{pid}_fea_vs_ir.html'
        fig.write_html(str(html_path), include_plotlyjs='cdn')
        print(f'FEA-vs-IR plot saved → {html_path}')
    return fig

print('FEA functions (shared-constant, chest-wall-aligned) ready.')


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestration — tmux-friendly cohort / synthetic runner
#  (appended to the verbatim notebook definitions above)
# ═══════════════════════════════════════════════════════════════════════════

# Geometry / optimisation settings — identical to the notebook's tunable cell.
GAUSSIAN_SIGMA  = 3.0    # smoothing for sigma_grid
MC_THRESHOLD    = None   # None = use CFG['mc_threshold']=0.3
REPAIR_HOLES    = True   # trimesh hole fill + normal fix
LAPLACIAN_ITERS = 20     # Laplacian mesh smoothing iterations
THERMAL_SIGMA   = 0.0    # light T_grid smoothing (0 = off)

# Diagnostic knob: multiplies the tumour heat source. 1.0 = production (UNCHANGED);
# --diagnose temporarily raises it. Read at call-time by BioheatPINN.Q_tumor.
Q_TUMOR_SCALE   = 1.0

# Canonical column order for the cohort CSV (kept stable for resume/append).
COLUMNS = ['patient_id', 'label', 'x_t_mm', 'y_t_mm', 'z_t_mm', 'r_t_mm',
           'Q_max', 'depth_mm', 'volume_mm3', 'quadrant', 'match_cost', 'seed',
           'fea_mean_residual', 'fea_max_residual', 'fea_status']


def build_dataset():
    pats = discover_patients(TIFF_DIR, UNET_DIR)
    return BreastThermDataset(pats, CFG), pats


def load_models():
    L_       = CFG['pos_enc_L']
    pos_dim  = 3 * 2 * L_
    feat_dim = CFG['feat_channels'] * 2
    enc = SiameseEncoder(out_channels=CFG['feat_channels']).to(device).eval()
    m   = ThermamNeRFMLP(pos_enc_dim=pos_dim, feat_dim=feat_dim,
                         hidden=CFG['mlp_hidden'],
                         n_layers=CFG['mlp_layers']).to(device).eval()
    ck = torch.load(NERF_CKPT, map_location=device)
    enc.load_state_dict({k.replace('module.', ''): v for k, v in ck['encoder'].items()})
    m.load_state_dict(  {k.replace('module.', ''): v for k, v in ck['mlp'].items()})
    n_params = sum(p.numel() for p in list(enc.parameters()) + list(m.parameters()))
    print(f'NeRF checkpoint loaded — {n_params:,} parameters')
    return enc, m


def pinn_recover_dirichlet(geo, T_target, x0, y0, z_init=None, r_init=15.0,
                           adam_steps=5000, n_colloc=5000,
                           lr_net=1e-3, lr_src=1e-1, verbose=True,
                           resample_every=200, tumor_bias=0.70):
    """Mukhmetov-style inverse: skin pinned as Dirichlet (= T_target), chest wall
    pinned 37 C, and (z_t, r_t) trained so the interior Pennes residual is
    consistent. ONE optimization, no grid search → fast. Magnitude fixed by radius.

    Key fix vs naive approach: collocation points are resampled every `resample_every`
    steps with `tumor_bias` fraction concentrated within 4×r_t of the current tumor
    estimate. Without this, the Gaussian source has near-zero gradient at random points
    spread over a 300 mm domain → z_t gradient ≈ 0 → optimizer can't steer depth.

    Returns (z_mm, r_mm, state_dict, final_skin_loss)."""
    surf_pts  = torch.tensor(geo['surface_pts'],  dtype=torch.float32, device=device)
    T         = torch.tensor(T_target,            dtype=torch.float32, device=device)
    all_int   = torch.tensor(geo['interior_pts'], dtype=torch.float32, device=device)
    bbox_min  = torch.tensor(geo['bbox_min'],     dtype=torch.float32, device=device)
    bbox_max  = torch.tensor(geo['bbox_max'],     dtype=torch.float32, device=device)
    extent_m  = ((bbox_max - bbox_min) / 2 + 1e-8) * 1e-3
    surf_norm = normalise_coords(surf_pts, bbox_min, bbox_max)

    cw_np   = chest_wall_mask_from_pts(geo['surface_pts'])
    cw_mask = torch.tensor(cw_np, device=device); sk_mask = ~cw_mask
    cw_norm   = surf_norm[cw_mask]
    skin_norm = surf_norm[sk_mask]; skin_tgt = T[sk_mask]
    n_cw = int(cw_mask.sum().item())
    z_lo, z_hi = float(bbox_min[2]), float(bbox_max[2])
    if z_init is None:
        z_init = 0.5 * (z_lo + z_hi)

    n_uniform = max(1, int(n_colloc * (1.0 - tumor_bias)))
    n_local   = n_colloc - n_uniform

    def sample_colloc(z_c, r_c):
        """70% near tumor, 30% uniform — ensures non-zero dQt/dz_t gradient."""
        sel_u = torch.randperm(all_int.shape[0], device=device)[:n_uniform]
        pts_u = all_int[sel_u]
        # biased: Gaussian draw around current estimate, clipped to mesh bbox
        sigma = float(r_c) * 4.0
        centre = torch.tensor([[x0, y0, float(z_c)]], dtype=torch.float32, device=device)
        noise  = torch.randn(n_local * 4, 3, device=device) * sigma + centre
        # keep only points inside bbox
        inside = ((noise >= bbox_min) & (noise <= bbox_max)).all(dim=1)
        noise  = noise[inside]
        if noise.shape[0] < n_local:
            # fallback: pad with uniform samples
            extra = all_int[torch.randperm(all_int.shape[0], device=device)[:n_local]]
            noise = torch.cat([noise, extra], dim=0)
        pts_l  = noise[:n_local]
        pts    = torch.cat([pts_u, pts_l], dim=0)
        nrm    = normalise_coords(pts, bbox_min, bbox_max)
        return pts, nrm

    model = BioheatPINN().to(device)
    model.x_t.requires_grad_(False); model.y_t.requires_grad_(False)
    model.z_t.requires_grad_(True);  model.r_t.requires_grad_(True)
    with torch.no_grad():
        model.x_t.fill_(x0); model.y_t.fill_(y0)
        model.z_t.fill_(float(z_init)); model.r_t.fill_(float(r_init))

    int_pts, int_norm = sample_colloc(z_init, r_init)

    def residuals(i_pts, i_norm):
        L_skin = ((model(skin_norm) - skin_tgt) ** 2).mean()
        L_cw   = ((model(cw_norm) - T_ARTERIAL) ** 2).mean() if n_cw > 0 \
                 else torch.zeros((), device=device)
        ii = i_norm.clone().detach().requires_grad_(True)
        Ti = model(ii); lap = compute_laplacian(Ti, ii, extent_m)
        Qt = model.Q_tumor(i_pts)
        L_pde = ((K_TISSUE * lap + PERFUSION * (T_ARTERIAL - Ti)
                  + Q_METAB + Qt) ** 2).mean()
        return L_skin, L_cw, L_pde

    # ── Phase 1: fit BCs only (freeze source params) ───────────────────────────
    # Drive skin+CW loss to near-zero so the network represents the correct
    # temperature field BEFORE we try to identify the source parameters.
    model.z_t.requires_grad_(False); model.r_t.requires_grad_(False)
    opt1 = torch.optim.Adam(model.net.parameters(), lr=lr_net)
    steps_p1 = max(1000, adam_steps // 4)
    if verbose:
        print(f'    [phase 1] fitting BCs for {steps_p1} steps …', flush=True)
    for step in range(steps_p1):
        opt1.zero_grad()
        L_skin = ((model(skin_norm) - skin_tgt) ** 2).mean()
        L_cw   = ((model(cw_norm) - T_ARTERIAL) ** 2).mean() if n_cw > 0 \
                 else torch.zeros((), device=device)
        (L_skin + L_cw).backward(); opt1.step()
        if verbose and step % (steps_p1 // 4) == 0:
            print(f'      p1 step {step:4d}  skin={float(L_skin):.4f} cw={float(L_cw):.4f}',
                  flush=True)
    last_skin = float(L_skin)
    if verbose:
        print(f'    [phase 1] done  skin={last_skin:.4f}', flush=True)

    # ── Phase 2: source identification (freeze network) ──────────────────────
    # With the temperature field fixed, minimise PDE residual over (z_t, r_t).
    # The PDE residual is now a well-posed least-squares problem: find the
    # Gaussian source that makes k∇²T + ω(Ta-T) + Qm + Qt ≈ 0 everywhere.
    for p in model.net.parameters():
        p.requires_grad_(False)
    model.z_t.requires_grad_(True); model.r_t.requires_grad_(True)
    opt2 = torch.optim.Adam([model.z_t, model.r_t], lr=lr_src)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=adam_steps, eta_min=1e-3)
    steps_p2 = adam_steps
    if verbose:
        print(f'    [phase 2] source ID for {steps_p2} steps …', flush=True)
    for step in range(steps_p2):
        if step > 0 and step % resample_every == 0:
            with torch.no_grad():
                int_pts, int_norm = sample_colloc(
                    model.z_t.item(), model.r_t.item())
        opt2.zero_grad()
        ii = int_norm.clone().detach().requires_grad_(True)
        Ti = model(ii); lap = compute_laplacian(Ti, ii, extent_m)
        Qt = model.Q_tumor(int_pts)
        L_pde = ((K_TISSUE * lap + PERFUSION * (T_ARTERIAL - Ti)
                  + Q_METAB + Qt) ** 2).mean()
        L_pde.backward(); opt2.step(); sched2.step()
        with torch.no_grad():
            model.r_t.clamp_(5.0, 40.0); model.z_t.clamp_(z_lo, z_hi)
        if verbose and step % 500 == 0:
            print(f'    p2 step {step:5d}  pde={float(L_pde):.2f}'
                  f'  z={model.z_t.item():.1f} r={model.r_t.item():.1f}', flush=True)

    state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return float(model.z_t.item()), float(model.r_t.item()), state, last_skin


def train_pinn_mukhmetov(geo, device, adam_steps=5000, hot_pct=0.90):
    """Mukhmetov-style Dirichlet-skin inverse on the patient's real surface T.
    Lateral pinned to the IR hot-spot; depth & radius recovered in one optimization."""
    sp = geo['surface_pts']; Tm = geo['T_measured']
    sk = ~chest_wall_mask_from_pts(sp)
    thr = np.quantile(Tm[sk], hot_pct); hot = sk & (Tm >= thr)
    x0, y0 = float(sp[hot, 0].mean()), float(sp[hot, 1].mean())
    z_lo, z_hi = float(geo['bbox_min'][2]), float(geo['bbox_max'][2])
    print(f'  [mukhmetov] hot-spot lateral=({x0:.1f},{y0:.1f}); recovering depth+radius…',
          flush=True)
    z, r, state, match = pinn_recover_dirichlet(
        geo, geo['T_measured'], x0, y0, z_init=0.5 * (z_lo + z_hi), adam_steps=adam_steps)
    Q = float(tumor_Qm_from_radius(torch.tensor(r)).item())
    centre = torch.tensor([x0, y0, z])
    depth_mm = float((torch.tensor(sp[sk]) - centre).norm(dim=1).min().item())
    results = {
        'patient_id': geo['patient_id'], 'label': geo['label'],
        'x_t_mm': x0, 'y_t_mm': y0, 'z_t_mm': z, 'r_t_mm': r, 'Q_max': Q,
        'depth_mm': depth_mm, 'volume_mm3': (4.0 / 3.0) * np.pi * r ** 3,
        'quadrant': classify_quadrant(x0, y0), 'match_cost': match, 'seed': 0,
    }
    print(f"  → lateral=({x0:.1f},{y0:.1f})[pinned]  z={z:.1f}mm  r={r:.1f}mm  "
          f"depth={depth_mm:.1f}mm  Q={Q:.0f}W/m³  {results['quadrant']}", flush=True)
    return results, state


def process_one(dataset, idx, encoder, mlp, do_fea=True, method='forward_match'):
    """Full per-patient pipeline (mirrors the notebook cohort body) → results dict."""
    item = dataset[idx]
    pid, lab = item['patient_id'], item['label']
    patient_dir = RESULTS_DIR / pid
    patient_dir.mkdir(exist_ok=True, parents=True)

    geo = load_patient_geo(
        dataset, idx, encoder, mlp, CFG, device,
        gaussian_sigma=GAUSSIAN_SIGMA, mc_threshold_override=MC_THRESHOLD,
        repair_holes=REPAIR_HOLES, laplacian_iters=LAPLACIAN_ITERS,
        thermal_sigma=THERMAL_SIGMA)

    stl_path = patient_dir / f'{pid}.stl'
    save_stl_binary(str(stl_path), geo['surface_pts'], geo['faces'])
    np.save(str(patient_dir / f'{pid}_T_measured.npy'), geo['T_measured'])
    np.save(str(patient_dir / f'{pid}_surf_pts.npy'),   geo['surface_pts'])

    if method == 'mukhmetov':
        pinn_results, pinn_state = train_pinn_mukhmetov(geo, device)
    else:
        pinn_results, pinn_state = train_pinn_single(geo, device)
    torch.save(pinn_state, str(patient_dir / f'{pid}_pinn.pth'))

    if do_fea:
        msh_path = patient_dir / f'{pid}.msh'
        try:
            stl_to_tet_mesh(str(stl_path), str(msh_path), mesh_size_mm=3.0)
            T_sol, fea_msh = run_fea_forward(str(msh_path), pinn_results)
            res, mean_res, max_res, T_fea_interp = compute_fea_residual(
                T_sol, fea_msh, geo['surface_pts'], geo['T_measured'])
            np.save(str(patient_dir / f'{pid}_fea_residuals.npy'), res)
            np.save(str(patient_dir / f'{pid}_T_fea.npy'), T_fea_interp)
            pinn_results['fea_mean_residual'] = mean_res
            pinn_results['fea_max_residual']  = max_res
            pinn_results['fea_status'] = 'OK' if mean_res < 1.5 else 'HIGH_RESIDUAL'
            print(f'  FEA: {mean_res:.3f}°C → {pinn_results["fea_status"]}')
            plot_fea_vs_pinn(geo, pinn_results, T_fea_interp, save_html=True)
        except Exception as e:
            print(f'  ⚠ FEA failed: {e}')
            pinn_results.update({'fea_mean_residual': np.nan,
                                 'fea_max_residual': np.nan,
                                 'fea_status': f'FAILED: {e}'})
    else:
        pinn_results.update({'fea_mean_residual': np.nan,
                             'fea_max_residual': np.nan,
                             'fea_status': 'SKIPPED'})
    return pinn_results


def append_row(csv_path, row):
    """Append one result row to the CSV immediately (crash-safe incremental write)."""
    df = pd.DataFrame([row]).reindex(columns=COLUMNS)
    df.to_csv(csv_path, mode='a', header=not Path(csv_path).exists(), index=False)


def done_ids(csv_path):
    """patient_ids already present in the CSV with a real result (for --resume)."""
    if not Path(csv_path).exists():
        return set()
    try:
        d = pd.read_csv(csv_path)
        if 'patient_id' not in d.columns:
            return set()
        if 'Q_max' in d.columns:
            d = d[d['Q_max'].notna()]
        return set(d['patient_id'].astype(str))
    except Exception:
        return set()


def select_indices(patients, args):
    n = len(patients)
    if args.indices:
        return [int(i) for i in args.indices.split(',') if i.strip() != '']
    if args.patients:
        want = set(s.strip() for s in args.patients.split(','))
        return [i for i, p in enumerate(patients) if p['id'] in want]
    if args.subset:
        from collections import defaultdict
        groups = defaultdict(list)
        for i, p in enumerate(patients):
            groups[p['label']].append(i)
        glists = list(groups.values())
        order, pos = [], 0
        while len(order) < args.subset:
            progressed = False
            for g in glists:
                if pos < len(g):
                    order.append(g[pos]); progressed = True
                    if len(order) >= args.subset:
                        break
            if not progressed:
                break
            pos += 1
        return sorted(order)
    if args.limit:
        return list(range(min(args.limit, n)))
    return list(range(n))


def run_diagnose(args):
    """Bug-vs-physics diagnostics for the inverse.
      TEST 2 — how many °C does the tumour actually move the skin? (same seed, on vs off)
      TEST 1 — does depth/radius recover when the source is 100x stronger?
    Uses the global Q_TUMOR_SCALE knob; production runs are untouched (scale stays 1.0)."""
    global Q_TUMOR_SCALE
    dataset, patients = build_dataset()
    encoder, mlp = load_models()
    idx = args.syn_idx
    geo = load_patient_geo(
        dataset, idx, encoder, mlp, CFG, device,
        gaussian_sigma=GAUSSIAN_SIGMA, mc_threshold_override=MC_THRESHOLD,
        repair_holes=REPAIR_HOLES, laplacian_iters=LAPLACIAN_ITERS,
        thermal_sigma=THERMAL_SIGMA)
    zlo, zhi = float(geo['bbox_min'][2]), float(geo['bbox_max'][2])
    z_plant = zlo + 0.50 * (zhi - zlo)
    r_plant = 15.0
    sk = ~chest_wall_mask_from_pts(geo['surface_pts'])   # skin-vertex mask

    # ── TEST 2: tumour on/off skin sensitivity (same seed → controlled comparison) ──
    if args.test in ('2', 'both'):
        print('\n' + '=' * 62)
        print('TEST 2 — tumour ON vs OFF: how many °C does the source move the skin?')
        print('=' * 62, flush=True)
        Q_TUMOR_SCALE = 1.0
        T_on,  _ = generate_synthetic_skin(geo, device, z_plant, r_plant, seed=0)
        Q_TUMOR_SCALE = 0.0
        T_off, _ = generate_synthetic_skin(geo, device, z_plant, r_plant, seed=0)
        Q_TUMOR_SCALE = 1.0
        dT = np.abs(T_on[sk] - T_off[sk])
        print(f'\n  tumour effect on skin:  max={dT.max():.4f} °C   mean={dT.mean():.4f} °C')
        print(f'  (skin field itself spans {T_on[sk].min():.1f}–{T_on[sk].max():.1f} °C)')
        print('  READ →  max ~ 0        : source not coupling at all        → BUG')
        print('          max << 0.1 °C  : real signal is tiny               → physics-limited (Bezerra)')
        print('          max several °C : strong signal but flat surface    → objective/search issue',
              flush=True)

    # ── TEST 1: ×100 source recovery ──
    if args.test in ('1', 'both'):
        print('\n' + '=' * 62)
        print('TEST 1 — x100 SOURCE: does depth/radius recover when the signal is huge?')
        print('=' * 62, flush=True)
        Q_TUMOR_SCALE = 100.0
        T_syn, planted = generate_synthetic_skin(geo, device, z_plant, r_plant, seed=0)
        geo_syn = dict(geo); geo_syn['T_measured'] = T_syn
        res, _ = train_pinn_single(geo_syn, device, save_plot=False)
        Q_TUMOR_SCALE = 1.0
        dz = abs(res['z_t_mm'] - planted['z_t_mm'])
        dr = abs(res['r_t_mm'] - planted['r_t_mm'])
        dl = ((planted['x_t_mm'] - res['x_t_mm']) ** 2
              + (planted['y_t_mm'] - res['y_t_mm']) ** 2) ** 0.5
        print(f'\n  planted  : z={planted["z_t_mm"]:6.1f}  r={planted["r_t_mm"]:5.1f}  '
              f'lateral=({planted["x_t_mm"]:.1f},{planted["y_t_mm"]:.1f})')
        print(f'  recovered: z={res["z_t_mm"]:6.1f}  r={res["r_t_mm"]:5.1f}  '
              f'lateral=({res["x_t_mm"]:.1f},{res["y_t_mm"]:.1f})')
        print(f'  errors   : depth={dz:.1f} mm   radius={dr:.1f} mm   lateral={dl:.1f} mm')
        print('  READ →  small errors  : search/code is FINE → real failure is weak signal (physics)')
        print('          still wrong   : genuine BUG in the inverse search/objective', flush=True)


def run_cohort(args):
    dataset, patients = build_dataset()
    if args.list:
        for i, p in enumerate(patients):
            print(f'{i:3d} | {p["id"]:22s} | {p["label"]}')
        return
    encoder, mlp = load_models()
    out_csv = Path(args.out)
    done = done_ids(out_csv) if args.resume else set()
    idxs = select_indices(patients, args)
    n_skip = sum(1 for i in idxs if patients[i]['id'] in done)
    print(f'Selected {len(idxs)} patients | resume will skip {n_skip} already done '
          f'| FEA={"off" if args.no_fea else "on"} | out={out_csv}', flush=True)

    t0, n_run = time.time(), 0
    for n, idx in enumerate(idxs, 1):
        pid, lab = patients[idx]['id'], patients[idx]['label']
        if args.resume and pid in done:
            print(f'[{n}/{len(idxs)}] {pid} — already in CSV, skipping')
            continue
        bar = '═' * 60
        print(f'\n{bar}\n[{n}/{len(idxs)}] {pid} ({lab})  idx={idx}\n{bar}', flush=True)
        ts = time.time()
        try:
            row = process_one(dataset, idx, encoder, mlp, do_fea=not args.no_fea,
                              method=args.method)
        except Exception as e:
            import traceback; traceback.print_exc()
            row = {'patient_id': pid, 'label': lab, 'Q_max': np.nan,
                   'r_t_mm': np.nan, 'fea_status': f'PIPELINE_FAILED: {e}'}
        append_row(out_csv, row)
        n_run += 1
        print(f'  ✓ {pid} in {(time.time()-ts)/60:.1f} min → appended to {out_csv}',
              flush=True)

    print(f'\nFinished: {n_run} processed this run in {(time.time()-t0)/60:.1f} min.')
    if Path(out_csv).exists():
        d = pd.read_csv(out_csv)
        print(f'CSV now holds {len(d)} rows ({int(d["Q_max"].notna().sum())} with results).')


def run_synthetic(args):
    """Verification layer #1 — plant known tumours, forward-generate self-consistent
    skin IR, run the unmodified inverse, check it recovers planted depth & radius."""
    dataset, patients = build_dataset()
    encoder, mlp = load_models()
    idx = args.syn_idx
    geo0 = load_patient_geo(
        dataset, idx, encoder, mlp, CFG, device,
        gaussian_sigma=GAUSSIAN_SIGMA, mc_threshold_override=MC_THRESHOLD,
        repair_holes=REPAIR_HOLES, laplacian_iters=LAPLACIAN_ITERS,
        thermal_sigma=THERMAL_SIGMA)

    zlo, zhi = float(geo0['bbox_min'][2]), float(geo0['bbox_max'][2])
    zr = zhi - zlo
    scenarios = [dict(z=zlo + 0.65 * zr, r=10.0),
                 dict(z=zlo + 0.50 * zr, r=15.0),
                 dict(z=zlo + 0.35 * zr, r=20.0)]

    rows = []
    for sc in scenarios:
        print(f"\n=== PLANT z={sc['z']:.1f} r={sc['r']:.1f} ===", flush=True)
        T_syn, planted = generate_synthetic_skin(geo0, device, sc['z'], sc['r'],
                                                 noise_pct=args.noise)
        geo_syn = dict(geo0); geo_syn['T_measured'] = T_syn
        res, _ = train_pinn_single(geo_syn, device, save_plot=False)
        rows.append((planted, res))
        print(f"  RECOVERED z={res['z_t_mm']:.1f} r={res['r_t_mm']:.1f} "
              f"depth={res['depth_mm']:.1f}  (planted depth={planted['depth_mm']:.1f})",
              flush=True)

    df = pd.DataFrame([{
        'planted_depth': p['depth_mm'], 'recovered_depth': r['depth_mm'],
        'planted_r':     p['r_t_mm'],   'recovered_r':     r['r_t_mm'],
        'lateral_err_mm': ((p['x_t_mm'] - r['x_t_mm']) ** 2
                           + (p['y_t_mm'] - r['y_t_mm']) ** 2) ** 0.5,
    } for p, r in rows])
    print('\nSynthetic recovery summary:\n', df.round(1).to_string(index=False))
    df.to_csv(str(RESULTS_DIR / 'synthetic_recovery.csv'), index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    lo, hi = df.planted_depth.min(), df.planted_depth.max()
    a1.plot([lo, hi], [lo, hi], 'k--', lw=1, label='ideal')
    a1.scatter(df.planted_depth, df.recovered_depth, s=90, c='crimson')
    a1.set_xlabel('planted depth (mm)'); a1.set_ylabel('recovered depth (mm)')
    a1.set_title('Depth recovery'); a1.legend()
    lo, hi = df.planted_r.min(), df.planted_r.max()
    a2.plot([lo, hi], [lo, hi], 'k--', lw=1, label='ideal')
    a2.scatter(df.planted_r, df.recovered_r, s=90, c='steelblue')
    a2.set_xlabel('planted radius (mm)'); a2.set_ylabel('recovered radius (mm)')
    a2.set_title('Radius recovery'); a2.legend()
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / 'synthetic_recovery.png'), dpi=150)
    print(f'Saved → {RESULTS_DIR / "synthetic_recovery.png"} '
          f'and {RESULTS_DIR / "synthetic_recovery.csv"}')


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='TherMAM-NeRF → PINN v3 cohort runner (tmux-friendly, resumable).')
    ap.add_argument('--all', action='store_true',
                    help='run the full cohort (this is the default with no selection)')
    ap.add_argument('--limit', type=int, default=None, help='only the first N patients')
    ap.add_argument('--subset', type=int, default=None,
                    help='balanced N patients across labels (benign/malignant)')
    ap.add_argument('--patients', type=str, default=None,
                    help='comma-separated patient IDs, e.g. Patient_1,Patient_408')
    ap.add_argument('--indices', type=str, default=None,
                    help='comma-separated dataset indices')
    ap.add_argument('--no-fea', dest='no_fea', action='store_true',
                    help='skip FEA forward verification (PINN-only, faster)')
    ap.add_argument('--method', choices=['forward_match', 'mukhmetov'],
                    default='forward_match',
                    help="inverse: 'mukhmetov' = fast Dirichlet-skin single-opt (no grid); "
                         "'forward_match' = v3 grid search (slow)")
    ap.add_argument('--resume', dest='resume', action='store_true', default=True,
                    help='skip patients already in the CSV (default on)')
    ap.add_argument('--no-resume', dest='resume', action='store_false',
                    help='reprocess every selected patient')
    ap.add_argument('--out', type=str, default=str(RESULTS_DIR / 'pinn_fea_results.csv'),
                    help='output CSV path')
    ap.add_argument('--list', action='store_true',
                    help='print the index→patient map and exit')
    ap.add_argument('--synthetic', action='store_true',
                    help='run the synthetic recovery validation instead of the cohort')
    ap.add_argument('--syn-idx', dest='syn_idx', type=int, default=0,
                    help='patient index to borrow geometry from (synthetic mode)')
    ap.add_argument('--noise', type=float, default=0.0,
                    help='synthetic skin-noise fraction (Bezerra-style robustness)')
    ap.add_argument('--diagnose', action='store_true',
                    help='bug-vs-physics diagnostics (tumour on/off + x100 source)')
    ap.add_argument('--test', choices=['1', '2', 'both'], default='both',
                    help='which diagnostic test to run (default both)')
    args = ap.parse_args()

    print(f'Device: {device}', flush=True)
    if args.diagnose:
        run_diagnose(args)
    elif args.synthetic:
        run_synthetic(args)
    else:
        run_cohort(args)


if __name__ == '__main__':
    main()
