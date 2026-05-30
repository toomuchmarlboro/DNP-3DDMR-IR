#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# Physics-Informed Neural Network (PINN) + FEA Bioheat Pipeline for Thermography
# ==============================================================================


import os, sys, time, random, struct, math
import numpy as np
import cv2
import tifffile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict
from tqdm.auto import tqdm
from scipy.ndimage import (gaussian_filter, binary_closing, binary_fill_holes, generate_binary_structure, label, map_coordinates)
from scipy.spatial import cKDTree
from scipy import stats
from skimage.measure import marching_cubes
import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# FEA stack (bioheat conda only)
import gmsh
from mpi4py import MPI
import dolfinx, dolfinx.io
try:
    from dolfinx.io import gmshio
except ImportError:
    from dolfinx.io import gmsh as gmshio
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Warm up CUDA and initialize primary context to prevent lazy cuBLAS context warnings
if device.type == "cuda":
    torch.cuda.init()
    _ = torch.zeros(1, device=device)


# ── Paths (v-maxwell-pc2 server) ──
ROOT        = Path("/mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR")
TIFF_BASE   = ROOT / "data" / "organized_by_patient"
UNET_CKPT   = ROOT / "UNET_Segmentation" / "breast_segmentation_unet_best_gpu.pth"
BREASTNET_CKPT = ROOT / "UNET_Segmentation" / "3DBreastnet" / "checkpoints_3d_finetuned" / "3dbreastnet_finetuned_best.pth"

# Output directories
STL_DIR     = ROOT / "UNET_Segmentation" / "PINNpdeSolver" / "exported_stls"
RESULTS_DIR = ROOT / "UNET_Segmentation" / "PINNpdeSolver" / "results"
STL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print(f"TIFF base exists : {TIFF_BASE.exists()}")
print(f"U-Net ckpt exists: {UNET_CKPT.exists()}")
print(f"3D ckpt exists   : {BREASTNET_CKPT.exists()}")


# ==============================================================================
# 1. 3D RECONSTRUCTION & SEGMENTATION MODELS
# ==============================================================================


# ── Building blocks ──
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

# ── U-Net (segmentation, frozen) ──
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

# ── Encoder 2D (5x128x128 -> 1000-d latent) ──
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

# ── Decoder 3D (1000-d -> 1x128x128x128) ──
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
        self.out = nn.Sequential(nn.Conv3d(8,1,1), nn.Sigmoid())
        self.apply(_init)
        nn.init.constant_(self.out[0].bias, -4.0)

    def _s4(self, x): return self.d4(self.up4(x))
    def _s5(self, x): return self.d5(self.up5(x))
    def _s6(self, x): return self.d6(self.up6(x))

    def forward(self, x):
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
        return self.out(x)

print("Model classes defined: UNet, Encoder2D, Decoder3D")


# ---
# # DATA LOADING (dual-track thermal: normalised + absolute °C)
# #### Matches v5's build_patient_groups/PatientDataset pattern
# #### ADDS thermals_abs_5ch for PINN boundary conditions
# ---

# In[5]:


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

class PatientDatasetDualTrack(Dataset):
    """Extended PatientDataset that returns BOTH normalised and absolute thermals."""
    def __init__(self, groups, unet, device):
        self.groups, self.unet, self.device = groups, unet, device
        self.views = ["RL", "RO", "F", "LO", "LL"]

    def __len__(self): return len(self.groups)

    def __getitem__(self, idx):
        g = self.groups[idx]
        thermals_norm, thermals_abs, masks_128 = [], [], []

        for v in self.views:
            raw = tifffile.imread(str(g.views[v])).astype(np.float32)
            raw_256 = cv2.resize(raw, (256, 256))

            # Normalised [0,1] — for U-Net input + 3D reconstruction
            mn, mx = raw_256.min(), raw_256.max()
            norm = (raw_256 - mn) / (mx - mn + 1e-8)

            with torch.no_grad():
                inp = torch.tensor(norm).unsqueeze(0).unsqueeze(0).to(self.device)
                m = (torch.sigmoid(self.unet(inp)).squeeze().cpu().numpy() > 0.5
                     ).astype(np.float32)

            masks_128.append(cv2.resize(m, (128, 128), interpolation=cv2.INTER_NEAREST))

            # Normalised 128x128 (for BreastNet3D + visual overlay)
            norm_128 = cv2.resize(norm, (128, 128), interpolation=cv2.INTER_LINEAR)
            thermals_norm.append(norm_128)

            # Absolute °C 128x128 (for PINN boundary condition!)
            abs_128 = cv2.resize(raw_256, (128, 128), interpolation=cv2.INTER_LINEAR)
            thermals_abs.append(abs_128)

        return {
            "masks_5ch"       : torch.tensor(np.stack(masks_128), dtype=torch.float32),
            "thermals_5ch"    : torch.tensor(np.stack(thermals_norm), dtype=torch.float32),
            "thermals_abs_5ch": torch.tensor(np.stack(thermals_abs), dtype=torch.float32),
            "patient_id"      : g.patient_id,
            "label"           : g.label,
        }

# ── Load all models ──
print("Loading U-Net...")
unet = UNet().to(device)
unet.load_state_dict(torch.load(str(UNET_CKPT), map_location=device, weights_only=False))
unet.eval()
for p in unet.parameters(): p.requires_grad = False

print("Loading Encoder2D + Decoder3D...")
ckpt = torch.load(str(BREASTNET_CKPT), map_location=device, weights_only=False)
enc = Encoder2D().to(device); enc.load_state_dict(ckpt["enc"]); enc.eval()
dec = Decoder3D().to(device); dec.load_state_dict(ckpt["dec"]); dec.eval()

print("Building patient groups...")
groups = build_patient_groups(str(TIFF_BASE))
dataset = PatientDatasetDualTrack(groups, unet, device)
print(f"Dataset ready: {len(dataset)} patients")

# Sanity check — verify absolute thermal range
sample = dataset[0]
t_abs = sample["thermals_abs_5ch"]
print(f"Sample absolute thermal range: {t_abs.min():.1f} – {t_abs.max():.1f} "
      f"(expect ~28–38°C)")


# ---
# # GEOMETRY: mesh generation, thermal overlay, IMF registration
# #### Produces the PINN inputs: surface_pts, T_measured, interior_pts
# ---

# In[6]:


# ── Thermal overlay (from v5 Cell 16, extended for absolute °C) ──
def compute_thermal_overlay(verts, normals, thermals_5ch, masks_5ch,
                            use_absolute=False, thermals_abs_5ch=None):
    """
    use_absolute=False → [0,1] normalised  (for 3D visualisation)
    use_absolute=True  → absolute °C       (for PINN boundary condition)
    Returns: per-vertex thermal values, per-vertex confidence weights
    """
    src = thermals_abs_5ch if (use_absolute and thermals_abs_5ch is not None) \
          else thermals_5ch

    intensities = np.zeros(len(verts))
    weight_sum  = np.zeros(len(verts))

    w_in = verts[:, 2] - 63.5
    d_in = verts[:, 0] - 63.5
    h_in = verts[:, 1]

    outward_normals = -normals
    view_angles = [-90, -45, 0, 45, 90]

    for i, angle in enumerate(view_angles):
        rad = np.deg2rad(angle)
        c, s = np.cos(rad), np.sin(rad)
        cam_vec_w = -s
        cam_vec_d = -c

        dot_prod = outward_normals[:, 2] * cam_vec_w + outward_normals[:, 0] * cam_vec_d
        weights = np.maximum(dot_prod + 0.3, 0.0) ** 12

        valid_idx = np.where(weights > 0)[0]
        if len(valid_idx) == 0: continue

        x_img = c * w_in[valid_idx] - s * d_in[valid_idx]
        x_coords = x_img + 63.5
        y_coords = h_in[valid_idx]

        thermal_img = src[0, i].cpu().numpy()
        sampled = map_coordinates(thermal_img, [y_coords, x_coords],
                                  order=1, mode='nearest')

        intensities[valid_idx] += sampled * weights[valid_idx]
        weight_sum[valid_idx]  += weights[valid_idx]

    untextured = weight_sum == 0
    weight_sum[untextured] = 1.0
    final = intensities / weight_sum

    # Frontal fallback for untextured vertices
    if np.any(untextured):
        x_base = w_in[untextured] + 63.5
        y_base = h_in[untextured]
        thermal_base = src[0, 2].cpu().numpy()  # Index 2 = Frontal
        final[untextured] = map_coordinates(thermal_base, [y_base, x_base],
                                            order=1, mode='nearest')

    # Confidence weights for PINN data loss
    confidence = np.ones(len(verts))
    low_weight = weight_sum < np.percentile(weight_sum[~untextured], 25)
    confidence[low_weight]  = 0.5
    confidence[untextured]  = 0.3

    return final, confidence

# ── IMF-based anatomical registration ──
def extract_imf_centroid_2d(mask_frontal):
    rows = np.where(mask_frontal.any(axis=1))[0]
    if len(rows) == 0: raise ValueError("Empty mask")
    imf_row  = rows.max()
    imf_cols = np.where(mask_frontal[imf_row])[0]
    return np.array([imf_cols.mean(), imf_row])

def extract_superior_apex_2d(mask_frontal):
    rows = np.where(mask_frontal.any(axis=1))[0]
    apex_row  = rows.min()
    apex_cols = np.where(mask_frontal[apex_row])[0]
    return np.array([apex_cols.mean(), apex_row])

def _normalise_vec(v):
    return v / (np.linalg.norm(v) + 1e-8)

def build_anatomical_frame(imf_3d, apex_3d):
    y_axis  = _normalise_vec(apex_3d - imf_3d)
    z_world = np.array([0.0, 0.0, 1.0])
    x_axis  = _normalise_vec(np.cross(y_axis, z_world))
    z_axis  = _normalise_vec(np.cross(x_axis, y_axis))
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return R, imf_3d

def register_mesh_to_imf(verts, mask_frontal):
    """Register verts (in voxel coords) to IMF-centred anatomical frame."""
    imf_2d  = extract_imf_centroid_2d(mask_frontal)
    apex_2d = extract_superior_apex_2d(mask_frontal)
    # Map 2D landmarks to 3D voxel coords (col→Z, row→Y, depth≈64)
    imf_3d  = np.array([64.0, imf_2d[1],  imf_2d[0]])
    apex_3d = np.array([64.0, apex_2d[1], apex_2d[0]])
    R, t = build_anatomical_frame(imf_3d, apex_3d)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    T_inv = np.linalg.inv(T)
    ones = np.ones((len(verts), 1))
    verts_h = np.hstack([verts, ones])
    return (T_inv @ verts_h.T).T[:, :3]

# ── Generate interior points inside mesh ──
def sample_interior_points(verts, n_points=5000):
    """Sample random points inside the convex hull of the mesh vertices."""
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    margin = (bbox_max - bbox_min) * 0.05
    pts = np.random.uniform(bbox_min - margin, bbox_max + margin,
                            size=(n_points * 3, 3))
    # Simple rejection: keep points close to centroid
    centroid = verts.mean(axis=0)
    extents  = (bbox_max - bbox_min) / 2
    rel = np.abs(pts - centroid) / (extents + 1e-8)
    inside = np.all(rel < 1.0, axis=1)
    return pts[inside][:n_points]

# ── Full per-patient geometry pipeline ──
def process_patient_geometry(item, enc, dec, device):
    """
    Takes a dataset item → returns everything the PINN needs.
    Verts stay in voxel units (≈ mm scale at 128³).
    """
    pid   = item["patient_id"]
    label = item["label"]
    m5    = item["masks_5ch"].unsqueeze(0).to(device)
    t5    = item["thermals_5ch"].unsqueeze(0).to(device)
    t5abs = item["thermals_abs_5ch"].unsqueeze(0).to(device)

    # 1. Reconstruct 128³ volume
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        vol = dec(enc(m5)).float()
    vol_np = vol[0, 0].cpu().numpy()

    # 2. Clean and mesh (same as v5 Cell 18)
    vol_np = np.pad(vol_np, pad_width=1, mode='constant', constant_values=0)
    vol_np = gaussian_filter(vol_np, sigma=2.0)
    verts, faces, normals, _ = marching_cubes(vol_np, level=0.02)
    verts = verts - 1.0  # compensate for padding

    # 3. Thermal overlay — ABSOLUTE °C for PINN
    T_measured, confidence = compute_thermal_overlay(
        verts, normals, t5, m5,
        use_absolute=True, thermals_abs_5ch=t5abs
    )

    # 4. IMF registration
    mask_frontal = item["masks_5ch"][2].numpy()  # index 2 = Frontal
    verts_reg = register_mesh_to_imf(verts, mask_frontal)

    # 5. Interior points
    interior_pts = sample_interior_points(verts_reg, n_points=5000)

    # 6. Bounding box for coordinate normalisation
    bbox_min = verts_reg.min(axis=0)
    bbox_max = verts_reg.max(axis=0)
    bbox_extents = bbox_max - bbox_min

    result = {
        "patient_id"   : pid,
        "label"        : label,
        "surface_pts"  : verts_reg.astype(np.float32),         # [N_s, 3] mm
        "T_measured"   : T_measured.astype(np.float32),        # [N_s]    °C
        "confidence"   : confidence.astype(np.float32),        # [N_s]    weights
        "interior_pts" : interior_pts.astype(np.float32),      # [N_v, 3] mm
        "bbox_min"     : bbox_min.astype(np.float32),
        "bbox_max"     : bbox_max.astype(np.float32),
        "bbox_extents" : bbox_extents.astype(np.float32),
        "verts_raw"    : verts.astype(np.float32),             # for STL export
        "faces"        : faces,
    }
    return result

# ── Quick test on one patient ──
test_item = dataset[0]
test_geo  = process_patient_geometry(test_item, enc, dec, device)
print(f"Patient: {test_geo['patient_id']} ({test_geo['label']})")
print(f"  Surface pts : {test_geo['surface_pts'].shape}")
print(f"  T_measured  : {test_geo['T_measured'].min():.1f} – {test_geo['T_measured'].max():.1f} °C")
print(f"  Interior pts: {test_geo['interior_pts'].shape}")
print(f"  BBox extents: {test_geo['bbox_extents']}")


# ==============================================================================
# 2. PINN INVERSE BIOHEAT SOLVER
# ==============================================================================

# ── Biophysical constants (SI) ──
K_TISSUE   = 0.48       # W/(m·K)  thermal conductivity
OMEGA_B    = 0.0005     # 1/s      blood perfusion rate
C_BLOOD    = 3600.0     # J/(kg·K) specific heat of blood
T_ARTERIAL = 37.0       # °C       arterial blood temperature
Q_METAB    = 450.0      # W/m³     healthy metabolic heat
COORD_SCALE = 1e-3      # mm → m   applied in PDE residual only

class BioheatPINN(nn.Module):
    def __init__(self, hidden=256, depth=6):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

        # Learnable tumour parameters
        self.x_t   = nn.Parameter(torch.tensor([0.0]))
        self.y_t   = nn.Parameter(torch.tensor([0.0]))
        self.z_t   = nn.Parameter(torch.tensor([0.0]))
        self.r_t   = nn.Parameter(torch.tensor([10.0]))    # mm
        self.Q_max = nn.Parameter(torch.tensor([5000.0]))  # W/m³

    def forward(self, xyz):
        return self.net(xyz).squeeze(-1)

    def Q_tumor(self, xyz):
        d2 = ((xyz[:,0] - self.x_t)**2 +
              (xyz[:,1] - self.y_t)**2 +
              (xyz[:,2] - self.z_t)**2)
        return self.Q_max * torch.exp(-d2 / self.r_t**2)

def compute_laplacian(T, xyz, extent_meters):
    """∇²T via autograd in physical space (meters) — requires xyz.requires_grad=True"""
    dT = torch.autograd.grad(
        T, xyz, grad_outputs=torch.ones_like(T), create_graph=True
    )[0]
    lap = 0.0
    for i in range(3):
        d2T = torch.autograd.grad(
            dT[:,i], xyz,
            grad_outputs=torch.ones_like(dT[:,i]),
            create_graph=True
        )[0][:,i]
        lap = lap + d2T / (extent_meters[i]**2)
    return lap

def normalise_coords(pts, bbox_min, bbox_max):
    """Map mm coordinates to [-1, 1] for network input."""
    centre = (bbox_max + bbox_min) / 2
    extent = (bbox_max - bbox_min) / 2 + 1e-8
    return (pts - centre) / extent

def denormalise_coord(val, bbox_min_i, bbox_max_i):
    """Map normalised [-1,1] back to mm."""
    centre = (bbox_max_i + bbox_min_i) / 2
    extent = (bbox_max_i - bbox_min_i) / 2
    return val * extent + centre

def classify_quadrant(x_mm, y_mm):
    if x_mm > 0 and y_mm > 0: return "Upper Outer"
    if x_mm < 0 and y_mm > 0: return "Upper Inner"
    if x_mm > 0 and y_mm < 0: return "Lower Outer"
    return "Lower Inner"

def train_pinn_single(geo, device, n_starts=3, adam_steps=3000, lbfgs_steps=100):
    """
    Train PINN for one patient geometry.
    Multi-start: n_starts random initialisations, keep lowest loss.
    """
    surf_pts = torch.tensor(geo["surface_pts"], dtype=torch.float32, device=device)
    T_meas   = torch.tensor(geo["T_measured"],  dtype=torch.float32, device=device)
    conf     = torch.tensor(geo["confidence"],   dtype=torch.float32, device=device)
    int_pts  = torch.tensor(geo["interior_pts"], dtype=torch.float32, device=device)
    bbox_min = torch.tensor(geo["bbox_min"],     dtype=torch.float32, device=device)
    bbox_max = torch.tensor(geo["bbox_max"],     dtype=torch.float32, device=device)

    # Calculate extents in meters for physical Laplacian scaling
    extent_mm = (bbox_max - bbox_min) / 2 + 1e-8
    extent_meters = extent_mm * 1e-3

    # Normalise coordinates to [-1, 1]
    surf_norm = normalise_coords(surf_pts, bbox_min, bbox_max)
    int_norm  = normalise_coords(int_pts,  bbox_min, bbox_max)

    best_loss  = float("inf")
    best_state = None
    best_params = None
    best_history = None

    for start in range(n_starts):
        model = BioheatPINN().to(device)
        # Random tumour position init
        model.x_t.data = torch.tensor([random.uniform(-0.5, 0.5)], device=device)
        model.y_t.data = torch.tensor([random.uniform(-0.5, 0.5)], device=device)
        model.z_t.data = torch.tensor([random.uniform(-0.5, 0.5)], device=device)

        # Separate LR: tumour params need 10x higher LR to overcome
        # the weak PDE→tumour gradient signal
        tumour_params = [model.x_t, model.y_t, model.z_t, model.r_t, model.Q_max]
        net_params    = list(model.net.parameters())
        optimizer = torch.optim.Adam([
            {"params": net_params,    "lr": 1e-3},
            {"params": tumour_params, "lr": 1e-2},   # 10x higher
        ])
        lambda_data = 1.0
        lambda_pde  = None  # set at step 0, refreshed every 1000 steps

        # History tracking for convergence plotting
        history = {"step": [], "L_data": [], "L_pde": [], "loss": []}

        # ── Phase 1: Adam ──
        for step in range(adam_steps):
            model.train()
            optimizer.zero_grad()

            # Data loss (surface)
            T_pred_surf = model(surf_norm)
            L_data = (conf * (T_pred_surf - T_meas)**2).mean()

            # PDE loss (interior)
            int_input = int_norm.clone().detach().requires_grad_(True)
            T_pred_int = model(int_input)
            lap = compute_laplacian(T_pred_int, int_input, extent_meters)

            # Pennes equation: k∇²T + ωb·cb·(Ta-T) + Qm + Qt = 0
            # Physical Laplacian scaling is already incorporated in extent_meters
            Q_t = model.Q_tumor(int_input)
            pde_residual = (K_TISSUE * lap
                          + OMEGA_B * C_BLOOD * (T_ARTERIAL - T_pred_int)
                          + Q_METAB + Q_t)
            L_pde = (pde_residual**2).mean()

            # Adaptive normalisation: rebalance every 1000 steps
            if lambda_pde is None or step % 1000 == 0:
                lambda_pde = (L_data / (L_pde + 1e-12)).detach()

            loss = lambda_data * L_data + lambda_pde * L_pde
            loss.backward()
            optimizer.step()

            # Clamp physical constraints
            model.r_t.data.clamp_(min=2.0)
            model.Q_max.data.clamp_(min=0.0)

            # Record history (every 50 steps)
            if step % 50 == 0:
                history["step"].append(step)
                history["L_data"].append(L_data.item())
                history["L_pde"].append(L_pde.item())
                history["loss"].append(loss.item())

            if step % 2000 == 0:
                print(f"  [start {start+1}/{n_starts}] step {step:5d} | "
                      f"L_data={L_data.item():.4f} L_pde={L_pde.item():.4f} | "
                      f"r_t={model.r_t.item():.2f} Q_max={model.Q_max.item():.1f}")

        # ── Phase 2: L-BFGS ──
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=20,
                                   line_search_fn="strong_wolfe")
        for lbfgs_idx in range(lbfgs_steps // 20):
            def closure():
                lbfgs.zero_grad()
                T_s = model(surf_norm)
                ld = (conf * (T_s - T_meas)**2).mean()
                ii = int_norm.clone().detach().requires_grad_(True)
                T_i = model(ii)
                lap_ = compute_laplacian(T_i, ii, extent_meters)
                Qt_ = model.Q_tumor(ii)
                res = (K_TISSUE * lap_
                      + OMEGA_B * C_BLOOD * (T_ARTERIAL - T_i)
                      + Q_METAB + Qt_)
                lp = (res**2).mean()
                l = lambda_data * ld + lambda_pde * lp
                l.backward()
                model.r_t.data.clamp_(min=2.0)
                model.Q_max.data.clamp_(min=0.0)
                return l
            lbfgs.step(closure)

            # Record history at the end of each L-BFGS step block (20 steps)
            curr_step = adam_steps + (lbfgs_idx + 1) * 20
            # MUST enable gradients to compute the autograd Laplacian!
            with torch.enable_grad():
                ii = int_norm.clone().detach().requires_grad_(True)
                T_i = model(ii)
                lap_ = compute_laplacian(T_i, ii, extent_meters)
                Qt_ = model.Q_tumor(ii)
                res = (K_TISSUE * lap_
                      + OMEGA_B * C_BLOOD * (T_ARTERIAL - T_i)
                      + Q_METAB + Qt_)
                lp = (res**2).mean().item()

            with torch.no_grad():
                T_s = model(surf_norm)
                ld = (conf * (T_s - T_meas)**2).mean().item()
                l_weighted = lambda_data * ld + lambda_pde.item() * lp

            history["step"].append(curr_step)
            history["L_data"].append(ld)
            history["L_pde"].append(lp)
            history["loss"].append(l_weighted)

        # Evaluate final loss
        with torch.no_grad():
            final_loss = (conf * (model(surf_norm) - T_meas)**2).mean().item()

        if final_loss < best_loss:
            best_loss  = final_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_params = {
                "x_t_norm": model.x_t.item(),
                "y_t_norm": model.y_t.item(),
                "z_t_norm": model.z_t.item(),
                "r_t_mm"  : abs(model.r_t.item()),
                "Q_max"   : model.Q_max.item(),
            }
            best_history = {k: list(v) for k, v in history.items()}
        print(f"  Start {start+1} final data loss: {final_loss:.6f} "
              f"{'★ best' if final_loss == best_loss else ''}")

    # Denormalise tumour coordinates to mm
    bp = best_params
    x_mm = denormalise_coord(bp["x_t_norm"], geo["bbox_min"][0], geo["bbox_max"][0])
    y_mm = denormalise_coord(bp["y_t_norm"], geo["bbox_min"][1], geo["bbox_max"][1])
    z_mm = denormalise_coord(bp["z_t_norm"], geo["bbox_min"][2], geo["bbox_max"][2])

    # ── Generate and save convergence plot for the best start ──
    if best_history is not None and len(best_history["step"]) > 0:
        plt.figure(figsize=(10, 5))
        sns.set_theme(style="whitegrid")
        steps = best_history["step"]
        plt.plot(steps, best_history["L_data"], label="Data Loss ($L_{data}$)", color="#1f77b4", linewidth=2)
        plt.plot(steps, best_history["L_pde"], label="PDE Loss ($L_{pde}$)", color="#ff7f0e", linewidth=2)
        plt.plot(steps, best_history["loss"], label="Weighted Total Loss ($L_{total}$)", color="#2ca02c", linestyle="--", linewidth=1.5)
        plt.yscale("log")
        plt.xlabel("Optimization Steps", fontsize=11, fontweight="bold")
        plt.ylabel("Loss Value (Log Scale)", fontsize=11, fontweight="bold")
        plt.title(f"PINN Loss Convergence — Patient: {geo['patient_id']} ({geo['label']})", fontsize=13, fontweight="bold", pad=15)
        plt.axvline(x=adam_steps, color="red", linestyle=":", label="L-BFGS Fine-Tuning Transition", linewidth=1.5)
        plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()
        plot_path = RESULTS_DIR / f"{geo['patient_id']}_loss_convergence.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  ★ Saved loss convergence plot to {plot_path.name}")

    results = {
        "patient_id" : geo["patient_id"],
        "label"      : geo["label"],
        "x_t_mm"     : x_mm,
        "y_t_mm"     : y_mm,
        "z_t_mm"     : z_mm,
        "r_t_mm"     : bp["r_t_mm"],
        "Q_max"      : bp["Q_max"],
        "volume_mm3" : (4/3) * np.pi * bp["r_t_mm"]**3,
        "quadrant"   : classify_quadrant(x_mm, y_mm),
        "data_loss"  : best_loss,
    }
    return results, best_state

print("PINN solver defined. Ready to train.")


# ==============================================================================
# 3. FEA FORWARD VERIFICATION (FEniCSx / dolfinx)
# ==============================================================================


def stl_to_tet_mesh(stl_path, out_msh_path, mesh_size_mm=3.0):
    """Convert watertight STL → tetrahedral .msh via gmsh without re-parametrization."""
    gmsh.initialize()
    gmsh.model.add("breast")
    gmsh.merge(str(stl_path))
    
    # Retrieve imported STL surface entities directly
    s = gmsh.model.getEntities(2)
    l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([l])
    gmsh.model.geo.synchronize()
    
    # Define physical groups (required by FEniCSx dolfinx.io mesh reader)
    v_entities = gmsh.model.getEntities(3)
    gmsh.model.addPhysicalGroup(3, [e[1] for e in v_entities], 1)
    gmsh.model.setPhysicalName(3, 1, "breast_volume")
    
    s_entities = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [e[1] for e in s_entities], 2)
    gmsh.model.setPhysicalName(2, 2, "breast_surface")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_mm)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_mm * 0.5)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.write(str(out_msh_path))
    gmsh.finalize()

def save_stl_binary(filename, verts, faces):
    """Write binary STL (no trimesh dependency needed)."""
    with open(filename, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(faces)))
        for face in faces:
            tri = verts[face].astype(np.float32)
            v0, v1, v2 = tri
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            normal = (normal / norm).astype(np.float32) if norm > 0 else np.zeros(3, dtype=np.float32)
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))

def run_fea_forward(msh_path, pinn_results):
    """FEA forward solve using PINN-estimated tumour parameters."""
    fea_import = gmshio.read_from_msh(
        str(msh_path), MPI.COMM_WORLD, gdim=3
    )
    msh = fea_import.mesh if hasattr(fea_import, "mesh") else fea_import[0]
    msh.geometry.x[:] *= 1e-3  # mm → m

    V = fem.functionspace(msh, ("Lagrange", 1))
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    x_t  = pinn_results["x_t_mm"] * 1e-3
    y_t  = pinn_results["y_t_mm"] * 1e-3
    z_t  = pinn_results["z_t_mm"] * 1e-3
    r_t  = pinn_results["r_t_mm"] * 1e-3
    Qmax = pinn_results["Q_max"]

    d2      = (x[0]-x_t)**2 + (x[1]-y_t)**2 + (x[2]-z_t)**2
    Q_tumor = Qmax * ufl.exp(-d2 / r_t**2)

    k=0.48; wb=0.0005; cb=3600.0; Ta=37.0; Qm=450.0
    h_conv=10.0; T_air=20.0

    # Locate flat chest wall boundary (near minimum Z coordinate)
    z_min_meters = np.min(msh.geometry.x[:, 2])
    def chest_wall_boundary(pt):
        return pt[2] < (z_min_meters + 0.005)  # 5mm tolerance

    facets = dolfinx.mesh.locate_entities_boundary(msh, msh.topology.dim - 1, chest_wall_boundary)
    dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    
    # Core body temperature at chest wall boundary is 37°C
    bc_chest = fem.dirichletbc(37.0, dofs, V)

    a = (k * ufl.inner(ufl.grad(T), ufl.grad(v))
         + wb * cb * T * v) * ufl.dx + h_conv * T * v * ufl.ds
    L = (wb * cb * Ta + Qm + Q_tumor) * v * ufl.dx + h_conv * T_air * v * ufl.ds

    try:
        problem = LinearProblem(
            a, L, bcs=[bc_chest],
            petsc_options_prefix="bioheat_",
            petsc_options={"ksp_type": "cg", "pc_type": "gamg"}
        )
    except TypeError:
        problem = LinearProblem(
            a, L, bcs=[bc_chest],
            petsc_options={"ksp_type": "cg", "pc_type": "gamg"}
        )
    return problem.solve(), msh

def compute_fea_residual(T_fea_sol, msh, surface_pts_mm, T_measured_abs):
    """Compare FEA surface temps vs measured → residual."""
    fea_coords = msh.geometry.x * 1e3   # m → mm
    fea_vals   = T_fea_sol.x.array      # °C

    tree = cKDTree(fea_coords)
    _, idx = tree.query(surface_pts_mm, k=3)
    T_fea_interp = fea_vals[idx].mean(axis=1)

    residuals = np.abs(T_fea_interp - T_measured_abs)
    return residuals, residuals.mean(), residuals.max()

print("FEA verification functions defined.")


# ==============================================================================
# 4. MAIN COHORT PROCESSING PIPELINE (122 Patients Loop)
# ==============================================================================


all_results = []

for patient_idx in tqdm(range(len(dataset)), desc="Processing patients"):
    item = dataset[patient_idx]
    pid  = item["patient_id"]
    lab  = item["label"]
    print(f"\n{'═'*60}")
    print(f"Patient {patient_idx+1}/{len(dataset)}: {pid} ({lab})")
    print(f"{'═'*60}")

    try:
        # Create patient-specific output directory
        patient_dir = RESULTS_DIR / pid
        patient_dir.mkdir(exist_ok=True, parents=True)

        # ── Step 1: Geometry extraction ──
        print("  [1/4] Extracting geometry + thermal overlay...")
        geo = process_patient_geometry(item, enc, dec, device)

        # Save intermediate files in patient folder
        np.save(patient_dir / f"{pid}_T_measured.npy",  geo["T_measured"])
        np.save(patient_dir / f"{pid}_surf_pts.npy",    geo["surface_pts"])
        np.save(patient_dir / f"{pid}_confidence.npy",  geo["confidence"])

        # Export STL in patient folder (in registered mm coordinates)
        stl_path = patient_dir / f"{pid}.stl"
        save_stl_binary(stl_path, geo["surface_pts"], geo["faces"])

        # ── Step 2: PINN training ──
        print("  [2/4] Training PINN (multi-start)...")
        pinn_results, pinn_state = train_pinn_single(
            geo, device, n_starts=3, adam_steps=3000, lbfgs_steps=100
        )
        torch.save(pinn_state, patient_dir / f"{pid}_pinn.pth")

        # ── Step 3: FEA verification ──
        print("  [3/4] Running FEA forward verification...")
        msh_path = patient_dir / f"{pid}.msh"
        try:
            stl_to_tet_mesh(stl_path, msh_path, mesh_size_mm=3.0)
            T_fea_sol, fea_msh = run_fea_forward(msh_path, pinn_results)
            residuals, mean_res, max_res = compute_fea_residual(
                T_fea_sol, fea_msh,
                geo["surface_pts"], geo["T_measured"]
            )
            np.save(patient_dir / f"{pid}_T_fea.npy", residuals)
            pinn_results["fea_mean_residual"] = mean_res
            pinn_results["fea_max_residual"]  = max_res
            pinn_results["fea_status"] = "OK" if mean_res < 1.5 else "HIGH_RESIDUAL"
            print(f"  FEA residual: mean={mean_res:.3f}°C, max={max_res:.3f}°C")
        except Exception as e:
            print(f"  ⚠ FEA failed: {e}")
            pinn_results["fea_mean_residual"] = np.nan
            pinn_results["fea_max_residual"]  = np.nan
            pinn_results["fea_status"] = f"FAILED: {e}"

        # ── Step 4: Collect results ──
        all_results.append(pinn_results)
        print(f"  [4/4] Done — Q_max={pinn_results['Q_max']:.1f} W/m³, "
              f"r_t={pinn_results['r_t_mm']:.2f} mm, "
              f"quadrant={pinn_results['quadrant']}")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_results.append({
            "patient_id": pid, "label": lab,
            "Q_max": np.nan, "r_t_mm": np.nan,
            "fea_status": f"PIPELINE_FAILED: {e}",
        })

# ── Save master CSV ──
df = pd.DataFrame(all_results)
csv_path = RESULTS_DIR / "pinn_fea_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n{'═'*60}")
print(f"All results saved to {csv_path}")
print(f"Processed: {len(df)} | Failed: {df['Q_max'].isna().sum()}")
print(df[["patient_id", "label", "Q_max", "r_t_mm", "quadrant",
          "fea_mean_residual"]].to_string())
