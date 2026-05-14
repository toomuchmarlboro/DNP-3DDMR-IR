"""
evaluate_v7.py — Post-training evaluation & visualization
===========================================================
Copy each section into separate notebook cells.

Section A: View Training Metrics
Section B: 3D Visualization (interactive Plotly)
Section C: Per-Patient Projection Validation (2×5 grids)
Section D: Batch STL Export
"""

# ════════════════════════════════════════════════════════════════
# SECTION A: VIEW TRAINING METRICS
# ════════════════════════════════════════════════════════════════
# %matplotlib inline
import torch
import matplotlib.pyplot as plt
from pathlib import Path

ckpt_path = Path("checkpoints_3d_v7/3dbreastnet_best.pth")
if ckpt_path.exists():
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hist = ckpt["hist"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
    ax[0].plot(hist["epoch"], hist["val_loss"], label="val")
    ax[0].set_title("Dice Loss"); ax[0].legend()
    ax[1].plot(hist["epoch"], hist["val_dice"], color="green")
    ax[1].set_title("Validation Dice Score")
    ax[2].plot(hist["epoch"], hist["val_hd"], color="red")
    ax[2].set_title("Validation HD95")
    for a in ax: a.set_xlabel("Epoch")
    plt.tight_layout(); plt.show()
else:
    print(f"Checkpoint not found at {ckpt_path}.")


# ════════════════════════════════════════════════════════════════
# SECTION B: 3D VISUALIZATION (Plotly)
# ════════════════════════════════════════════════════════════════
import random, math, struct
import numpy as np
import torch, torch.nn.functional as F
from pathlib import Path
from IPython.display import display
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

from models_v7 import (
    UNet, Encoder2D, Decoder3D, compute_visual_hull,
)
from train_v7 import build_patient_groups, PatientDataset, REPO_ROOT

def visualize_random_patients(n=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    unet_ckpt = str(REPO_ROOT / "UNET_Segmentation" / "breast_segmentation_unet_best_gpu.pth")
    tiff_base = str(REPO_ROOT / "data" / "organized_by_patient")
    ckpt_path = Path("checkpoints_3d_v7/3dbreastnet_best.pth")

    if not ckpt_path.exists():
        print(f"Error: No trained model found at {ckpt_path}.")
        return

    print("Loading checkpoints...")
    unet = UNet().to(device)
    unet.load_state_dict(torch.load(unet_ckpt, map_location=device, weights_only=False))
    unet.eval()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = Encoder2D().to(device)
    dec = Decoder3D().to(device)
    enc.load_state_dict(ckpt["enc"])
    dec.load_state_dict(ckpt["dec"])
    enc.eval(); dec.eval()

    groups = build_patient_groups(tiff_base)
    if not groups:
        print("No patients found.")
        return

    selected = random.sample(groups, min(n, len(groups)))
    dataset = PatientDataset(selected, unet, device)
    print(f"Generating 3D models for {len(selected)} patients...")

    for i in range(len(dataset)):
        item = dataset[i]
        pid, label = item["patient_id"], item["label"]
        m5 = item["masks_5ch"].unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            hull = compute_visual_hull(m5, device)
            vol = dec(enc(m5), hull)
        vol_np = vol[0, 0].float().cpu().numpy()
        vmin, vmax = float(vol_np.min()), float(vol_np.max())
        print(f"{pid}: min={vmin:.4f} max={vmax:.4f} mean={float(vol_np.mean()):.4f}")

        if vmax <= 1e-6:
            print(f"Skipping {pid} (empty volume).")
            continue

        vol_np = gaussian_filter(vol_np, sigma=1.5)
        try:
            verts, faces, _, _ = marching_cubes(vol_np, level=0.5)
        except ValueError:
            # Fallback to lower threshold
            try:
                verts, faces, _, _ = marching_cubes(vol_np, level=0.3 * vmax)
            except ValueError:
                print(f"Skipping {pid} (mesh extraction failed).")
                continue

        fig = go.Figure(data=[go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            colorscale='Hot', intensity=verts[:, 2], showscale=False,
        )])
        fig.update_layout(
            title=f"Patient: {pid} | Label: {label} (V7, faces={faces.shape[0]})",
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.show()

# visualize_random_patients(n=5)


# ════════════════════════════════════════════════════════════════
# SECTION C: PER-PATIENT PROJECTION VALIDATION
# ════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm

from models_v7 import (
    UNet, Encoder2D, Decoder3D, compute_visual_hull,
    render_projection, dice_loss,
)
from train_v7 import build_patient_groups, PatientDataset, REPO_ROOT

def evaluate_all_patients():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path("projection_plots_v7")
    out_dir.mkdir(exist_ok=True)

    unet_ckpt = REPO_ROOT / "UNET_Segmentation" / "breast_segmentation_unet_best_gpu.pth"
    tiff_base = REPO_ROOT / "data" / "organized_by_patient"
    ckpt_path = Path("checkpoints_3d_v7/3dbreastnet_best.pth")

    print("Loading checkpoints...")
    unet = UNet().to(device)
    unet.load_state_dict(torch.load(unet_ckpt, map_location=device, weights_only=False))
    unet.eval()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = Encoder2D().to(device)
    dec = Decoder3D().to(device)
    enc.load_state_dict(ckpt["enc"])
    dec.load_state_dict(ckpt["dec"])
    enc.eval(); dec.eval()

    groups = build_patient_groups(tiff_base)
    if not groups:
        print("No patients found.")
        return
    dataset = PatientDataset(groups, unet, device)

    view_angles = [-90.0, -45.0, 0.0, 45.0, 90.0]
    view_names = ["Right Lateral (-90°)", "Right Oblique (-45°)", "Frontal (0°)",
                  "Left Oblique (45°)", "Left Lateral (90°)"]
    all_dices = []

    print(f"\nEvaluating {len(dataset)} patients...")
    for item in tqdm(dataset):
        pid = item["patient_id"]
        m5 = item["masks_5ch"].unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            hull = compute_visual_hull(m5, device)
            vol = dec(enc(m5), hull).float()

            patient_dices, projections = [], []
            for i, angle in enumerate(view_angles):
                th = torch.tensor([angle], device=device, dtype=torch.float32)
                proj = render_projection(vol, th)
                gt_mask = m5[:, i:i+1]
                dl = dice_loss(proj, gt_mask)
                patient_dices.append(1.0 - dl.item())
                projections.append((proj[0, 0] > 0.5).cpu().numpy().astype(np.float32))

        all_dices.append(np.mean(patient_dices))

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f"Patient: {pid} | Average Dice Score: {np.mean(patient_dices):.4f}", fontsize=16)
        for i in range(5):
            axes[0, i].imshow(m5[0, i].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f"GT: {view_names[i]}"); axes[0, i].axis('off')
            axes[1, i].imshow(projections[i], cmap='gray')
            axes[1, i].set_title(f"Pred (Dice: {patient_dices[i]:.3f})"); axes[1, i].axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / f"{pid}_validation_grid.png", dpi=150)
        plt.close(fig)

    print(f"\n{'='*50}")
    print(f"Overall Mean Dice: {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}")
    print(f"Plots saved to: {out_dir.absolute()}")

# evaluate_all_patients()


# ════════════════════════════════════════════════════════════════
# SECTION D: BATCH STL EXPORT
# ════════════════════════════════════════════════════════════════
import struct
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from models_v7 import UNet, Encoder2D, Decoder3D, compute_visual_hull
from train_v7 import build_patient_groups, PatientDataset, REPO_ROOT

def save_stl(filename, verts, faces):
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

def export_all_stls():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("exported_stls_v7")
    out_dir.mkdir(exist_ok=True)

    unet_ckpt = str(REPO_ROOT / "UNET_Segmentation" / "breast_segmentation_unet_best_gpu.pth")
    tiff_base = str(REPO_ROOT / "data" / "organized_by_patient")
    ckpt_path = Path("checkpoints_3d_v7/3dbreastnet_best.pth")

    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}.")
        return

    print("Loading checkpoints...")
    unet = UNet().to(device)
    unet.load_state_dict(torch.load(unet_ckpt, map_location=device, weights_only=False))
    unet.eval()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = Encoder2D().to(device)
    dec = Decoder3D().to(device)
    enc.load_state_dict(ckpt["enc"])
    dec.load_state_dict(ckpt["dec"])
    enc.eval(); dec.eval()

    groups = build_patient_groups(tiff_base)
    if not groups: return
    dataset = PatientDataset(groups, unet, device)
    print(f"Exporting {len(dataset)} patients to .STL...")

    for item in tqdm(dataset):
        pid = item["patient_id"]
        m5 = item["masks_5ch"].unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            hull = compute_visual_hull(m5, device)
            vol = dec(enc(m5), hull).float()

        vol_np = vol[0, 0].cpu().numpy()
        vol_np = np.pad(vol_np, pad_width=1, mode='constant', constant_values=0)
        vol_np = gaussian_filter(vol_np, sigma=2.0)

        try:
            verts, faces, _, _ = marching_cubes(vol_np, level=0.5)
            verts = verts - 1.0  # compensate for padding
            verts = (verts / 63.5) - 1.0  # normalize to [-1, 1]
            save_stl(out_dir / f"{pid}_3d_geometry.stl", verts, faces)
        except Exception as e:
            print(f"Skipping {pid}: {e}")

    print(f"\nDone! Meshes saved to {out_dir.absolute()}")

# export_all_stls()
