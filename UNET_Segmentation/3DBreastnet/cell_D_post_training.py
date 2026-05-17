###############################################################################
# CELL D — View Training Metrics (keep after training cell)
# Works with BOTH old checkpoints and new U-Net checkpoints
###############################################################################

%matplotlib inline
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Try new U-Net checkpoint first, fall back to old
ckpt_path = Path("checkpoints_3d_v6/3dbreastnet_unet_best.pth")
if not ckpt_path.exists():
    ckpt_path = Path("checkpoints_3d_v6/3dbreastnet_best.pth")
if not ckpt_path.exists():
    ckpt_path = Path("checkpoints_3d_v5/3dbreastnet_best.pth")

if ckpt_path.exists():
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hist = ckpt["hist"]
    # Handle both old and new history key names
    ep = hist.get("epoch", list(range(1, len(hist.get("train_loss", hist.get("train_dice_loss", [])))+1)))
    t_loss = hist.get("train_dice_loss", hist.get("train_loss", []))
    v_loss = hist.get("val_dice_loss", hist.get("val_loss", []))
    v_dice = hist.get("val_dice_score", hist.get("val_dice", []))
    v_hd   = hist.get("val_hd95", hist.get("val_hd", []))
    p_loss = hist.get("prior_loss", [0]*len(ep))

    n = min(3 if all(x == 0 for x in p_loss) else 4, 4)
    fig, ax = plt.subplots(1, n, figsize=(6*n, 5))
    ax[0].plot(ep, t_loss, label="train"); ax[0].plot(ep, v_loss, label="val")
    ax[0].set_title("Dice Loss"); ax[0].legend()
    ax[1].plot(ep, v_dice, color="green"); ax[1].set_title("Val Dice Score")
    ax[2].plot(ep, v_hd, color="red"); ax[2].set_title("Val HD95")
    if n == 4:
        ax[3].plot(ep, p_loss, color="purple"); ax[3].set_title("Prior Loss")
    for a in ax: a.set_xlabel("Epoch")
    plt.tight_layout(); plt.show()
    print(f"Best dice: {ckpt.get('best_dice', max(v_dice)):.4f}")
else:
    print(f"No checkpoint found.")


###############################################################################
# CELL E — 3D Visualization (adapted for BreastNet3D_UNet)
# Loads the combined model instead of separate enc/dec
###############################################################################

import random, math
import numpy as np
from IPython.display import display
import torch, torch.nn.functional as F
from pathlib import Path
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

# Seamless imports
from models_v6_upgraded import (
    UNet,
    Encoder2D,
    Decoder3D,
    BreastNet3D_UNet,
    compute_visual_hull
)
from cell_C_training import build_patient_groups, PatientDataset  # Reuse dataset

def visualize_random_patients_local(n=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    unet_ckpt = r"/mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/UNET_Segmentation/breast_segmentation_unet_best_gpu.pth"
    tiff_base = r"/mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/data/organized_by_patient"

    # Try U-Net checkpoint first
    ckpt_path = Path("checkpoints_3d_v6/3dbreastnet_unet_best.pth")
    use_unet_model = True
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints_3d_v5/3dbreastnet_best.pth")
        use_unet_model = False
    if not ckpt_path.exists():
        print(f"Error: No checkpoint found."); return

    print("Loading checkpoints...")
    unet = UNet().to(device)
    unet.load_state_dict(torch.load(unet_ckpt, map_location=device, weights_only=False))
    unet.eval()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if use_unet_model:
        model = BreastNet3D_UNet().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
    else:
        enc = Encoder2D().to(device); dec = Decoder3D().to(device)
        enc.load_state_dict(ckpt["enc"]); dec.load_state_dict(ckpt["dec"], strict=False)
        enc.eval(); dec.eval()

    groups = build_patient_groups(tiff_base)
    if not groups: print("No patients found."); return
    selected = random.sample(groups, min(n, len(groups)))
    dataset = PatientDataset(selected, unet, device)
    print(f"Generating 3D models for {len(selected)} patients...")

    for i in range(len(dataset)):
        item = dataset[i]
        pid, label = item["patient_id"], item["label"]
        m5 = item["masks_5ch"].unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            hull = compute_visual_hull(m5, device)
            if use_unet_model:
                vol = model(m5, hull)
            else:
                try: vol = dec(enc(m5), hull)
                except TypeError: vol = dec(enc(m5))

        vol_np = vol[0, 0].float().cpu().numpy()
        vmin, vmax = float(vol_np.min()), float(vol_np.max())
        print(f"{pid}: min={vmin:.4f} max={vmax:.4f} mean={float(vol_np.mean()):.4f}")
        if vmax <= 1e-6:
            print(f"Empty volume for {pid}."); continue
        vol_np = gaussian_filter(vol_np, sigma=1.5)
        levels = [0.12*vmax, 0.18*vmax, 0.25*vmax,
                  float(np.percentile(vol_np,80)), float(np.percentile(vol_np,90)),
                  float(np.percentile(vol_np,95)), float(np.percentile(vol_np,98)), 0.6*vmax]
        levels = [lv for lv in levels if vmin < lv < vmax]
        if not levels: print(f"No valid level for {pid}."); continue
        verts, faces, best_faces = None, None, -1
        for level in levels:
            try:
                v, f, _, _ = marching_cubes(vol_np, level=level)
                if f.shape[0] > best_faces: verts, faces, best_faces = v, f, f.shape[0]
                if f.shape[0] >= 2000: break
            except ValueError: continue
        if verts is None: print(f"No mesh for {pid}."); continue
        fig = go.Figure(data=[go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2],
            i=faces[:,0], j=faces[:,1], k=faces[:,2],
            colorscale='Hot', intensity=verts[:,2], showscale=False)])
        fig.update_layout(title=f"{pid} | {label} (faces={faces.shape[0]})",
                          scene=dict(aspectmode='data'), margin=dict(l=0,r=0,b=0,t=40))
        fig.show()

visualize_random_patients_local(n=5)


###############################################################################
# CELL F — Evaluation Metrics (NEW)
###############################################################################

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from scipy.spatial.distance import directed_hausdorff

# Seamless imports
from models_v6_upgraded import (
    UNet,
    Encoder2D,
    Decoder3D,
    BreastNet3D_UNet,
    compute_visual_hull,
    render_projection
)
from cell_C_training import build_patient_groups, PatientDataset  # Reuse dataset

def compute_metrics(pred_proj, target_sil, threshold=0.5):
    pred_bin = (pred_proj >= threshold).astype(np.float32)
    tgt = target_sil.astype(np.float32)
    TP = (pred_bin * tgt).sum()
    TN = ((1 - pred_bin) * (1 - tgt)).sum()
    FP = (pred_bin * (1 - tgt)).sum()
    FN = ((1 - pred_bin) * tgt).sum()
    accuracy = (TP + TN) / pred_bin.size
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    jaccard = TP / (TP + FP + FN + 1e-6)
    p_pts = np.argwhere(pred_bin > 0).astype(float)
    t_pts = np.argwhere(tgt > 0).astype(float)
    if len(p_pts) == 0 or len(t_pts) == 0:
        hd = float('nan')
    else:
        hd = max(directed_hausdorff(p_pts, t_pts)[0], directed_hausdorff(t_pts, p_pts)[0])
    return {'Accuracy': accuracy, 'Dice Index': dice, 'Jaccard Index': jaccard,
            'Hausdorff Distance': hd}

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_ckpt = r"/mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/UNET_Segmentation/breast_segmentation_unet_best_gpu.pth"
    tiff_base = r"/mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/data/organized_by_patient"

    ckpt_path = Path("checkpoints_3d_v6/3dbreastnet_unet_best.pth")
    use_unet_model = True
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints_3d_v5/3dbreastnet_best.pth")
        use_unet_model = False
    if not ckpt_path.exists():
        print("No checkpoint found."); return

    unet = UNet().to(device)
    unet.load_state_dict(torch.load(unet_ckpt, map_location=device, weights_only=False))
    unet.eval()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if use_unet_model:
        model = BreastNet3D_UNet().to(device)
        model.load_state_dict(ckpt["model"]); model.eval()
    else:
        enc = Encoder2D().to(device); dec = Decoder3D().to(device)
        enc.load_state_dict(ckpt["enc"]); dec.load_state_dict(ckpt["dec"], strict=False)
        enc.eval(); dec.eval()

    groups = build_patient_groups(tiff_base)
    if not groups: return
    dataset = PatientDataset(groups, unet, device)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    view_names = ['RL (−90°)', 'RO (−45°)', 'F (0°)', 'LO (+45°)', 'LL (+90°)']
    std_angles = [-90, -45, 0, 45, 90]
    results_per_view = {v: [] for v in view_names}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            masks = batch["masks_5ch"].to(device)
            hull = compute_visual_hull(masks, device)
            if use_unet_model:
                V_pred = model(masks, hull)
            else:
                V_pred = dec(enc(masks), hull)
            V_pred = V_pred.float()
            for vi, (vn, ang) in enumerate(zip(view_names, std_angles)):
                proj = render_projection(V_pred, ang)
                proj_np = proj.squeeze(1).cpu().numpy()
                sil_np = masks[:, vi].cpu().numpy()
                for b in range(proj_np.shape[0]):
                    results_per_view[vn].append(compute_metrics(proj_np[b], sil_np[b]))

    mk = ['Accuracy', 'Dice Index', 'Jaccard Index', 'Hausdorff Distance']
    print(f"\n{'View':<15} {'Accuracy':>10} {'Dice':>10} {'Jaccard':>10} {'Hausdorff':>12}")
    print("-" * 60)
    all_v = {k: [] for k in mk}
    for vn in view_names:
        row = {k: np.nanmean([v[k] for v in results_per_view[vn]]) for k in mk}
        for k in mk: all_v[k].extend([v[k] for v in results_per_view[vn]])
        print(f"{vn:<15} {row['Accuracy']:>10.4f} {row['Dice Index']:>10.4f} "
              f"{row['Jaccard Index']:>10.4f} {row['Hausdorff Distance']:>12.4f}")
    print("-" * 60)
    ov = {k: np.nanmean(all_v[k]) for k in mk}
    print(f"{'Overall':<15} {ov['Accuracy']:>10.4f} {ov['Dice Index']:>10.4f} "
          f"{ov['Jaccard Index']:>10.4f} {ov['Hausdorff Distance']:>12.4f}")

run_evaluation()


###############################################################################
# CELL G — Thermal Projection (LAST cell — run after checking 3D model)
###############################################################################

import torch
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

# Seamless imports
from models_v6_upgraded import render_projection

def estimate_view_angle(V_pred_np, sil_np, angle_range=(-90, 90), step=1):
    best_angle, best_loss = 0, float('inf')
    V_t = torch.from_numpy(V_pred_np).unsqueeze(0).unsqueeze(0).float()
    for angle in np.arange(angle_range[0], angle_range[1] + step, step):
        proj = render_projection(V_t, float(angle)).squeeze().numpy()
        inter = (proj * sil_np).sum()
        loss = 1 - (2 * inter) / (proj.sum() + sil_np.sum() + 1e-6)
        if loss < best_loss: best_loss, best_angle = loss, angle
    return best_angle

def overlay_temperatures_on_volume(V_pred_np, thermal_images, estimated_angles, volume_size=128):
    D, H, W = V_pred_np.shape
    temp_vol = np.full((D, H, W), np.nan, dtype=np.float32)
    count_vol = np.zeros((D, H, W), dtype=np.float32)
    vox_coords = np.argwhere(V_pred_np > 0.5).astype(np.float32)
    for angle, temp_2d in zip(estimated_angles, thermal_images):
        if temp_2d is None: continue
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        centre = np.array([D/2, H/2, W/2])
        shifted = vox_coords - centre
        d, h, w = shifted[:,0], shifted[:,1], shifted[:,2]
        w_rot = w*cos_t + d*sin_t; d_rot = -w*sin_t + d*cos_t
        front = d_rot > 0
        if front.sum() == 0: continue
        h_pix = (h[front] + centre[1]).astype(int)
        w_pix = (w_rot[front] + centre[2]).astype(int)
        th, tw = temp_2d.shape
        h_img = np.clip((h_pix * th / H).astype(int), 0, th-1)
        w_img = np.clip((w_pix * tw / W).astype(int), 0, tw-1)
        oc = np.argwhere(V_pred_np > 0.5)[front]
        for i, (vd, vh, vw) in enumerate(oc):
            t = temp_2d[h_img[i], w_img[i]]
            if np.isnan(t): continue
            if np.isnan(temp_vol[vd,vh,vw]):
                temp_vol[vd,vh,vw] = t; count_vol[vd,vh,vw] = 1
            else:
                temp_vol[vd,vh,vw] = (temp_vol[vd,vh,vw]*count_vol[vd,vh,vw]+t)
                count_vol[vd,vh,vw] += 1
                temp_vol[vd,vh,vw] /= count_vol[vd,vh,vw]
    occ = ~np.isnan(temp_vol) & (V_pred_np > 0.5)
    unocc = np.isnan(temp_vol) & (V_pred_np > 0.5)
    if occ.sum() > 0 and unocc.sum() > 0:
        tree = cKDTree(np.argwhere(occ).astype(np.float32))
        _, idx = tree.query(np.argwhere(unocc).astype(np.float32), k=1)
        temp_vol[unocc] = temp_vol[occ][idx]
    return temp_vol

# ── Usage example (uncomment and adapt to your patient loop) ──
# patient_id = 'example_patient_id'
# views_order = ['RL', 'RO', 'F', 'LO', 'LL']
# std_angles = [-90, -45, 0, 45, 90]
# V_pred_np = np.load(f'outputs/{patient_id}_volume_soft.npy')
# V_bin = (V_pred_np > 0.5).astype(np.float32)
# silhouettes = [...]   # 5 binary (128,128) arrays
# thermal_abs = [...]   # 5 float arrays in °C (raw .tiff values)
# estimated_angles = []
# for vi in range(5):
#     ae = estimate_view_angle(V_bin, silhouettes[vi],
#                              angle_range=(std_angles[vi]-20, std_angles[vi]+20))
#     estimated_angles.append(ae)
#     print(f"View {views_order[vi]}: estimated angle = {ae:.1f}°")
# temp_vol = overlay_temperatures_on_volume(V_bin, thermal_abs, estimated_angles)
# Path('outputs').mkdir(exist_ok=True)
# np.save(f'outputs/{patient_id}_thermal_overlay.npy', temp_vol)
# print(f"Saved: {temp_vol.shape}, range [{np.nanmin(temp_vol):.1f}, {np.nanmax(temp_vol):.1f}] °C")
