# Project context — 3D breast thermography reconstruction and PINN inverse bioheat solver
## Full pipeline reference document

---

## 1. Project overview

This project reconstructs an anatomically accurate 3D model of the breast from five 2D infrared thermography views, then solves the inverse Pennes Bioheat Equation to localise a subsurface tumour from the resulting surface temperature distribution.

The pipeline has two major stages:

**Stage A — 2D to 3D reconstruction (BreastNet3D v5)**
Five-view infrared TIFF images are segmented by a trained U-Net. The inframammary fold (IMF) centroid is extracted from the frontal mask. An encoder-decoder reconstructs a watertight 3D volume using a differentiable renderer whose rotation pivot is anchored to the IMF centroid — enforcing a shared anatomical coordinate frame across all patients by construction, without post-hoc registration.

**Stage B — Inverse bioheat solving (PINN)**
The anatomically registered STL and the masked thermal projection from all five TIFFs feed into a Physics-Informed Neural Network. The network simultaneously fits the surface temperature boundary condition and enforces the Pennes Bioheat PDE in the interior volume. Tumour parameters are learnable PyTorch scalars optimised jointly with the network weights.

---

## 2. Dataset

- **Cohort:** 122 patients, DMR-IR dataset (Hugging Face: SemilleroCV/DMR-IR)
- **Per patient:**
  - `view_{1..5}.tiff` — 5-angle calibrated infrared images, absolute °C
- **Shared models (loaded once, all patients):**
  - `unet_mask.pth` — trained U-Net weights for breast segmentation
  - `breastnet3d_v5_best.pth` — trained BreastNet3D v5 encoder-decoder weights

### Five-view acquisition protocol

| View | Index | Angle $\theta$ |
|------|-------|----------------|
| Right lateral | 0 | −90° |
| Right oblique | 1 | −45° |
| Frontal | 2 | 0° |
| Left oblique | 3 | +45° |
| Left lateral | 4 | +90° |

The frontal view (index 2) is the anatomical reference view. All IMF extraction uses this view only.

---

## 3. Evolution from v4 to v5

### Why v4 is insufficient

BreastNet3D v4 is geometrically consistent but anatomically unconstrained. Its differentiable renderer rotates the 128³ occupancy volume around the **voxel grid centre** (64, 64, 64) — an arbitrary point with no anatomical meaning. Consequences:

- Each patient's volume lives in its own arbitrary coordinate frame
- Cross-patient comparison of tumour coordinates $(x_t, y_t, z_t)$ is meaningless
- Post-hoc registration is required, introducing additional error

### The single architectural fix in v5

**Change the renderer's rotation pivot from the voxel grid centre to the IMF centroid.**

In v4:
$$V_{\text{rot}} = R_y(\theta) \cdot V$$

In v5:
$$V_{\text{rot}} = R_y(\theta) \cdot (V - \mathbf{p}_{\text{IMF}}) + \mathbf{p}_{\text{IMF}}$$

Translate so IMF is at origin → rotate → translate back. The network is now forced to organise the volume so the IMF sits at the known pivot. The encoder-decoder architecture is completely unchanged. No post-hoc registration is needed.

### What v5 gives you that v4 doesn't

| Property | v4 | v5 |
|---|---|---|
| Geometrically consistent | ✓ | ✓ |
| Rotation around anatomical point | ✗ | ✓ |
| Cross-patient comparable coords | ✗ | ✓ |
| Known rotation axis | ✗ | ✓ Y-axis through IMF |
| Post-hoc registration required | ✓ | ✗ |
| Architecture change from v4 | — | Renderer + IMF loss only |
| Encoder-decoder unchanged | — | ✓ |

---

## 4. Full pipeline

```
unet_mask.pth (loaded once)
       │
       │ inference — frontal TIFF only
       ▼
frontal mask ──► IMF centroid extraction (8 lines numpy)
                        │
                        │ imf_pivot → into renderer
                        │
view_1..5.tiff ──► U-Net inference ──► mask_1..5
                                            │
                              BreastNet3D v5 encoder-decoder
                              pivot-aware differentiable renderer
                              + IMF consistency loss
                                            │
                                     128³ occupancy volume
                                     (anatomically anchored)
                                            │
                                     marching cubes → STL
                                     (in shared anatomical frame)
                                            │
                        ┌───────────────────┘
                        │
           Masked thermal projection (5 TIFFs + U-Net masks)
                        │
              T_measured on STL surface + confidence weights
                        │
           PINN T_θ(x,y,z) + {x_t,y_t,z_t,r_t,Q_max}
                        │
           L_total = λ_data·L_data + λ_pde·L_pde
                        │
           Adam (10k steps) → L-BFGS (500 steps)
                        │
           Output: x_t, y_t, z_t, r_t, Q_max
           (in shared anatomical frame, cross-patient comparable)
```

---

## 5. Stage A1 — U-Net mask generation (per TIFF, at inference)

### Load once

```python
unet = UNet()
unet.load_state_dict(torch.load('unet_mask.pth'))
unet.eval().to(device)
```

### Inference per TIFF

```python
def get_masked_tiff(unet, tiff_path, device):
    tiff = np.array(Image.open(tiff_path)).astype(np.float32)  # [H, W], °C

    t_min, t_max = tiff.min(), tiff.max()
    tiff_norm = (tiff - t_min) / (t_max - t_min + 1e-8)

    x = torch.tensor(tiff_norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = unet(x)
        mask   = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy()

    from scipy.ndimage import binary_erosion
    mask = binary_erosion(mask, iterations=2)   # remove ambiguous boundary pixels

    valid_temps = tiff.copy()
    valid_temps[~mask] = np.nan
    return tiff, mask, valid_temps              # raw °C, binary mask, gated °C
```

---

## 6. Stage A2 — IMF centroid extraction (frontal mask only)

The IMF centroid is the single anatomical reference point used throughout v5. It is extracted from the frontal U-Net mask only — no multi-view dependency, no curve fitting, no cascade failure.

### Why the nipple is not used

Thermal nipple signal is suppressed in dense tissue and post-surgical patients. It is not reliably detectable across the full 122-patient cohort. The IMF bottommost row is always present as long as the U-Net mask is valid.

### Why KPE_Current is not used

KPE_Current.ipynb fails too frequently on complex patient geometries to be reliable across all 122 patients. Its Phase 3 (IMF extraction) is the only part needed — the 8-line implementation below replaces it entirely with no cascade risk.

```python
def extract_imf_centroid_2d(mask_frontal):
    """
    Bottommost row of the breast mask → horizontal midpoint.
    Returns 2D pixel coordinate [col, row].
    Raises ValueError if mask is empty (U-Net failure — flag patient).
    """
    rows = np.where(mask_frontal.any(axis=1))[0]
    if len(rows) == 0:
        raise ValueError("Empty mask — U-Net failed, flag this patient")

    imf_row  = rows.max()
    imf_cols = np.where(mask_frontal[imf_row])[0]
    return np.array([imf_cols.mean(), imf_row])     # [col, row]


def extract_superior_apex_2d(mask_frontal):
    """
    Topmost row of the breast mask → horizontal midpoint.
    Defines +Y (superior) direction together with IMF.
    """
    rows = np.where(mask_frontal.any(axis=1))[0]
    apex_row  = rows.min()
    apex_cols = np.where(mask_frontal[apex_row])[0]
    return np.array([apex_cols.mean(), apex_row])   # [col, row]


def get_imf_volume_coord(mask_frontal, volume_size=128):
    """
    Convert 2D IMF centroid (pixels) to volume voxel coordinates.
    Used as the rotation pivot in the v5 renderer.
    """
    imf_2d = extract_imf_centroid_2d(mask_frontal)
    H, W   = mask_frontal.shape

    imf_y_vol = (imf_2d[1] / H) * volume_size      # vertical   → Y axis
    imf_x_vol = (imf_2d[0] / W) * volume_size      # horizontal → X axis
    imf_z_vol = volume_size / 2                    # depth unknown, use centre

    return torch.tensor([imf_x_vol, imf_y_vol, imf_z_vol], dtype=torch.float32)
```

---

## 7. Stage A3 — BreastNet3D v5 (anatomically anchored reconstruction)

### Architecture (unchanged from v4)

```
Input  : 5-channel mask tensor [5, H, W]
Encoder: 2D CNN → 1000-dimensional latent vector
Decoder: 3D CNN → 128×128×128 occupancy volume [0, 1]
```

The encoder-decoder weights from `breastnet3d_best.pth` (v4) can be used as initialisation. Only the renderer changes.

### Pivot-aware differentiable renderer (the v5 change)

```python
def render_projection_v5(volume, theta_deg, imf_pivot):
    """
    volume    : [D, H, W] occupancy grid, values in [0, 1]
    theta_deg : rotation angle in degrees (from acquisition protocol)
    imf_pivot : [3] IMF location in volume voxel coords
    Returns   : [H, W] differentiable projection
    """
    theta = torch.tensor(theta_deg * np.pi / 180.0)

    # Rotation matrix around Y axis
    R = torch.tensor([
        [ torch.cos(theta), 0, torch.sin(theta)],
        [ 0,                1, 0               ],
        [-torch.sin(theta), 0, torch.cos(theta)]
    ], dtype=torch.float32)

    # All voxel coordinates [D*H*W, 3]
    D, H, W = volume.shape
    coords  = torch.stack(torch.meshgrid(
        torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij'
    ), dim=-1).reshape(-1, 3).float()

    # Translate to IMF → rotate → translate back
    coords_centred = coords - imf_pivot
    coords_rotated = (R @ coords_centred.T).T
    coords_final   = coords_rotated + imf_pivot

    # Resample volume at rotated coordinates (grid_sample)
    # Normalise coords to [-1, 1] for grid_sample
    coords_norm = coords_final / torch.tensor([D, H, W], dtype=torch.float32) * 2 - 1
    coords_norm = coords_norm.reshape(1, D, H, W, 3)[..., [2, 1, 0]]  # DHW → WHD

    V_rot = torch.nn.functional.grid_sample(
        volume.unsqueeze(0).unsqueeze(0),
        coords_norm,
        align_corners=True,
        mode='bilinear',
        padding_mode='zeros'
    ).squeeze().reshape(D, H, W)

    # Differentiable visual-hull projection (same as v4)
    projection = 1 - torch.exp(-V_rot.sum(dim=0))  # [H, W]
    return projection
```

### IMF consistency loss (new in v5)

Forces the projected bottom of the frontal view to match the detected IMF centroid:

```python
def imf_consistency_loss(volume, imf_2d_gt, imf_pivot):
    """
    Project volume into frontal view (theta=0).
    Penalise distance between predicted and detected IMF centroid.
    imf_2d_gt : [2] detected IMF in pixel coords [col, row]
    """
    proj = render_projection_v5(volume, theta_deg=0.0, imf_pivot=imf_pivot)
    H, W = proj.shape

    occupied = (proj > 0.5).float()
    rows_idx = torch.arange(H, device=volume.device).float()
    cols_idx = torch.arange(W, device=volume.device).float()

    total = occupied.sum() + 1e-8
    imf_row_pred = (occupied * rows_idx.unsqueeze(1)).sum() / total
    imf_col_pred = (occupied * cols_idx.unsqueeze(0)).sum() / total

    imf_pred = torch.stack([imf_col_pred, imf_row_pred])
    imf_gt   = torch.tensor(imf_2d_gt, dtype=torch.float32, device=volume.device)

    return torch.nn.functional.mse_loss(imf_pred, imf_gt)
```

### Total training loss (v5)

$$\mathcal{L}_{\text{v5}} = \mathcal{L}_{\text{proj}} + \lambda_{\text{IMF}}\,\mathcal{L}_{\text{IMF}}$$

```python
lambda_imf  = 0.1    # start here — enough to anchor without overpowering projection

loss = loss_proj + lambda_imf * loss_imf
```

$\lambda_{\text{IMF}} = 0.1$ is sufficient — the projection loss does the heavy geometric lifting. The IMF loss only pins the coordinate system. The Dice score target remains ~0.84 or above.

### Training procedure

- Initialise from `breastnet3d_best.pth` (v4 weights) — only the renderer changes, so v4 weights are a valid warm start
- Fine-tune with the pivot-aware renderer and IMF loss active from epoch 0
- All other training hyperparameters (batch size, learning rate, FP32 enforcement) unchanged from v4

### Output per patient

After training, inference produces:
- 128³ occupancy volume in the shared anatomical frame
- Marching cubes → watertight STL, also in the shared frame
- The STL coordinate origin corresponds to the IMF centroid
- $+Y$ axis is superior, $-Z$ axis is into the chest wall

---

## 8. Stage A4 — STL validation and unit check

```python
def load_and_validate_stl(stl_path):
    mesh = trimesh.load(stl_path)

    # Unit check — real breast half-width ~60-80 mm
    extents = mesh.bounding_box.extents
    if extents.max() < 1.0:
        mesh.apply_scale(1000.0)    # metres → mm
    elif extents.max() < 10.0:
        mesh.apply_scale(100.0)     # cm → mm

    assert 80 < extents.max() < 300, \
        f"Suspicious extents after scaling: {extents}"

    assert mesh.is_watertight, "Mesh is not watertight"

    if mesh.volume < 0:
        trimesh.repair.fix_normals(mesh)

    return mesh
```

Because v5 produces STLs already in the anatomical frame, no further registration transform is needed. `load_and_validate_stl` replaces the old `load_mesh_mm` + `register_to_imf` two-step.

---

## 9. Stage A5 — Masked thermal projection onto STL surface

### Why masking matters

Without the U-Net mask, projection near the breast boundary reads pixels blending breast skin with ambient air (~20 °C). Even a few contaminated vertices in $T_{\text{measured}}$ create a physically inconsistent boundary condition — the PINN wastes capacity on impossible temperature gradients and tumour parameters drift to compensate.

### Visibility test (ray–mesh intersection)

For each STL surface vertex $\mathbf{p}_i$ and camera view $v$:
- Cast ray from $\mathbf{p}_i$ toward camera origin $\mathbf{c}_v$
- Self-occlusion test via `trimesh.ray.intersects_location`
- Visible if no intersection found before the camera

### Temperature readout (mask-gated)

For each visible vertex under view $v$:
1. Project $\mathbf{p}_i$ to pixel $(u, v) = \mathbf{P}_v \mathbf{p}_i$
2. Check `valid_temps[u, v]` — if `NaN` (outside U-Net mask), discard
3. Otherwise bilinearly interpolate and record $T_v(\mathbf{p}_i)$

### Multi-view fusion

$$T_{\text{measured}}(\mathbf{p}_i) = \frac{1}{|\mathcal{V}_i|} \sum_{v \in \mathcal{V}_i} T_v(\mathbf{p}_i)$$

$\mathcal{V}_i$ = views where vertex $i$ is both geometrically visible and within the U-Net mask.

### Confidence weights (three-tier)

$$w_i = \begin{cases} 1.0 & \text{directly projected, low inter-view variance} \\ 0.5 & \text{directly projected, high inter-view variance} \\ 0.3 & \text{KNN filled (}|\mathcal{V}_i| = 0\text{)} \end{cases}$$

KNN gap-fill: $K=5$ neighbours from already-projected vertices.

**Output:**
```
surface_pts  : np.ndarray  [N_s, 3]  — STL vertex coords, mm (anatomical frame)
T_measured   : np.ndarray  [N_s]     — surface temperature, °C
confidence   : np.ndarray  [N_s]     — per-vertex weight {1.0, 0.5, 0.3}
```

---

## 10. Stage B1 — Domain sampling

**Surface:** Subsample $N_s = 2000$ STL vertices per batch. Normalise to $[-1, 1]^3$ using mesh bounding box.

**Interior collocation:** $N_v = 5000$ random interior points per batch:
```python
interior_pts = trimesh.sample.volume_mesh(mesh, count=5000)  # [N_v, 3], mm
```
Works on the hollow watertight shell — `trimesh` uses ray parity to determine interior. Same bounding box normalisation applied.

---

## 11. Stage B2 — PINN architecture

### Network

```
Input:   (x, y, z)    — 3 neurons, normalised to [-1, 1] from mm bounding box
Hidden:  6 × 256      — tanh activation (mandatory — ReLU kills ∂²T/∂x²)
Output:  T_θ(x,y,z)  — 1 neuron, linear
```

```python
class BioheatPINN(nn.Module):
    def __init__(self, hidden=256, depth=6):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

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
```

### Biophysical constants (breast tissue, SI units)

| Symbol | Value | Units | Description |
|--------|-------|-------|-------------|
| $k$ | 0.48 | W/(m·K) | Thermal conductivity |
| $\omega_b$ | 0.0005 | m³/(m³·s) | Blood perfusion rate |
| $c_b$ | 3600 | J/(kg·K) | Specific heat of blood |
| $T_a$ | 37.0 | °C | Arterial blood temperature |
| $Q_m$ | 450 | W/m³ | Healthy metabolic heat generation |

---

## 12. Stage B3 — Loss functions

### Unit handling

Network operates on normalised $[-1,1]$ coords. PDE residual requires SI units:

```python
COORD_SCALE = 1e-3   # mm → m, applied only inside PDE residual
```

### Pennes Bioheat equation (steady-state)

$$k\,\nabla^2 T + \omega_b c_b (T_a - T) + Q_m + Q_{\text{tumor}}(x,y,z) = 0$$

### Tumour heat source (Gaussian)

$$Q_{\text{tumor}} = Q_{\max} \cdot \exp\!\left(-\frac{(x-x_t)^2+(y-y_t)^2+(z-z_t)^2}{r_t^2}\right)$$

### Data loss

$$\mathcal{L}_{\text{data}} = \frac{1}{N_s}\sum_{i=1}^{N_s} w_i\,\big|T_\theta(\mathbf{p}_i) - T_{\text{measured},i}\big|^2$$

### Physics loss

$$f = k\,\nabla^2 T_\theta + \omega_b c_b (T_a - T_\theta) + Q_m + Q_{\text{tumor}}$$

$$\mathcal{L}_{\text{PDE}} = \frac{1}{N_v}\sum_{j=1}^{N_v} \big|f(x_j,y_j,z_j)\big|^2$$

### Laplacian via autograd

```python
def laplacian(T, xyz):
    dT = torch.autograd.grad(
        T, xyz, grad_outputs=torch.ones_like(T), create_graph=True
    )[0]
    return sum(
        torch.autograd.grad(
            dT[:,i], xyz,
            grad_outputs=torch.ones_like(dT[:,i]),
            create_graph=True
        )[0][:,i]
        for i in range(3)
    )
```

### Loss weighting (static, epoch 0)

```python
lambda_data = 1.0
lambda_pde  = (L_data_0 / L_pde_0).detach()
```

$$\mathcal{L}_{\text{total}} = \lambda_{\text{data}}\,\mathcal{L}_{\text{data}} + \lambda_{\text{PDE}}\,\mathcal{L}_{\text{PDE}}$$

---

## 13. Stage B4 — Training schedule

| Phase | Optimiser | Steps | Purpose |
|-------|-----------|-------|---------|
| 1 | Adam, lr=1e-3 | 10,000 | Broad exploration, moves tumour params |
| 2 | L-BFGS, strong Wolfe | 500 | Fine convergence, drives residual to zero |

**Multi-start:** 5 random initialisations of $(x_t, y_t, z_t)$ from inside the mesh. Keep lowest final $\mathcal{L}_{\text{total}}$.

---

## 14. Stage B5 — Output per patient

```python
results = {
    'x_t_mm'  : (model.x_t.item() + 1) / 2 * bbox_extents_mm[0],
    'y_t_mm'  : (model.y_t.item() + 1) / 2 * bbox_extents_mm[1],
    'z_t_mm'  : (model.z_t.item() + 1) / 2 * bbox_extents_mm[2],
    'r_t_mm'  : model.r_t.item(),
    'Q_max'   : model.Q_max.item(),
    'quadrant': classify_quadrant(x_t_mm, y_t_mm),
}
```

Because all patients share the same anatomical frame (origin = IMF centroid, +Y = superior), tumour coordinates are directly cross-patient comparable and map to clinical quadrant notation:

| Sign | Anatomical meaning |
|---|---|
| $x_t > 0$ | Lateral quadrant |
| $x_t < 0$ | Medial quadrant |
| $y_t > 0$ | Upper quadrant |
| $y_t < 0$ | Lower quadrant |
| $z_t$ | Depth from IMF plane |

Estimated tumour volume: $V \approx \frac{4}{3}\pi r_t^3$

---

## 15. Full dataset iteration

```python
unet = UNet()
unet.load_state_dict(torch.load('unet_mask.pth'))
unet.eval().to(device)

breastnet = BreastNet3D_v5()
breastnet.load_state_dict(torch.load('breastnet3d_v5_best.pth'))
breastnet.eval().to(device)

for patient_dir in Path('dmr_ir/').glob('patient_*/'):
    tiff_paths   = sorted(patient_dir.glob('view_*.tiff'))
    frontal_tiff = tiff_paths[2]                        # index 2 = 0° frontal

    # A1 — masks (generated per TIFF, not stored)
    _, frontal_mask, _ = get_masked_tiff(unet, frontal_tiff, device)
    all_masks = [get_masked_tiff(unet, t, device)[1] for t in tiff_paths]

    # A2 — IMF pivot (from frontal mask only)
    imf_pivot = get_imf_volume_coord(frontal_mask, volume_size=128).to(device)

    # A3 — BreastNet3D v5 inference
    mask_tensor = torch.stack([
        torch.tensor(m, dtype=torch.float32) for m in all_masks
    ]).unsqueeze(0).to(device)                          # [1, 5, H, W]

    with torch.no_grad():
        volume = breastnet(mask_tensor, imf_pivot)      # [128, 128, 128]

    # STL export
    stl_path = patient_dir / 'mesh_v5.stl'
    export_stl(volume, stl_path)

    # A4 — validate STL
    mesh = load_and_validate_stl(stl_path)

    # A5 — masked thermal projection
    surface_pts, T_measured, confidence = project_thermal(
        unet, mesh, tiff_paths, device
    )

    # B1 — interior sampling
    interior_pts = trimesh.sample.volume_mesh(mesh, count=5000)

    # B2–B4 — train PINN
    model   = BioheatPINN()
    results = train_pinn(
        model, surface_pts, T_measured,
        confidence, interior_pts,
        mesh.bounding_box.extents
    )

    save_results(patient_dir, results)
```

---

## 16. Known failure modes

| Risk | Symptom | Fix |
|------|---------|-----|
| Empty frontal mask | IMF extraction raises ValueError | Flag patient, inspect U-Net output |
| Wrong STL units | $r_t$ returns ~0.003 not ~3 mm | `load_and_validate_stl` with assertion |
| Inverted mesh normals | `volume_mesh` samples outside breast | Check `mesh.volume > 0`, run `fix_normals` |
| $\lambda_{\text{IMF}}$ too large | Projection loss overwhelmed, low Dice | Reduce to 0.05, monitor Dice during training |
| Background pixels in projection | Noisy $T_{\text{measured}}$ at breast edge | U-Net mask + 2px erosion before readout |
| Loss scale mismatch (PINN) | PDE dominates, surface fit poor | Static $\lambda$ normalisation at epoch 0 |
| Local minima in tumour location | Result at mesh edge | Multi-start (5 random inits) |
| $r_t$ collapses to zero | $Q_{\max}$ diverges | `r_t.data.clamp_(min=2.0)` every step |
| Autograd OOM | CUDA out of memory | Reduce collocation batch to 2000 |

---

## 17. Thesis scientific claim

> *BreastNet3D v5 introduces anatomically-anchored self-supervised 3D breast reconstruction by fixing the differentiable renderer's rotation pivot to the inframammary fold centroid, extracted from the frontal segmentation mask at training time. This enforces a shared anatomical coordinate frame across all 122 patients without additional annotations, post-hoc registration, or architectural changes to the encoder-decoder. The recovered tumour parameters from the subsequent PINN inverse bioheat solver are directly cross-patient comparable and map to the clinical breast quadrant system.*

---

## 18. Literature (verified DOIs)

1. Mukhmetov O. et al. (2023) — PINN for breast bioheat, forward solver, 12× faster than FEM
   *Computer Methods and Programs in Biomedicine*, 242, 107834
   `doi:10.1016/j.cmpb.2023.107834`

2. Karniadakis G.E. et al. (2021) — Physics-informed machine learning, foundational PINN inverse problem framework
   *Nature Reviews Physics*, 3(6), 422–440
   `doi:10.1038/s42254-021-00314-5`

3. Wang S., Yu X., Perdikaris P. (2022) — When and why PINNs fail, NTK-based adaptive loss weighting
   *Journal of Computational Physics*, 449, 110768
   `doi:10.1016/j.jcp.2021.110768`

4. Majdoubi J. et al. (2021) — Feedforward NN inverse bioheat, tumour location and radius from surface temperatures
   *Computer Methods and Programs in Biomedicine*, 106092
   `doi:10.1016/j.cmpb.2021.106092`

5. Perez-Raya I., Gutierrez C., Kandlikar S.G. (2024) — Inverse PINN breast cancer detection, biopsy-validated
   *ASME Journal of Heat and Mass Transfer*, 146(10), 101201
   `doi:10.1115/1.4065673`
