# Project context — PINN bioheat solver for 3D breast thermography

## What this document is

A compact record of all design decisions, architecture choices, and confirmed pipeline steps for the inverse Pennes Bioheat PINN. Use this as the single source of truth when implementing.

---

## Dataset

- **Cohort:** 122 patients from the DMR-IR dataset
- **Per patient:**
  - `mesh.stl` — 3D breast surface mesh, watertight hollow shell reconstructed from U-Net segmentation
  - `view_{1..5}.tiff` — 5-angle calibrated thermal images, absolute °C
- **Shared model (one file, all patients):**
  - `unet_mask.pth` — trained U-Net weights for breast segmentation

The U-Net is loaded **once** at pipeline start and run at inference time on each TIFF to generate a breast mask on the fly. Masks are not stored — they are produced per TIFF as needed.

---

## STL geometry — important properties

### Watertight hollow shell

The STL meshes are **watertight hollow shells** — this is correct and expected. The U-Net segments the breast boundary in each 2D view. The 3D reconstruction stitches those boundaries into a closed surface shell with empty interior. This is not a defect.

Watertight (closed manifold, no holes) is the only property the pipeline requires. `trimesh` determines interior/exterior via ray parity against the shell — it does not need a solid mesh.

```python
mesh = trimesh.load('patient_01.stl')
print(mesh.is_watertight)   # must be True
print(mesh.volume)          # must be positive — negative means inverted normals

# Fix inverted normals if volume is negative
trimesh.repair.fix_normals(mesh)
```

### Unit scaling

STL export units are inconsistent across the dataset. At least one mesh has been confirmed at ~6.2 mm midpoint-to-edge, indicating **metre-scale export** (real breast half-width ~60–80 mm). All meshes must be normalised to millimetres at load time.

```python
def load_mesh_mm(path):
    mesh = trimesh.load(path)
    extents = mesh.bounding_box.extents

    if extents.max() < 1.0:
        mesh.apply_scale(1000.0)    # metres → mm
    elif extents.max() < 10.0:
        mesh.apply_scale(100.0)     # cm → mm
    # else already mm

    assert 80 < extents.max() < 300, \
        f"Suspicious bounding box after scaling: {extents} — check {path}"
    return mesh
```

Run this check across all 122 patients before training. Unit mismatches are often inconsistent within a dataset if meshes came from different export pipelines.

---

## Problem statement

Given only the surface temperature map of a patient's breast, recover the 3D location, size, and heat generation rate of a subsurface tumour by enforcing the Pennes Bioheat PDE everywhere inside the volume.

Mesh-free inverse PDE problem solved entirely in PyTorch — no FEM, no COMSOL.

**PINN output (5 numbers per patient):**

| Parameter | Symbol | Units | Meaning |
|-----------|--------|-------|---------|
| Tumour centroid x | $x_t$ | mm | Location |
| Tumour centroid y | $y_t$ | mm | Location |
| Tumour centroid z | $z_t$ | mm | Location |
| Tumour radius | $r_t$ | mm | Effective Gaussian radius |
| Peak heat generation | $Q_{\max}$ | W/m³ | Tumour metabolic activity |

---

## Pipeline overview

```
unet_mask.pth (loaded once)
       │
       │ inference on each TIFF
       ▼
view_1..5.tiff ──► U-Net ──► mask_1..5 ──► valid breast pixels
                                                    │
mesh.stl ──► load_mesh_mm() ──► surface sampling ───┼──► T_measured on surface
                            └──► interior sampling   │
                                        │            │
                                        └─────┬──────┘
                                              ▼
                               PINN T_θ(x,y,z) + {x_t,y_t,z_t,r_t,Q_max}
                                              │
                               L_total = λ_data·L_data + λ_pde·L_pde
                                              │
                               Adam (10k steps) → L-BFGS (500 steps)
                                              │
                               Output: x_t, y_t, z_t, r_t, Q_max
```

---

## Stage 1 — U-Net mask generation per TIFF

### Load once at pipeline start

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

    x = torch.tensor(tiff_norm).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

    with torch.no_grad():
        logits = unet(x)
        mask   = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy()  # [H,W] bool

    # Erode by 2px — U-Net confidence is lowest at boundary
    from scipy.ndimage import binary_erosion
    mask = binary_erosion(mask, iterations=2)

    valid_temps = tiff.copy()
    valid_temps[~mask] = np.nan     # background → invalid
    return valid_temps              # [H, W], NaN outside breast
```

---

## Stage 2 — Masked thermal projection onto STL surface

### Why masking matters

Without the mask, projection near the breast boundary reads pixels blending breast skin with ambient air (~20 °C). Contaminated vertices in $T_{\text{measured}}$ create a boundary condition that physically contradicts the interior — the PINN wastes capacity on an impossible gradient and tumour parameters drift to compensate.

### Visibility test

For each STL surface vertex $\mathbf{p}_i$ and camera view $v$:
- Ray from $\mathbf{p}_i$ toward camera origin $\mathbf{c}_v$
- Self-occlusion test via `trimesh.ray.intersects_location`
- Visible if no intersection found before the camera

### Temperature readout (mask-gated)

For each visible vertex under view $v$:
1. Project to pixel $(u, v) = \mathbf{P}_v \mathbf{p}_i$
2. Read `valid_temps[u, v]` — if `NaN`, discard this view for this vertex
3. Otherwise bilinearly interpolate and record $T_v(\mathbf{p}_i)$

### Multi-view fusion

$$T_{\text{measured}}(\mathbf{p}_i) = \frac{1}{|\mathcal{V}_i|} \sum_{v \in \mathcal{V}_i} T_v(\mathbf{p}_i)$$

$\mathcal{V}_i$ = views where vertex $i$ is both geometrically visible and within the U-Net mask.

### KNN gap-fill

Vertices with $|\mathcal{V}_i| = 0$: fill by KNN ($K=5$) from projected neighbours.
Assign confidence weight $w_i = 0.3$ (vs $w_i = 1.0$ for directly projected vertices).

**Output:**
```
surface_pts  : np.ndarray  [N_s, 3]  — STL vertex coords, mm
T_measured   : np.ndarray  [N_s]     — surface temperature, °C
confidence   : np.ndarray  [N_s]     — 1.0 (projected) or 0.3 (KNN filled)
```

---

## Stage 3 — Domain sampling

**Surface:** Subsample $N_s = 2000$ STL vertices per batch. Normalise to $[-1, 1]^3$ using mesh bounding box (in mm).

**Interior collocation:** $N_v = 5000$ random interior points per batch:
```python
interior_pts = trimesh.sample.volume_mesh(mesh, count=5000)  # [N_v, 3], mm
```
Same bounding box normalisation. Works correctly on the hollow watertight shell — `trimesh` uses ray parity against the shell to determine interior.

---

## Stage 4 — PINN architecture

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

## Stage 5 — Loss functions

### Unit handling

The network operates on normalised coordinates derived from the mm bounding box. The PDE residual requires SI units. Apply conversion only inside the residual:

```python
COORD_SCALE = 1e-3   # mm → m, used only in PDE evaluation

def pde_residual(model, xyz_norm, bbox_extents_mm):
    # Denormalise from [-1,1] back to mm, then convert to m
    xyz_mm = (xyz_norm + 1) / 2 * bbox_extents_mm
    xyz_m  = xyz_mm * COORD_SCALE

    T   = model(xyz_norm)
    lap = laplacian(T, xyz_norm) / (COORD_SCALE * bbox_extents_mm / 2) ** 2
    Q_t = model.Q_tumor(xyz_norm)

    f = k * lap + wb * cb * (Ta - T) + Qm + Q_t
    return f
```

### Pennes Bioheat equation (steady-state)

$$k\,\nabla^2 T + \omega_b c_b (T_a - T) + Q_m + Q_{\text{tumor}}(x,y,z) = 0$$

### Tumour heat source (Gaussian)

$$Q_{\text{tumor}} = Q_{\max} \cdot \exp\!\left(-\frac{(x-x_t)^2+(y-y_t)^2+(z-z_t)^2}{r_t^2}\right)$$

### Data loss

$$\mathcal{L}_{\text{data}} = \frac{1}{N_s}\sum_{i=1}^{N_s} w_i\,\big|T_\theta(\mathbf{p}_i) - T_{\text{measured},i}\big|^2$$

### Physics loss

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

### Loss weighting (static, computed at epoch 0)

```python
lambda_data = 1.0
lambda_pde  = (L_data_0 / L_pde_0).detach()
```

$$\mathcal{L}_{\text{total}} = \lambda_{\text{data}}\,\mathcal{L}_{\text{data}} + \lambda_{\text{PDE}}\,\mathcal{L}_{\text{PDE}}$$

---

## Stage 6 — Training schedule

| Phase | Optimiser | Steps | Purpose |
|-------|-----------|-------|---------|
| 1 | Adam, lr=1e-3 | 10,000 | Broad exploration, moves tumour params |
| 2 | L-BFGS, strong Wolfe | 500 | Fine convergence, drives PDE residual to zero |

**Multi-start:** 5 random initialisations of $(x_t, y_t, z_t)$ sampled from inside the mesh. Keep lowest final $\mathcal{L}_{\text{total}}$.

---

## Stage 7 — Output per patient

```python
# Denormalise tumour params from [-1,1] back to mm
results = {
    'x_t_mm'  : (model.x_t.item() + 1) / 2 * bbox_extents_mm[0],
    'y_t_mm'  : (model.y_t.item() + 1) / 2 * bbox_extents_mm[1],
    'z_t_mm'  : (model.z_t.item() + 1) / 2 * bbox_extents_mm[2],
    'r_t_mm'  : model.r_t.item(),
    'Q_max'   : model.Q_max.item(),
}
```

Estimated tumour volume: $V \approx \frac{4}{3}\pi r_t^3$

---

## Dataset iteration

```python
unet = UNet()
unet.load_state_dict(torch.load('unet_mask.pth'))
unet.eval().to(device)                                     # loaded once

for patient_dir in Path('dmr_ir/').glob('patient_*/'):
    stl_path   = patient_dir / 'mesh.stl'
    tiff_paths = sorted(patient_dir.glob('view_*.tiff'))   # 5 per patient

    mesh = load_mesh_mm(stl_path)                          # unit-corrected
    assert mesh.is_watertight
    if mesh.volume < 0:
        trimesh.repair.fix_normals(mesh)

    surface_pts, T_measured, confidence = project_thermal(
        unet, mesh, tiff_paths, device
    )
    interior_pts = trimesh.sample.volume_mesh(mesh, count=5000)

    model   = BioheatPINN()
    results = train_pinn(model, surface_pts, T_measured,
                         confidence, interior_pts, mesh.bounding_box.extents)

    save_results(patient_dir, results)
```

---

## Known failure modes

| Risk | Symptom | Fix |
|------|---------|-----|
| Wrong STL units | $r_t$ returns ~0.003 instead of ~3 mm | `load_mesh_mm()` with extent heuristic + assertion |
| Inverted mesh normals | `volume_mesh` samples outside breast | Check `mesh.volume > 0`, run `fix_normals` |
| Background pixels in projection | Noisy $T_{\text{measured}}$ at breast edge | U-Net mask + 2px erosion before pixel readout |
| Soft U-Net boundary | Ambiguous pixels at mask edge | 2px erosion on mask |
| Loss scale mismatch | PDE dominates, surface fit poor | Static $\lambda$ normalisation at epoch 0 |
| Local minima in tumour location | Result clusters at mesh edge | Multi-start (5 random inits) |
| $r_t$ collapses to zero | $Q_{\max}$ diverges | `r_t.data.clamp_(min=2.0)` after every step |
| Autograd OOM | CUDA out of memory | Reduce collocation batch to 2000 |
| KNN fill polluting boundary | High $\mathcal{L}_{\text{data}}$ in occluded zones | Confidence weight $w_i = 0.3$ |

---

## Literature (verified DOIs)

1. Mukhmetov O. et al. (2023) — PINN for breast bioheat, forward solver, validated 12× faster than FEM
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
