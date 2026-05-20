# Project context — 3D breast thermography reconstruction and PINN inverse bioheat solver
## Full pipeline reference document — reflects actual implementation

---

## 1. Project overview

Two-stage pipeline:

**Stage A — 2D to 3D reconstruction (BreastNet3D)**
Five-view infrared TIFF images → U-Net segmentation → Encoder2D/Decoder3D volumetric reconstruction → 128³ occupancy grid → marching cubes → STL. Thermal overlay applied via Lambertian weighted fusion.

**Stage B — Inverse bioheat solving (PINN + FEA validation)**
Registered STL surface + absolute thermal boundary condition → PINN recovers tumour parameters $(x_t, y_t, z_t, r_t, Q_{\max})$ → FEA verifies the result by comparing predicted surface temperatures against measured.

---

## 2. Actual file paths and model names

```
Server: v-maxwell-pc2
Root:   /mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/

Models:
  UNET_Segmentation/breast_segmentation_unet_best_gpu.pth   ← U-Net weights
  checkpoints_3d_finetuned/3dbreastnet_finetuned_best.pth   ← BreastNet3D weights
                                                               keys: "enc", "dec"

Data:
  data/organized_by_patient/
    {patient_id}/
      {label}/                         ← 'malignant' or 'benign'
        *right later*.tiff  → key: RL  (angle -90°)
        *right obli*.tiff   → key: RO  (angle -45°)
        *frontal*.tiff      → key: F   (angle   0°)
        *left obliq*.tiff   → key: LO  (angle +45°)
        *left later*.tiff   → key: LL  (angle +90°)

View order in all tensors: [RL, RO, F, LO, LL] → indices [0, 1, 2, 3, 4]
Frontal view = index 2
```

---

## 3. Model architectures (confirmed from source)

### U-Net (segmentation)
Standard encoder-decoder, input 1-channel normalised TIFF, output 1-channel mask logit.
Load:
```python
unet = UNet().to(device)
unet.load_state_dict(torch.load(
    'UNET_Segmentation/breast_segmentation_unet_best_gpu.pth',
    map_location=device, weights_only=False
))
unet.eval()
```

### BreastNet3D (Encoder2D + Decoder3D)
- **Encoder2D**: 5-channel mask input → 6 DoubleConv2D blocks → FC → 1000-dim latent
- **Decoder3D**: 1000-dim → FC → 6 ConvTranspose3D blocks → 128³ occupancy [0,1]
- Output uses `nn.Sigmoid()` with bias initialised to −4.0 (sparse prior)
- Gradient checkpointing on decoder stages 4–6 (memory optimisation)

Load:
```python
ckpt = torch.load(
    'checkpoints_3d_finetuned/3dbreastnet_finetuned_best.pth',
    map_location=device, weights_only=False
)
enc = Encoder2D().to(device); enc.load_state_dict(ckpt["enc"]); enc.eval()
dec = Decoder3D().to(device); dec.load_state_dict(ckpt["dec"]); dec.eval()
```

---

## 4. Current thermal projection — Lambertian weighted fusion

### What the current code does

`compute_thermal_overlay(verts, normals, thermals_5ch, masks_5ch)` in the existing notebook:

1. For each of the 5 views, compute camera view vector from acquisition angle
2. Dot product between **outward normals** and camera vector → Lambertian weight
3. Exponential weight: `w = max(dot + 0.3, 0) ** 12` — exponent 12 aggressively isolates front-facing surfaces
4. Sample the thermal image at projected vertex coordinates via `map_coordinates`
5. Weighted sum across all views → normalised by total weight
6. Fallback: untextured vertices get frontal view (index 2) values

**No ray intersection, no KNN gap-fill.** Occlusion is handled implicitly by the Lambertian weighting — back-facing vertices receive near-zero weight from all cameras and fall through to the frontal fallback.

### ⚠️ Critical issue — thermal values are normalised [0,1], not absolute °C

In `PatientDataset.__getitem__`:
```python
mn, mx = raw_256.min(), raw_256.max()
norm = (raw_256 - mn) / (mx - mn + 1e-8)   # ← normalised for U-Net
thermals_128.append(cv2.resize(norm, (128,128)))   # ← stored as [0,1]
```

The `thermals_5ch` tensor passed to `compute_thermal_overlay` contains **normalised [0,1] values**, not physical temperatures. The Pennes Bioheat equation requires absolute temperatures in °C (approx 30–37°C range). Feeding [0,1] values into the PINN boundary condition produces a physically meaningless PDE.

### Fix — dual-track thermal loading

Store both the normalised version (for U-Net) and the absolute version (for PINN) separately:

```python
class PatientDataset(Dataset):
    def __getitem__(self, idx):
        g = self.groups[idx]
        thermals_norm, thermals_abs, masks_128 = [], [], []

        for v in self.views:
            raw = tifffile.imread(str(g.views[v])).astype(np.float32)
            raw_256 = cv2.resize(raw, (256, 256))

            # Normalised — for U-Net input only
            mn, mx = raw_256.min(), raw_256.max()
            norm = (raw_256 - mn) / (mx - mn + 1e-8)

            with torch.no_grad():
                inp = torch.tensor(norm).unsqueeze(0).unsqueeze(0).to(self.device)
                m = (torch.sigmoid(self.unet(inp)).squeeze().cpu().numpy() > 0.5
                     ).astype(np.float32)

            masks_128.append(cv2.resize(m, (128,128), interpolation=cv2.INTER_NEAREST))

            # Normalised (for 3D reconstruction)
            norm_128 = cv2.resize(norm, (128,128), interpolation=cv2.INTER_LINEAR)
            thermals_norm.append(norm_128)

            # ── NEW: absolute °C (for PINN boundary condition) ──
            abs_128 = cv2.resize(raw_256, (128,128), interpolation=cv2.INTER_LINEAR)
            thermals_abs.append(abs_128)

        return {
            "masks_5ch"        : torch.tensor(np.stack(masks_128), dtype=torch.float32),
            "thermals_5ch"     : torch.tensor(np.stack(thermals_norm), dtype=torch.float32),
            "thermals_abs_5ch" : torch.tensor(np.stack(thermals_abs),  dtype=torch.float32),
            "patient_id"       : g.patient_id,
            "label"            : g.label,
        }
```

Then in `compute_thermal_overlay`, add a parameter to select which thermal channel to use:

```python
def compute_thermal_overlay(verts, normals, thermals_5ch, masks_5ch,
                             use_absolute=False, thermals_abs_5ch=None):
    """
    use_absolute=False → [0,1] normalised  (for 3D visualisation)
    use_absolute=True  → absolute °C       (for PINN boundary condition)
    """
    src = thermals_abs_5ch if (use_absolute and thermals_abs_5ch is not None) \
          else thermals_5ch
    # ... rest unchanged, just replace thermals_5ch with src
```

**Summary of what changes:**
- `thermals_5ch` → used for 3D reconstruction and visualisation (unchanged)
- `thermals_abs_5ch` → used as PINN boundary condition $T_{\text{measured}}$

---

## 5. Marching cubes and STL export

```python
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
import trimesh

vol_np = vol.float()[0, 0].cpu().numpy()
vol_np = gaussian_filter(vol_np, sigma=2.0)     # smooth before MC

verts, faces, normals, _ = marching_cubes(vol_np, level=0.01)

# Export STL
mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

# Validate
assert mesh.is_watertight, "Mesh is not watertight — check volume quality"
if mesh.volume < 0:
    trimesh.repair.fix_normals(mesh)

mesh.export(f'exported_stls/{patient_id}.stl')
```

### Unit check

The voxel grid is 128³ — vertices are in voxel units [0, 128]. Real breast half-width ~60–80mm → one voxel ≈ 1mm at this scale. **Verify per patient:**

```python
extents = mesh.bounding_box.extents
print(f"Extents: {extents}")   # expect ~[60-120, 80-140, 60-100] voxels
# If extents.max() < 1.0  → metres, apply scale 1000
# If extents.max() < 10.0 → cm,     apply scale 100
# else already mm-scale
```

---

## 6. IMF-based anatomical registration (v5 upgrade)

The current `3dbreastnet_finetuned_best.pth` rotates around the voxel grid centre — anatomically arbitrary. For cross-patient comparable tumour coordinates, register every STL to a shared anatomical frame using the IMF centroid.

### Extract IMF from frontal mask (index 2)

```python
def extract_imf_centroid_2d(mask_frontal):
    rows = np.where(mask_frontal.any(axis=1))[0]
    if len(rows) == 0:
        raise ValueError("Empty mask")
    imf_row  = rows.max()
    imf_cols = np.where(mask_frontal[imf_row])[0]
    return np.array([imf_cols.mean(), imf_row])     # [col, row]

def extract_superior_apex_2d(mask_frontal):
    rows = np.where(mask_frontal.any(axis=1))[0]
    apex_row  = rows.min()
    apex_cols = np.where(mask_frontal[apex_row])[0]
    return np.array([apex_cols.mean(), apex_row])
```

### Build anatomical frame and register STL

```python
def build_anatomical_frame(imf_3d, apex_3d):
    y_axis  = normalise(apex_3d - imf_3d)
    z_world = np.array([0.0, 0.0, 1.0])
    x_axis  = normalise(np.cross(y_axis, z_world))
    z_axis  = normalise(np.cross(x_axis, y_axis))
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return R, imf_3d

def register_mesh(mesh, R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    mesh.apply_transform(np.linalg.inv(T))
    return mesh

def normalise(v):
    return v / (np.linalg.norm(v) + 1e-8)
```

After registration:
- Origin = IMF centroid
- $+Y$ = superior, $-Z$ = into chest wall
- Tumour coordinates are clinically interpretable across all 122 patients

---

## 7. PINN inverse bioheat solver

### 7.1 Input to PINN

```
surface_pts    : [N_s, 3]  STL vertex coordinates, mm (registered)
T_measured     : [N_s]     ABSOLUTE temperatures in °C from thermals_abs_5ch
                           ← must use thermals_abs_5ch, NOT thermals_5ch
interior_pts   : [N_v, 3]  random points inside STL volume
bbox_extents   : [3]       bounding box in mm
```

### 7.2 Network architecture

```
Input:   (x, y, z)    — normalised to [-1, 1] from mm bounding box
Hidden:  6 × 256      — tanh (mandatory — ReLU kills ∂²T/∂x²)
Output:  T_θ(x,y,z)  — 1 neuron, linear, outputs temperature in °C
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

### 7.3 Biophysical constants (SI units)

| Symbol | Value | Units | Description |
|--------|-------|-------|-------------|
| $k$ | 0.48 | W/(m·K) | Thermal conductivity |
| $\omega_b$ | 0.0005 | 1/s | Blood perfusion rate |
| $c_b$ | 3600 | J/(kg·K) | Specific heat of blood |
| $T_a$ | 37.0 | °C | Arterial blood temperature |
| $Q_m$ | 450 | W/m³ | Healthy metabolic heat |
| `COORD_SCALE` | 1e-3 | — | mm → m, applied in PDE only |

### 7.4 Loss functions

**Pennes Bioheat (steady-state):**
$$k\,\nabla^2 T + \omega_b c_b (T_a - T) + Q_m + Q_{\text{tumor}} = 0$$

**Tumour source:**
$$Q_{\text{tumor}} = Q_{\max} \cdot \exp\!\left(-\frac{(x-x_t)^2+(y-y_t)^2+(z-z_t)^2}{r_t^2}\right)$$

**Data loss** (surface, absolute °C):
$$\mathcal{L}_{\text{data}} = \frac{1}{N_s}\sum_i w_i\,|T_\theta(\mathbf{p}_i) - T_{\text{measured},i}|^2$$

Confidence weights from Lambertian overlay:
- $w_i = 1.0$ — high Lambertian weight, consistent across views
- $w_i = 0.5$ — low Lambertian weight (near-grazing surface)
- $w_i = 0.3$ — frontal fallback (untextured vertex)

**Physics loss** (interior):
$$\mathcal{L}_{\text{PDE}} = \frac{1}{N_v}\sum_j |f(x_j,y_j,z_j)|^2$$

**Laplacian via autograd:**
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

**Static loss normalisation at epoch 0:**
```python
lambda_data = 1.0
lambda_pde  = (L_data_0 / L_pde_0).detach()
```

### 7.5 Training schedule

| Phase | Optimiser | Steps | Purpose |
|-------|-----------|-------|---------|
| 1 | Adam, lr=1e-3 | 10,000 | Broad exploration |
| 2 | L-BFGS, strong Wolfe | 500 | Fine convergence |

Multi-start: 5 random $(x_t, y_t, z_t)$ initialisations. Keep lowest final loss.
After every step: `model.r_t.data.clamp_(min=2.0)` and `model.Q_max.data.clamp_(min=0.0)`

### 7.6 Output per patient

```python
results = {
    'patient_id' : pid,
    'label'      : label,          # 'malignant' or 'benign'
    'x_t_mm'     : denorm(model.x_t.item(), bbox),
    'y_t_mm'     : denorm(model.y_t.item(), bbox),
    'z_t_mm'     : denorm(model.z_t.item(), bbox),
    'r_t_mm'     : abs(model.r_t.item()),
    'Q_max'      : model.Q_max.item(),
    'volume_mm3' : (4/3) * np.pi * abs(model.r_t.item())**3,
    'quadrant'   : classify_quadrant(x_t_mm, y_t_mm),
}
```

Quadrant classification (anatomical frame, origin = IMF):

| $x_t$ | $y_t$ | Quadrant |
|--------|--------|----------|
| > 0 | > 0 | Upper Outer |
| < 0 | > 0 | Upper Inner |
| > 0 | < 0 | Lower Outer |
| < 0 | < 0 | Lower Inner |

---

## 8. FEA forward verification

### Purpose

FEA runs **after** PINN using the PINN-estimated tumour parameters. It asks: *"if the tumour is where the PINN says it is, do the physics reproduce the surface temperatures we measured?"*

This is not data generation — it is physics-based answer verification.

### Installation (separate conda environment — do not mix with .venv)

```bash
conda create -n bioheat python=3.11 -y
conda activate bioheat
conda install -c conda-forge fenics-dolfinx mpich petsc -y
pip install gmsh meshio trimesh pyvista seaborn
pip install torch torchvision tifffile opencv-python numpy scipy scikit-image matplotlib tqdm pandas
```

Verify:
```bash
python -c "import dolfinx, gmsh, meshio; print('FEA stack OK')"
```

### STL → tetrahedral mesh (gmsh)

```python
import gmsh, numpy as np

def stl_to_tet_mesh(stl_path, out_msh_path, mesh_size_mm=3.0):
    gmsh.initialize()
    gmsh.model.add("breast")
    gmsh.merge(str(stl_path))
    gmsh.model.mesh.classifySurfaces(np.pi, True, True, np.pi)
    gmsh.model.mesh.createGeometry()
    s = gmsh.model.getEntities(2)
    l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([l])
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_mm)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_mm * 0.5)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.write(str(out_msh_path))
    gmsh.finalize()
```

### FEA forward solve (FEniCSx / dolfinx)

```python
from mpi4py import MPI
import dolfinx, dolfinx.io
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl

def run_fea_forward(msh_path, pinn_results):
    msh, _, _ = dolfinx.io.gmshio.read_from_msh(
        str(msh_path), MPI.COMM_WORLD, gdim=3
    )
    msh.geometry.x[:] *= 1e-3      # mm → m

    V  = fem.functionspace(msh, ("Lagrange", 1))
    T  = ufl.TrialFunction(V)
    v  = ufl.TestFunction(V)
    x  = ufl.SpatialCoordinate(msh)

    x_t  = pinn_results['x_t_mm']  * 1e-3
    y_t  = pinn_results['y_t_mm']  * 1e-3
    z_t  = pinn_results['z_t_mm']  * 1e-3
    r_t  = pinn_results['r_t_mm']  * 1e-3
    Qmax = pinn_results['Q_max']

    d2      = (x[0]-x_t)**2 + (x[1]-y_t)**2 + (x[2]-z_t)**2
    Q_tumor = Qmax * ufl.exp(-d2 / r_t**2)

    k=0.48; wb=0.0005; cb=3600.0; Ta=37.0; Qm=450.0
    h_conv=10.0; T_air=20.0

    a = (k  * ufl.inner(ufl.grad(T), ufl.grad(v))
         + wb * cb * T * v) * ufl.dx + h_conv * T * v * ufl.ds

    L = (wb * cb * Ta + Qm + Q_tumor) * v * ufl.dx + h_conv * T_air * v * ufl.ds

    problem = LinearProblem(a, L, bcs=[],
        petsc_options={"ksp_type": "cg", "pc_type": "gamg"})
    return problem.solve(), msh
```

### Residual computation

```python
from scipy.spatial import cKDTree

def compute_residual(T_fea_sol, msh, surface_pts_mm, T_measured_abs):
    fea_coords = msh.geometry.x * 1e3          # m → mm
    fea_vals   = T_fea_sol.x.array             # °C at FEM nodes

    tree = cKDTree(fea_coords)
    _, idx = tree.query(surface_pts_mm, k=3)
    T_fea_interp = fea_vals[idx].mean(axis=1)

    residuals  = np.abs(T_fea_interp - T_measured_abs)
    return residuals, residuals.mean(), residuals.max()
```

**Interpretation:**
- Mean residual < 0.5°C → good PINN convergence
- Mean residual > 1.5°C → PINN did not converge, increase Adam steps or re-run multistart

---

## 9. Run on both classes — benign AND malignant

**All 122 patients are processed.** Class label is only used at the analysis stage.

### Scientific rationale

| Reason | Explanation |
|--------|-------------|
| Negative control | Benign cases establish the baseline $Q_{\max}$ distribution |
| Statistical test | Mann-Whitney U on $Q_{\max}$: malignant > benign expected |
| Warburg effect | Malignant cells have elevated glycolytic metabolism → higher $Q_{\max}$ |
| No ground truth needed | The label comparison IS the validation |
| Dataset efficiency | Skipping benign wastes half the 122-patient cohort |

### Analysis code

```python
from scipy import stats
import seaborn as sns

def analyse_results(df):
    mal = df[df['label']=='malignant']['Q_max'].dropna()
    ben = df[df['label']=='benign'  ]['Q_max'].dropna()

    stat, p = stats.mannwhitneyu(mal, ben, alternative='greater')
    print(f"Malignant Q_max : {mal.mean():.1f} ± {mal.std():.1f} W/m³  (n={len(mal)})")
    print(f"Benign    Q_max : {ben.mean():.1f} ± {ben.std():.1f} W/m³  (n={len(ben)})")
    print(f"Mann-Whitney U  : p = {p:.4f} ({'significant' if p<0.05 else 'not significant'})")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(mal, label='Malignant', ax=ax, fill=True, alpha=0.4)
    sns.kdeplot(ben, label='Benign',    ax=ax, fill=True, alpha=0.4)
    ax.set_xlabel('Q_max (W/m³)')
    ax.set_title('Tumour Heat Generation Distribution')
    ax.legend()
    plt.savefig('Q_max_distribution.png', dpi=150)
```

---

## 10. Known failure modes

| Risk | Symptom | Fix |
|------|---------|-----|
| **T_measured in [0,1] not °C** | PINN converges but PDE residual nonsensical | Use `thermals_abs_5ch` not `thermals_5ch` |
| gmsh fails on non-manifold STL | `classifySurfaces` error | Run `trimesh.repair.fill_holes` first |
| Inverted mesh normals | `volume_mesh` samples outside | Check `mesh.volume > 0`, run `fix_normals` |
| Loss scale mismatch | PDE dominates | Static λ normalisation at epoch 0 |
| $r_t$ collapses to zero | $Q_{\max}$ diverges | `r_t.data.clamp_(min=2.0)` every step |
| $Q_{\max}$ goes negative | Unphysical | `Q_max.data.clamp_(min=0.0)` every step |
| FEniCSx / .venv conflict | MPI import errors | Use dedicated conda env `bioheat` |
| All patients same $Q_{\max}$ | Loss collapse | Verify `thermals_abs_5ch` scale (expect 28–38°C) |
| PINN FEA residual > 1.5°C | Poor convergence | Increase Adam to 20k steps, re-run multistart |

---

## 11. Output files per patient

```
exported_stls/
  {patient_id}.stl             ← watertight hollow shell (marching cubes)
  {patient_id}.msh             ← tetrahedral volume mesh (gmsh)

results/
  {patient_id}_T_measured.npy  ← absolute surface temperatures [N_s] °C
  {patient_id}_surf_pts.npy    ← surface point coordinates [N_s, 3] mm
  {patient_id}_confidence.npy  ← Lambertian-derived weights [N_s]
  {patient_id}_pinn.pth        ← trained PINN weights + tumour params
  {patient_id}_T_fea.npy       ← FEA surface temperatures [N_s] °C

pinn_fea_results.csv           ← one row per patient, all metrics + label
```

---

## 12. Literature (verified DOIs)

1. Mukhmetov O. et al. (2023) — PINN for breast bioheat, 12× faster than FEM
   *Computer Methods and Programs in Biomedicine*, 242, 107834
   `doi:10.1016/j.cmpb.2023.107834`

2. Karniadakis G.E. et al. (2021) — Physics-informed machine learning
   *Nature Reviews Physics*, 3(6), 422–440
   `doi:10.1038/s42254-021-00314-5`

3. Wang S., Yu X., Perdikaris P. (2022) — When and why PINNs fail
   *Journal of Computational Physics*, 449, 110768
   `doi:10.1016/j.jcp.2021.110768`

4. Majdoubi J. et al. (2021) — NN inverse bioheat, tumour params from surface temperatures
   *Computer Methods and Programs in Biomedicine*, 106092
   `doi:10.1016/j.cmpb.2021.106092`

5. Perez-Raya I., Gutierrez C., Kandlikar S.G. (2024) — Inverse PINN, biopsy-validated
   *ASME Journal of Heat and Mass Transfer*, 146(10), 101201
   `doi:10.1115/1.4065673`
