# Stage 3 — TherMAM-NeRF: Joint Geometry + Thermal Field Reconstruction

**Training source:** `finalized/thermamnerf_v3.0.py`
**Inference / extraction:** `TherMAM-NeRF/Thermamnerf_PINN_v3.ipynb`
**Predecessor (superseded):** `UNET_Segmentation/3DBreastnet/breastnet3d_v5.ipynb`

> *This file documents the **current** reconstruction stage (TherMAM-NeRF). The earlier 3DBreastNet approach is kept in §2 as the documented predecessor it replaced.*

---

## 1. Objective

For each patient we have five aligned infrared (IR) views — Right-Lateral, Right-Oblique, Frontal, Left-Oblique, Left-Lateral — each a calibrated temperature image plus a U-Net breast mask from Stage 2. This stage turns those five 2-D views into **one continuous 3-D neural field** that returns, at every point in space, both:

- an **occupancy density** $\sigma$ (geometry), and
- a **temperature** $T$ (the thermal field).

From that field we extract a **watertight 3-D breast surface** and a **per-vertex skin-temperature map**. These are the two inputs the Stage 4 Pennes bioheat inverse solver consumes:

1. a **3-D anatomical domain** $\partial\Omega$ on which the bioheat PDE is posed (skin surface + chest wall), and
2. a **per-vertex surface temperature** $T_{\text{measured}}$ — the boundary data the inverse problem is matched against.

The defining idea is that geometry and temperature are learned **jointly as one field**, not shape-first-then-paint.

### 1.1 Code map — what each object does

| Code object | Role | Doc § |
|---|---|---|
| `setup_distributed()` | initialise DDP / pick CUDA device | §7.4 |
| `discover_patients_split()` | keep only patients with all 5 views (TIFF + mask) | §7.2 |
| `BreastThermDataset` | load one patient → normalised tensors `[V,H,W]` | §7.2 |
| `SiameseEncoder` | shared 2-D CNN → per-view feature maps | §3.1 |
| `positional_encoding()` | Fourier features + coarse-to-fine window | §3.2 |
| `project_and_sample()` | aggregate the 5 views (mean + variance) at a 3-D point | §3.3 |
| `ThermamNeRFMLP` | decode features → $(\sigma, T)$ | §3.4 |
| `get_rays()` | parallel-projection ray origins / directions | §9.1 |
| `volume_render()` | α-composite a ray → opacity + temperature | §9.2–9.4 |
| `compute_loss()` | the five-term training loss | §8 |
| `run_one_batch()` | one training step: sample rays → render → loss | §8–9 |
| `render_view()` | full-image render for inference / audit | §13 |
| `extract_3d_volume()` | dense $128^3$ grid query for meshing | §4 |
| training loop | epochs, validation, checkpointing | §11–12 |
| post-training audit | all-patient geometry + thermal metrics + plots | §13 |

---

## 2. From 3DBreastNet to TherMAM-NeRF (why the method changed)

The predecessor, **3DBreastNet** (`breastnet3d_v5`, best validation Dice **0.8427**), learned a $128^3$ occupancy volume from the five **binary masks** alone, supervised by multi-view differentiable-projection (visual-hull) consistency. Temperature was attached *afterwards* by a Lambertian-weighted projection of the raw thermal images onto the extracted mesh.

That is fine for shape but **not** for bioheat inversion, which needs a temperature field that is *part of the reconstruction*, not a post-hoc paint job:

- 3DBreastNet discards the thermal signal during reconstruction — geometry is learned from silhouettes only.
- Its surface temperature inherits projection seams and occlusion artefacts from the blending step.

**TherMAM-NeRF** instead learns geometry and temperature *jointly* as one continuous field, and the bioheat solver consumes the reconstruction's own temperature output directly. 3DBreastNet remains the reference for the occupancy / visual-hull formulation and the asymmetry features.

---

## 3. Architecture

TherMAM-NeRF is a neural field

$$f_\Theta:\ \mathbb{R}^3 \to (\sigma, T),\qquad \sigma \ge 0,\ \ T\in[0,1]$$

conditioned on the five input views (~205k parameters total). It has four pieces: a view encoder, a positional encoding, a multi-view aggregator, and a field decoder.

### 3.1 Siamese view encoder (`SiameseEncoder`)

One **shared** 2-D CNN $E$ is applied to every view $k$. Its input is the two-channel stack of the normalised thermal image $\tilde I_k$ and its U-Net mask $M_k$; its output is masked back to the breast region:

$$F_k = E\big([\,\tilde I_k,\ M_k\,]\big)\,\odot\, M_k \ \in\ \mathbb{R}^{32\times H\times W}$$

The encoder is four convolutions (channels $2\to16\to32\to32\to32$) with GroupNorm + ReLU. **Sharing weights across all five views** ("Siamese") forces one feature extractor to explain every viewpoint, which regularises the features and keeps the parameter count tiny. GroupNorm (not BatchNorm) is used because the batch size is 1.

### 3.2 Positional encoding (`positional_encoding`)

A query point $\mathbf{x}\in[-1,1]^3$ is lifted to high-frequency Fourier features with $L=8$ bands so the MLP can represent sharp geometric and thermal detail:

$$\gamma(\mathbf{x}) = \Big[\,\sin(2^{\ell}\pi x_d),\ \cos(2^{\ell}\pi x_d)\,\Big]_{d\in\{x,y,z\},\ \ell=0\ldots L-1}\ \in\ \mathbb{R}^{6L}$$

A **coarse-to-fine window** $w_\ell$ gates each band during training (see §11.3); early on only the low-frequency bands are active, which prevents the "solid block" failure mode.

### 3.3 Multi-view feature aggregation (`project_and_sample`)

For a 3-D query point the network gathers evidence from **all five** feature maps. The point is rotated into each view's camera frame about the vertical axis (angles $0^\circ,\pm45^\circ,\pm90^\circ$) and projected; the feature map is read by bilinear interpolation:

$$x_r = \cos\theta_k\, x + \sin\theta_k\, z,\qquad y_r = y,\qquad \mathbf{g}_k = \text{grid\_sample}\big(F_k,\,(x_r,y_r)\big)$$

The per-view samples are summarised by their **mean** and **variance** across views — the variance is a cue for multi-view *disagreement* (surface ambiguity):

$$\mathbf{a}(\mathbf{x}) = \Big[\ \operatorname{mean}_k \mathbf{g}_k,\ \ \operatorname{var}_k \mathbf{g}_k\ \Big]\ \in\ \mathbb{R}^{64}$$

### 3.4 Field decoder (`ThermamNeRFMLP`)

A skip-connection MLP (256-wide, 4 layers) maps the concatenated positional encoding and aggregated features to the two field heads:

$$(\sigma, T) = \text{MLP}\big([\,\gamma(\mathbf{x}),\ \mathbf{a}(\mathbf{x})\,]\big),\qquad
\sigma = \text{softplus}(\cdot)\ge 0,\quad T = \text{sigmoid}(\cdot)\in[0,1]$$

A skip connection re-injects the input $[\gamma,\mathbf a]$ at the middle layer so high-frequency detail is not washed out by depth. $\sigma$ is occupancy (geometry); $T$ is the **normalised** temperature, later mapped back to °C (§4.3).

---

## 4. Volume extraction and mesh construction (`extract_3d_volume`)

At inference the field is evaluated on a dense $128^3$ grid (chunked to fit GPU memory), giving a density grid $\sigma$ and a temperature grid $T$.

### 4.1 Geometry

1. **Smooth** the density grid (Gaussian) to suppress voxel noise without touching the thermal field.
2. **Marching Cubes** at iso-level $\tau = 0.3$ extracts the raw surface.
3. **Axis reorder** to align array indices with the physical $(X,Y,Z)$ frame, where $X$ is the right–left (RL↔LL) axis matching the frontal view.
4. **Trimesh repair**: keep the largest connected component, `fill_holes`, `fix_normals`, optional Laplacian smoothing → a watertight mesh with outward vertex normals (required for the Stage 4 Robin skin boundary condition).

### 4.2 Per-patient physical scale calibration

A reconstruction in $[-1,1]^3$ must be scaled to millimetres before any physics is posed (the Laplacian scales as $1/\text{extent}^2$, so a wrong scale corrupts the PDE). Rather than a fixed radius, scale is calibrated **per patient** from the FLIR SC620 imaging geometry used in DMR-IR (1.0 m standoff, 24° horizontal FOV ⇒ a 425 mm scene across the 640-px sensor):

$$\text{breast\_radius\_mm} = \tfrac{1}{2}\, w_{\text{px}} \cdot \frac{425}{\text{img\_size}}$$

where $w_{\text{px}}$ is the breast width in the frontal mask. *Example:* Patient 1 → **157.7 mm** (mask 95 px wide) vs. a hardcoded 70 mm — removing a large patient-dependent error from the downstream PDE residual.

### 4.3 Surface temperature

The temperature grid is sampled at each mesh vertex by trilinear interpolation and de-normalised to °C using the frontal view's calibration anchor $(T_{\min}, T_{\max})$:

$$T_{\text{measured}}(\mathbf{v}) = \hat T(\mathbf{v})\,(T_{\max} - T_{\min}) + T_{\min}$$

*Example range (Patient 1):* $T_{\text{measured}} \in [21.9,\ 33.5]\,^\circ$C.

### 4.4 Two read-outs of the same field (ray-casting vs. grid query)

The field $f_\Theta$ is **not an image** — it returns $(\sigma,T)$ at a single 3-D point. It is read out two ways:

| Read-out | How the field is queried | Produces | Used in |
|---|---|---|---|
| **Ray-casting** (training) | march rays, composite $\sigma$ and $T$ along each ray | 2-D opacity / temperature maps | §9 — supervision against the 5 IR views |
| **Grid query** (extraction) | evaluate $f_\Theta$ on the dense $128^3$ lattice — no rays | $\sigma$ and $T$ **volumes** | §4 — mesh + per-vertex temperature |

Ray-casting is only the *differentiable bridge* used at training time to compare the 3-D field against 2-D ground truth. The final 3-D product does **not** ray-cast — it queries the field on the grid and runs Marching Cubes. Because temperature is part of the learned field, per-vertex $T_{\text{measured}}$ is read **straight from the temperature grid** — there is no visibility test, no KNN occlusion in-fill, and no projection seams (the key contrast with 3DBreastNet's painted temperature).

---

## 5. The trust boundary (critical for Stage 4)

TherMAM-NeRF produces a *volumetric* field, but **only its surface values are trusted.** A field trained on outside-in IR views cannot observe the breast interior — its interior temperature is unconstrained extrapolation. The bioheat solver therefore consumes only $T_{\text{measured}}$ on the skin and lets the **physics** (Pennes PDE + boundary conditions) determine the interior. This separation is what makes Stage 4 a physically-grounded inverse rather than a fit to invented interior data.

---

## 6. Outputs per patient

| Field | Description |
|---|---|
| `surface_pts` $[N_s,3]$ | watertight mesh vertices (mm) |
| `vertex_normals` $[N_s,3]$ | outward unit normals (Robin BC) |
| `T_measured` $[N_s]$ | per-vertex skin temperature (°C) |
| `interior_pts` $[N_v,3]$ | interior collocation points (PDE loss) |
| `bbox_min/max` | domain extent, for normalisation and the chest-wall slab |
| `{Patient}.stl` | exported surface for the FEA cross-check |

These feed directly into Stage 4. The dataset yields **122 patients** with a complete five-view set.

---

## 7. Training Pipeline

### 7.1 Configuration (`CFG`)

All hyperparameters live in one `CFG` dictionary so a run is reproducible from one place.

| Parameter | Value | Notes |
|---|---|---|
| `img_size` | 128 | Encoder feature maps and ray grids are 128×128 |
| `n_views` | 5 | RL, RO, F, LO, LL |
| `view_angles_deg` | [−90, −45, 0, +45, +90] | Camera rotation from frontal; +90 = Left Lateral |
| `feat_channels` | 32 | Encoder output depth; per-point feature = 64 (mean+var) |
| `pos_enc_L` | 8 | Fourier bands; PE dim = 3×2×8 = 48 |
| `mlp_hidden` | 256 | Width of every MLP layer |
| `mlp_layers` | 4 | Depth excluding output heads |
| `n_samples` | 256 | Quadrature points per ray |
| `near / far` | −1.0 / +1.0 | Ray integration range (normalised coords) |
| `density_scale` | 10.0 | Multiplies $\sigma$ before $\alpha$ (higher = sharper surface) |
| `freq_warmup_epochs` | 50 | Epochs over which the PE is annealed coarse→fine |
| `batch_size` | 1 | One patient per gradient step |
| `n_epochs` | 1 000 | Total training epochs |
| `lr` | 5×10⁻⁴ | Adam initial learning rate |
| `lambda_dice` | 1.0 | Geometry Dice weight |
| `lambda_bg` | 2.0 | Background suppression weight (at full ramp) |
| `lambda_thermal` | 20.0 | Thermal MSE weight |
| `lambda_tv` | 0.01 | 3-D total-variation weight (epoch 1) |
| `lambda_entropy` | 0.1 | Ray opacity entropy weight (at full ramp) |
| `n_rays` | 3 072 | Randomly sampled rays per view per step |
| `mc_threshold` | 0.3 | Marching-cubes iso-level |
| `mc_resolution` | 128 | Grid resolution for extraction |
| `use_amp` | True | Automatic mixed precision (CUDA only) |
| `use_grad_checkpoint` | True | Gradient checkpointing in the MLP to fit VRAM |

### 7.1.1 Runtime performance settings (do **not** change the trained weights)

Beyond `CFG`, the script sets a few global flags purely for speed:

| Setting | Purpose |
|---|---|
| `torch.backends.cudnn.benchmark = True` | autotunes conv kernels for the fixed 128×128 input |
| `torch.backends.cuda.matmul.allow_tf32 = True`, `cudnn.allow_tf32 = True` | TF32 matmuls on Ampere+ (negligible precision cost; delete for bit-exact fp32) |
| DataLoader `num_workers=4`, `pin_memory=True`, `persistent_workers=True` | overlaps per-patient TIFF loading with GPU compute |
| `use_grad_checkpoint = True` | trades extra compute for lower VRAM in the MLP; **on** here because turning it off overflows the GPUs. The gradients are identical either way. |

The first three are exactly numerically equivalent (or negligibly so); the last is a pure memory/compute trade-off with no effect on the model output.

### 7.2 Dataset and data loading

**Directory layout:**

```
data/
  organized_by_patient/        # TIFF_DIR — raw thermal images
    Patient_1/benign/
      right_lateral.tiff  right_oblique.tiff  frontal.tiff
      left_oblique.tiff   left_lateral.tiff
  organized_by_patient_unet/   # UNET_DIR — Stage-2 U-Net masks
    Patient_1/benign/*.png
```

`discover_patients_split` scans both trees and keeps **only** patients that have all five views in *both* directories. View keys come from filename keywords (`right later`→`RL`, `frontal`/`anterior`→`F`, …). The list is sorted by patient number and printed.

`BreastThermDataset.__getitem__` loads one patient:

1. Read each TIFF at float32 (`tifffile.imread`), resize bilinearly to `img_size` (128).
2. **Min–max normalise** to $[0,1]$, saving $(T_{\min},T_{\max})$ per view for later de-normalisation:
$$\tilde I_{i,j} = \frac{I_{i,j}-T_{\min}}{T_{\max}-T_{\min}+\varepsilon}$$
3. Load the U-Net mask PNG, binarise at 0.5.
4. Return stacked tensors `[V,H,W]` for normalised thermals, raw °C thermals (`tiffs_abs`), and masks, plus the per-view $(T_{\min},T_{\max})$.

### 7.3 Train / validation / test split (70 / 15 / 15)

The cohort is shuffled once with a fixed seed and partitioned three ways:

```python
random.seed(42)
random.shuffle(patients)
n       = len(patients)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
train_patients = patients[:n_train]                 # 70 %
val_patients   = patients[n_train:n_train + n_val]   # 15 %
test_patients  = patients[n_train + n_val:]          # 15 %
```

For the 122-patient cohort this is **85 train / 18 val / 19 test** (the script prints exact counts at startup). The roles are strictly separated:

| Split | Used for | Bias |
|---|---|---|
| **train** | gradient updates | — |
| **val** | per-epoch checkpoint selection (§12) | optimistic (model is selected on it) |
| **test** | **sealed** — opened only in the final audit (§13) | **unbiased — report this** |

The seed makes the split deterministic. Validation data are loaded only on rank 0 (`IS_MAIN`).

### 7.4 Distributed Data Parallel (DDP)

Launch on two GPUs with:

```bash
torchrun --nproc_per_node=2 finalized/thermamnerf_v3.0.py
```

`setup_distributed()` reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK` from the env vars `torchrun` sets, calls `dist.init_process_group(backend='nccl', timeout=1h)`, and gives each process its own CUDA device. A `DistributedSampler` shards the **training** set across processes; validation and the post-training audit run only on rank 0. Per-epoch training loss is combined across ranks with `dist.all_reduce` before printing. The 1-hour NCCL timeout is generous so the long rank-0 audit cannot trip the barrier while rank 1 waits.

> Single-GPU is also fine: `CUDA_VISIBLE_DEVICES=1 python finalized/thermamnerf_v3.0.py` (DDP is disabled when `WORLD_SIZE==1`).

---

## 8. Loss Functions (`compute_loss`)

All five losses are computed **per view** and averaged over the five views before combining with their weights.

### 8.1 Dice loss (geometry)

Drives the rendered opacity $\hat M$ to match the binary U-Net mask $M$:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i \hat M_i M_i + \varepsilon}{\sum_i \hat M_i + \sum_i M_i + \varepsilon},\qquad \varepsilon = 10^{-6}$$

Robust to class imbalance (the silhouette is sparse against the frame) because it normalises by both foreground areas.

### 8.2 Background suppression loss

Background pixels ($M\le 0.5$) must render near-zero opacity:

$$\mathcal{L}_{\text{bg}} = \frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}} \hat M_i^2,\qquad \mathcal{B}=\{i: M_i\le 0.5\}$$

Without it, rays outside the silhouette accumulate spurious density ("floaters"). The weight ramps $0\to\lambda_{\text{bg}}$ over the first 50 epochs so geometry can form first.

### 8.3 Thermal MSE loss

Temperature is supervised **only inside the breast** (foreground), in normalised space:

$$\mathcal{L}_{\text{thermal}} = \frac{1}{|\mathcal{F}|}\sum_{i\in\mathcal{F}} \big(\hat T_i - \tilde I_i\big)^2,\qquad \mathcal{F}=\{i: M_i>0.5\}$$

Computing it only on the foreground avoids penalising background temperature, which has no physical meaning.

### 8.4 Total-variation loss (3-D)

Regularises the density field against spatial noise on a coarse $32^3$ sub-grid sampled each step:

$$\mathcal{L}_{\text{tv}} = \frac{1}{N}\sum|\Delta_x \sigma| + \frac{1}{N}\sum|\Delta_y \sigma| + \frac{1}{N}\sum|\Delta_z \sigma|$$

The weight decays $\lambda_{\text{tv}}(\text{epoch}) = \lambda_{\text{tv}}\cdot\max(0.5,\,e^{-0.015\,\text{epoch}})$ — halving roughly every 46 epochs but flooring at 50 % to keep edges smooth throughout.

### 8.5 Entropy loss

Snaps semi-transparent voxels to fully empty/occupied (prevents "comb" artefacts at the boundary):

$$\mathcal{L}_{\text{entropy}} = \frac{1}{N}\sum_i \hat M_i(1-\hat M_i)$$

Minimised when $\hat M_i\in\{0,1\}$. The weight ramps $0\to\lambda_{\text{entropy}}$ over the first 100 epochs.

### 8.6 Combined loss

$$\boxed{\ \mathcal{L} = \lambda_{\text{dice}}\mathcal{L}_{\text{Dice}} + \lambda_{\text{bg}}\mathcal{L}_{\text{bg}} + \lambda_{\text{thermal}}\mathcal{L}_{\text{thermal}} + \lambda_{\text{tv}}\mathcal{L}_{\text{tv}} + \lambda_{\text{entropy}}\mathcal{L}_{\text{entropy}}\ }$$

| Weight | Value | Schedule |
|---|---|---|
| $\lambda_{\text{dice}}$ | 1.0 | Fixed |
| $\lambda_{\text{bg}}$ | 2.0 | Linear ramp, epochs 1–50 |
| $\lambda_{\text{thermal}}$ | 20.0 | Fixed |
| $\lambda_{\text{tv}}$ | 0.01 → ≥0.005 | Exponential decay, floor 50 % |
| $\lambda_{\text{entropy}}$ | 0.1 | Linear ramp, epochs 1–100 |

$\lambda_{\text{thermal}}=20$ is large because thermal MSE naturally sits near ~0.05 whereas Dice loss sits near ~0.1; the coefficient keeps the two gradient magnitudes comparable.

---

## 9. Volume Rendering (`get_rays`, `volume_render`, `run_one_batch`)

### 9.1 Ray sampling

For each view at angle $\theta_k$ rays are cast from a **parallel-projection** camera with direction $\mathbf{d}=(-\sin\theta_k,\,0,\,\cos\theta_k)^\top$; each ray's origin is its pixel position in the rotated camera plane. During training, `n_rays = 3072` rays are **randomly sampled** per view per step (not the full $128^2=16384$), with stratified noise on the $t$ values to avoid aliasing.

### 9.2 Transmittance and weights

Along each ray, $n_{\text{samples}}=256$ points are placed in $[-1,1]$. For sample $i$ with interval $\delta_i$ and $s=\text{density\_scale}=10$:

$$\alpha_i = 1 - \exp(-\sigma_i\, s\, \delta_i),\qquad
T_i = \prod_{j<i}(1-\alpha_j),\qquad
w_i = \alpha_i\, T_i$$

$\alpha_i$ is the opacity of the sample, $T_i$ the transmittance (fraction of light reaching it), $w_i$ its compositing weight.

### 9.3 Rendered outputs

$$\hat M = \sum_i w_i \quad\text{(rendered opacity / mask)}$$

$$\hat T = \frac{\sum_i w_i^{\,\text{sg}}\, T_i^{\text{field}}}{\sum_i w_i^{\,\text{sg}} + \varepsilon}\cdot \mathbb{1}\!\left[\hat M^{\text{sg}} > 0.05\right]$$

where ${\text{sg}}$ denotes `detach()` (stop-gradient); the rendered temperature is masked to zero outside the silhouette.

### 9.4 Detached weights for thermal (critical design decision)

In `volume_render` the occupancy weights are **detached before multiplying the temperature field**:

```python
rendered_temp = (weights.detach() * T_field).sum(dim=1)
```

If they were not detached, the thermal loss would back-propagate through $\sigma$ and corrupt the geometry with purely thermal gradients. Detaching means the thermal loss updates **only the $T$ head**, while $\sigma$ is updated only by Dice + BG + TV + Entropy. The two heads thus train independently despite sharing the MLP trunk.

---

## 10. View-Specific Depth Cropping

### 10.1 The ribcage problem

The ribcage is curved, so from a strict 90° side view the *far* breast protrudes slightly behind the *near* breast along the viewing axis. Because the U-Net masks annotate only the **near** breast in lateral views, the network would be wrongly penalised for predicting the far breast's outline.

### 10.2 Implementation

A simple $X$-coordinate mask zeros density for the unwanted breast before the volume integral:

```python
if angle <= -80:   # RL — far breast is on the left  (X > 0)
    sigma = sigma.masked_fill(pts_flat[..., 0:1] > 0.1, 0.0)
elif angle >= 80:  # LL — far breast is on the right (X < 0)
    sigma = sigma.masked_fill(pts_flat[..., 0:1] < -0.1, 0.0)
```

The buffer $|X|=0.1$ (normalised) sits around the mid-sagittal plane. This masking is applied in **both** training and inference (`render_view`) so the rendered silhouette stays consistent with the GT masks.

---

## 11. Optimiser, Scheduler, and Annealing

### 11.1 Adam + cosine annealing

```
Optimiser: Adam(lr=5e-4, betas=(0.9, 0.999))
Scheduler: CosineAnnealingLR(T_max=1000, eta_min=lr*0.05 = 2.5e-5)
```

The LR follows $\eta(t)=\eta_{\min}+\tfrac12(\eta_0-\eta_{\min})\big(1+\cos(\pi t/T)\big)$, decaying smoothly to $2.5\times10^{-5}$ at epoch 1000. Gradients are clipped to unit norm before each step:

```python
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
```

Mixed precision (`autocast` + `GradScaler`) wraps the forward/backward when `use_amp` is on.

### 11.2 Dynamic lambda scheduling

Three weights change each epoch via `cfg_step`:

| Weight | Schedule | Rationale |
|---|---|---|
| $\lambda_{\text{tv}}$ | $\lambda_{\text{tv}}\cdot\max(0.5,\,e^{-0.015\,\text{epoch}})$ | strong smoothing early, never disappears |
| $\lambda_{\text{bg}}$ | $\lambda_{\text{bg}}\cdot\min(1,\,\text{epoch}/50)$ | let geometry form before crushing background |
| $\lambda_{\text{entropy}}$ | $\lambda_{\text{entropy}}\cdot\min(1,\,\text{epoch}/100)$ | slow ramp prevents comb artefacts at epoch 1 |

### 11.3 Coarse-to-fine frequency annealing

The PE window parameter $\alpha$ rises linearly $0\to L=8$ over the first 50 epochs:

$$\alpha(\text{epoch}) = \min\!\left(\frac{L\cdot\text{epoch}}{\text{warmup\_epochs}},\ L\right)$$

and each Fourier band $\ell$ is gated by a cosine window:

$$w_\ell = \tfrac12\Big(1 - \cos\big(\pi\,\text{clamp}(\alpha-\ell,\,0,\,1)\big)\Big)$$

At epoch 0 only the coarsest band is active; bands unlock one by one, so the network learns global shape before fine texture — avoiding the "solid block" trap.

---

## 12. Validation and Checkpointing

### 12.1 Joint validation metric

Validation runs on rank 0 after each epoch. The **checkpoint-selection metric** is

$$J_{\text{val}} = \mathcal{L}_{\text{Dice}}^{\text{val}} + \frac{\lambda_{\text{thermal}}}{10}\,\mathcal{L}_{\text{thermal}}^{\text{val}} = \mathcal{L}_{\text{Dice}}^{\text{val}} + 2\,\mathcal{L}_{\text{thermal}}^{\text{val}}$$

A **lower** $J_{\text{val}}$ is better. The factor 2.0 keeps geometry the primary driver of selection while letting thermal act as a tie-breaker — this is a *free* selection criterion and deliberately does **not** equal the training weight $\lambda_{\text{thermal}}=20$. For interpretability, Dice and IoU are also logged:

$$\text{Dice} = 1 - \mathcal{L}_{\text{Dice}},\qquad \text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$$

### 12.2 Checkpoint strategy

Two checkpoints are saved each epoch on rank 0:

| File | Condition | Purpose |
|---|---|---|
| `thermamnerf_best.pth` | $J_{\text{val}}$ improved | model loaded for inference / Stage 4 |
| `thermamnerf_latest.pth` | every epoch | crash recovery |

Each contains `{'encoder': state_dict, 'mlp': state_dict, 'epoch': epoch}`. DDP wrapper keys (`module.`) are stripped via the `encoder_base` / `mlp_base` references before saving. When `RESUME=True`, the start epoch is read back from the checkpoint's `epoch` key (not hardcoded), and the scheduler is fast-forwarded by replaying `scheduler.step()` for the skipped epochs.

---

## 13. Post-training Audit (all patients) and Metrics

After training, the script reloads `thermamnerf_best.pth` and audits **every patient**, processing the three splits separately so the held-out **test** set is reported as the unbiased estimate. For each patient and each of the five views it renders the mask and temperature at the fully-annealed $\alpha=L=8$ and computes two families of metrics.

### 13.1 Geometry — Dice / IoU

The rendered opacity is thresholded at 0.5 and compared to the GT U-Net mask $B$:

$$\text{Dice} = \frac{2|A\cap B|}{|A|+|B|},\qquad \text{IoU} = \frac{|A\cap B|}{|A\cup B|},\qquad A=\{\hat M>0.5\}$$

### 13.2 Thermal — pixel accuracy vs the GT TIFF

The rendered **normalised** temperature is first **de-normalised to °C** using each view's own $(T_{\min},T_{\max})$, then compared pixel-for-pixel against the ground-truth thermal TIFF — both at 128×128, so it is apples-to-apples — over the GT breast region $\mathcal{F}=\{M>0.5\}$:

$$\hat T^{\,\circ\mathrm C}_i = \hat T_i\,(T_{\max}-T_{\min}) + T_{\min},\qquad
e_i = \big|\,\hat T^{\,\circ\mathrm C}_i - T^{\text{TIFF}}_i\,\big|,\quad i\in\mathcal{F}$$

From the per-pixel error $e_i$, four numbers are reported per view:

| Metric | Definition |
|---|---|
| MAE (°C) | $\dfrac{1}{|\mathcal F|}\sum_{i\in\mathcal F} e_i$ |
| RMSE (°C) | $\sqrt{\dfrac{1}{|\mathcal F|}\sum_{i\in\mathcal F} e_i^2}$ |
| Pixel accuracy @0.5 °C | $100\%\times \dfrac{|\{i\in\mathcal F: e_i\le 0.5\}|}{|\mathcal F|}$ |
| Pixel accuracy @1.0 °C | $100\%\times \dfrac{|\{i\in\mathcal F: e_i\le 1.0\}|}{|\mathcal F|}$ |

Tolerances live in `TOL_C = (0.5, 1.0)`. The comparison is restricted to the breast foreground (not the whole frame) because the rendered temperature is only physically meaningful inside the silhouette.

### 13.3 Granularity: per-view, per-patient, and cohort

- **Per-view, per-patient** → every `(patient, view)` row is written to `thermal_pixel_accuracy_per_view.csv` (columns: `patient_id, split, view, dice, iou, thermal_MAE_C, thermal_RMSE_C, pix_acc_0.5C_%, pix_acc_1.0C_%`) and annotated on the audit figure.
- **Per-patient** → the five views are averaged, printed live, and shown in the figure title.
- **Cohort** → a per-split and overall summary table. Dice / IoU / MAE are **macro-averaged** over patients; **pixel accuracy is the true micro value** (within-tolerance pixels ÷ all foreground pixels across the split):

$$\text{Acc}^{\text{split}}_{@\tau} = 100\%\times\frac{\sum_{\text{patients}}\sum_{\text{views}} \big|\{i\in\mathcal F: e_i\le\tau\}\big|}{\sum_{\text{patients}}\sum_{\text{views}} |\mathcal F|}$$

The console summary (also captured in `nerf_run.log`) looks like:

```
=================== COHORT PERFORMANCE ===================
 split    n  |   Dice   |   IoU    | MAE °C | Acc@0.5°C | Acc@1°C
 train   85  |  ......  |  ......  |  ....  |   ....    |  ....
 val     18  |  ......  |  ......  |  ....  |   ....    |  ....
 test    19  |  ......  |  ......  |  ....  |   ....    |  ....  <-- HELD-OUT
 ALL    122  |  ......  |  ......  |  ....  |   ....    |  ....
=========================================================
```

### 13.4 Figures and files

Each per-patient figure `audit_<split>_<id>.png` is a 3×5 grid:

- **Row 1:** GT masks (5 views)
- **Row 2:** rendered opacity (Dice / IoU in title, green if Dice > 0.85, else red)
- **Row 3:** rendered thermal (MAE °C and Acc@1 °C in title)

Cohort figures and tables:

| File | Content |
|---|---|
| `split_summary.png` | mean Dice/IoU **and** thermal pixel-accuracy, per split |
| `cohort_dice_by_split.png` | per-patient Dice, bars coloured by split (test in green) |
| `cohort_thermal_mae_by_split.png` | per-patient thermal MAE, coloured by split |
| `thermal_pixel_accuracy_per_view.csv` | full per-view, per-patient table |
| `training_curves.png` | 2×3: train loss, val Dice, val IoU, joint metric, val thermal MSE, LR |

The audit is wrapped in `try/except` (with traceback) so the training-curve plot and DDP cleanup still run if it fails.

**Output directory layout:**

```
finalized/thermamnerf_outputs_finalized/
  thermamnerf_best.pth          ← loaded for inference / Stage 4   {encoder, mlp, epoch}
  thermamnerf_latest.pth        ← crash recovery
  sanity_check_sample.png       ← first-patient data check (startup)
  training_curves.png           ← 2×3 metric panels
  audit_train_Patient_*.png
  audit_val_Patient_*.png
  audit_test_Patient_*.png      ← held-out
  split_summary.png
  cohort_dice_by_split.png
  cohort_thermal_mae_by_split.png
  thermal_pixel_accuracy_per_view.csv
```

> **Reporting rule.** Quote the **test** row of the cohort table (and the green bars / test column in the figures) as the model's performance. Train and val are shown only for reference and overfitting diagnosis.