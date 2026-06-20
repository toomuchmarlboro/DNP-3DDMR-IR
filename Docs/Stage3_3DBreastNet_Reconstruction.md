# Stage 3 — TherMAM-NeRF: Joint Geometry + Thermal Field Reconstruction

**Current source:** `TherMAM-NeRF/thermamnerf_v2.9.py` (training), `TherMAM-NeRF/Thermamnerf_PINN_v3.ipynb` (inference / extraction)
**Predecessor (superseded):** `UNET_Segmentation/3DBreastnet/breastnet3d_v5.ipynb`

> *Note: this file documents the **current** reconstruction stage (TherMAM-NeRF). The earlier 3DBreastNet approach is retained in §2 as the documented predecessor it replaced. The filename is kept for continuity and may be renamed.*

---

## 1. Objective

This stage reconstructs, for each patient, a **watertight 3D breast surface** *and* a **continuous surface-temperature field**, from the five aligned infrared views. Unlike a pure shape reconstructor, TherMAM-NeRF learns a single neural field that outputs, at every point in space, both an occupancy density $\sigma$ and a temperature $T$.

The output of this stage is the input contract for the Stage 4 bioheat solver:

1. A **3D anatomical domain** $\partial\Omega$ over which the Pennes bioheat PDE is posed (skin surface + chest wall).
2. A **per-vertex surface temperature** $T_{\text{measured}}$ — the boundary data the inverse problem is matched against.

---

## 2. From 3DBreastNet to TherMAM-NeRF (why the method changed)

The predecessor, **3DBreastNet** (`breastnet3d_v5`, best validation Dice **0.8427**), learned a $128^3$ occupancy volume from the five **binary masks** alone, supervised by multi-view differentiable-projection (visual-hull) consistency. Temperature was then attached *after the fact* by a Lambertian-weighted projection of the raw thermal images onto the extracted mesh.

This is sufficient for shape, but **not** for bioheat inversion, which needs a *temperature field that is part of the reconstruction*, not a post-hoc paint job:

- 3DBreastNet discards the thermal signal during reconstruction — geometry is learned only from silhouettes.
- Its surface temperature inherits projection seams and occlusion artefacts from the blending step.

**TherMAM-NeRF** instead learns geometry and temperature *jointly* as one continuous field. The bioheat solver then consumes the reconstruction's own temperature output directly. 3DBreastNet remains the reference for the occupancy / visual-hull formulation and the asymmetry features.

---

## 3. Architecture

TherMAM-NeRF is a neural field $f_\Theta:\ \mathbb{R}^3 \to (\sigma, T)$ conditioned on the five input views (~205k parameters total). It has two components.

### 3.1 Siamese view encoder

A single shared 2D CNN $E$ is applied to each view $k$. Its input is the two-channel stack of the normalised thermal image and its U-Net mask, and its output is masked to the breast region:

$$ F_k = E\big([\,\tilde I_k,\ M_k\,]\big) \odot M_k \in \mathbb{R}^{32 \times H \times W} $$

The encoder is four convolutions (channels $2\to16\to32\to32\to32$) with GroupNorm and ReLU. Sharing weights across views ("Siamese") means a single feature extractor must explain all five viewpoints, which regularises the learned features.

### 3.2 Positional encoding

A query point $\mathbf{x}\in[-1,1]^3$ is lifted to a high-frequency Fourier feature vector with $L=8$ bands, enabling the MLP to represent sharp geometric and thermal detail:

$$ \gamma(\mathbf{x}) = \Big[\,\sin(2^{\ell}\pi x_d),\ \cos(2^{\ell}\pi x_d)\,\Big]_{d\in\{x,y,z\},\,\ell=0\dots L-1} \in \mathbb{R}^{6L} $$

A windowing coefficient on the bands (coarse-to-fine annealing) is supported during training to stabilise early optimisation.

### 3.3 Multi-view feature aggregation

For a 3D query point, the network gathers evidence from all five feature maps. The point is rotated into each view's camera frame about the vertical axis (angles $0^\circ,\pm45^\circ,\pm90^\circ$) and projected; the feature map is sampled by bilinear interpolation:

$$ x_r = \cos\theta_k\, x + \sin\theta_k\, z,\qquad y_r = y,\qquad \mathbf{g}_k = \text{grid\_sample}\big(F_k,\,(x_r,y_r)\big) $$

The per-view samples are aggregated into a **mean** and a **variance** across views (the variance encodes multi-view *disagreement*, a useful cue for surface ambiguity):

$$ \mathbf{a}(\mathbf{x}) = \Big[\ \operatorname{mean}_k \mathbf{g}_k,\ \ \operatorname{var}_k \mathbf{g}_k\ \Big] \in \mathbb{R}^{64} $$

### 3.4 Field decoder

A skip-connection MLP (256-wide) maps the concatenated positional encoding and aggregated features to the two field outputs:

$$ (\sigma, T) = \text{MLP}\big([\,\gamma(\mathbf{x}),\ \mathbf{a}(\mathbf{x})\,]\big),\qquad \sigma = \text{softplus}(\cdot)\ \ge 0,\quad T = \text{sigmoid}(\cdot)\in[0,1] $$

$\sigma$ is the occupancy density (geometry); $T$ is the normalised temperature, later mapped back to °C.

---

## 4. Volume extraction and mesh construction

At inference the field is evaluated on a dense $128^3$ grid (chunked to fit GPU memory), yielding a density grid $\sigma$ and a temperature grid $T$.

### 4.1 Geometry

1. **Smooth** the density grid (Gaussian, $\sigma_{\text{geo}}$) to suppress voxel noise without touching the thermal field.
2. **Marching Cubes** at iso-level $\tau = 0.3$ extracts the raw surface.
3. **Axis reorder** to align array indices with the physical $(X,Y,Z)$ frame, where $X$ is the right–left (RL↔LL) axis matching the frontal view.
4. **Trimesh repair**: keep the largest connected component, `fill_holes`, `fix_normals`, and optional Laplacian smoothing → a watertight mesh with outward vertex normals (the normals are required for the Stage 4 Robin skin boundary condition).

### 4.2 Per-patient physical scale calibration

A reconstruction in $[-1,1]^3$ must be scaled to millimetres before any physics is posed (the Laplacian scales as $1/\text{extent}^2$, so a wrong scale corrupts the PDE). Rather than a fixed `breast_radius_mm`, the scale is calibrated **per patient** from the imaging geometry of the FLIR SC620 used in DMR-IR (1.0 m standoff, 24° horizontal field of view ⇒ a 425 mm scene across the 640-px sensor):

$$ \text{breast\_radius\_mm} = \tfrac{1}{2}\, w_{\text{px}} \cdot \frac{425}{\text{img\_size}} $$

where $w_{\text{px}}$ is the breast width measured in the frontal mask. *Example:* for Patient 1 this yields **157.7 mm** (mask 95 px wide) versus the previous hardcoded 70 mm — a correction that removes a large, patient-dependent error from the downstream PDE residual.

### 4.3 Surface temperature

The temperature grid is sampled at each cleaned mesh vertex by trilinear interpolation (`map_coordinates`) and denormalised to °C using the frontal view's calibration anchor $(T_{\min}, T_{\max})$:

$$ T_{\text{measured}}(\mathbf{v}) = \hat T(\mathbf{v}) \cdot (T_{\max} - T_{\min}) + T_{\min} $$

*Example range (Patient 1):* $T_{\text{measured}} \in [21.9,\ 33.5]\,^\circ$C.

### 4.4 Two read-outs of the same field (ray-casting vs. grid query)

The neural field $f_\Theta:\ \mathbb{R}^3 \to (\sigma, T)$ is **not an image** — it returns occupancy and temperature at a single 3-D point. It is read out in two distinct ways, and confusing them is easy:

| Read-out | How the field is queried | Produces | Used in |
|---|---|---|---|
| **Ray-casting** (training) | march rays through the volume, composite $\sigma$ and $T$ along each ray | 2-D opacity / temperature maps | §9 — supervision against the five IR views |
| **Grid query** (extraction) | evaluate $f_\Theta$ on the dense $128^3$ voxel lattice directly — no rays, no compositing | $\sigma$ and $T$ **volumes** | §4 — mesh + per-vertex temperature |

Ray-casting (§9) is only the *differentiable bridge* used at training time to compare the 3-D field against 2-D ground truth; it is the act of "looking at" the field from a viewpoint. The final 3-D product does **not** ray-cast — it queries the field on the voxel grid and runs Marching Cubes.

**Consequence for surface temperature.** Because temperature is part of the learned field, the per-vertex $T_{\text{measured}}$ is read **straight from the temperature grid** by trilinear interpolation (§4.3) — it is *not* back-projected from the raw input TIFFs through a visibility/occlusion test, and there is **no ray-cast visibility step and no KNN occlusion in-fill**. This is the key contrast with the 3DBreastNet predecessor (§2), which *painted* temperature onto the mesh by Lambertian-weighted projection of the thermal images and therefore inherited projection seams and occlusion artefacts. Here, occluded regions simply take the value the joint field assigns them, with no separate interpolation pass.

---

## 5. The trust boundary (critical for Stage 4)

TherMAM-NeRF produces a *volumetric* field, but **only its surface values are trusted.** A field trained on outside-in IR views cannot observe the breast interior — its interior temperature is an unconstrained extrapolation. The bioheat solver therefore consumes only $T_{\text{measured}}$ on the skin and lets the **physics** (Pennes PDE + boundary conditions) determine the interior. This separation is what makes Stage 4 a physically-grounded inverse rather than a fit to invented interior data.

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

### 7.1 Configuration (CFG)

All hyperparameters are kept in a single `CFG` dictionary so the full run is reproducible from one place.

| Parameter | Value | Notes |
|---|---|---|
| `img_size` | 128 | Both encoder feature maps and ray grids are 128×128 |
| `n_views` | 5 | RL, RO, F, LO, LL |
| `view_angles_deg` | [−90, −45, 0, +45, +90] | Camera rotation from frontal; +90 = Left Lateral |
| `feat_channels` | 32 | Encoder output depth; feature vector per point = 64 (mean+var) |
| `pos_enc_L` | 8 | Fourier bands; positional encoding dim = 3×2×8 = 48 |
| `mlp_hidden` | 256 | Width of every layer in the MLP |
| `mlp_layers` | 4 | Depth excluding output heads |
| `n_samples` | 256 | Quadrature points per ray |
| `near / far` | −1.0 / +1.0 | Ray integration range in normalised coordinates |
| `density_scale` | 10.0 | Multiplies $\sigma$ before $\alpha$ computation (higher = sharper surface) |
| `freq_warmup_epochs` | 50 | Epochs over which positional encoding is annealed coarse→fine |
| `batch_size` | 1 | One patient per gradient step |
| `n_epochs` | 1 000 | Total training epochs |
| `lr` | 5×10⁻⁴ | Adam initial learning rate |
| `lambda_dice` | 1.0 | Geometry Dice loss weight |
| `lambda_bg` | 2.0 | Background suppression weight (at full ramp) |
| `lambda_thermal` | 20.0 | Thermal MSE weight |
| `lambda_tv` | 0.01 | 3-D total-variation weight (at epoch 1) |
| `lambda_entropy` | 0.1 | Ray opacity entropy weight (at full ramp) |
| `n_rays` | 3 072 | Randomly sampled rays per view per step |
| `mc_threshold` | 0.3 | Marching-cubes iso-level on the extracted $\sigma$ grid |
| `mc_resolution` | 128 | Grid resolution for volume extraction |
| `use_amp` | True | PyTorch automatic mixed precision (CUDA only) |
| `use_grad_checkpoint` | True | Gradient checkpointing in MLP to save VRAM |

### 7.2 Dataset and data loading

**Directory layout:**

```
data/
  organized_by_patient/        # TIFF_DIR — raw thermal images
    Patient_1/
      benign/
        right_lateral.tiff
        right_oblique.tiff
        frontal.tiff
        left_oblique.tiff
        left_lateral.tiff
  organized_by_patient_unet/   # UNET_DIR — pre-computed UNet binary masks
    Patient_1/
      benign/
        *.png
```

`discover_patients_split` scans both trees and only keeps patients that have **all five views** in both the thermal and the mask directories. View keys are assigned by filename keyword matching (`right later` → `RL`, `frontal`/`anterior` → `F`, etc.). The final list is sorted by patient number and printed.

`BreastThermDataset.__getitem__` loads and pre-processes one patient:

1. Load each TIFF at float32 precision (`tifffile.imread`), resize bilinearly to `img_size`.
2. Min–max normalise to $[0,1]$ (saving `tmin`/`tmax` for later denormalisation).
3. Load the UNet mask PNG, binarise at 0.5.
4. Return stacked tensors of shape `[V, H, W]` for both normalised thermals and masks.

### 7.3 Train / val split

```python
random.seed(42)
random.shuffle(patients)
split = int(0.8 * len(patients))
train_patients = patients[:split]   # 80 %
val_patients   = patients[split:]   # 20 %
```

The 80/20 split is applied once over the full **122-patient** list. The random seed makes it deterministic. Validation data are only loaded on rank 0 (`IS_MAIN`).

### 7.4 Distributed Data Parallel (DDP)

The script is DDP-ready and is launched with:

```bash
torchrun --nproc_per_node=2 thermamnerf_v2.9.py
```

`setup_distributed()` reads `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` from the environment variables set by `torchrun` and calls `dist.init_process_group(backend='nccl')`. Each process gets its own CUDA device (`LOCAL_RANK`). A `DistributedSampler` shards the training dataset across processes; validation runs only on rank 0 to avoid duplicate metric accumulation. At the end of each epoch, training losses are all-reduced with `dist.all_reduce` before printing.

---

## 8. Loss Functions

All five losses are computed **per view** and averaged over the five views before combining with their weights.

### 8.1 Dice Loss (geometry)

The soft Dice loss drives the rendered opacity $\hat M$ to match the binary UNet mask $M$:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i \hat M_i M_i + \varepsilon}{\sum_i \hat M_i + \sum_i M_i + \varepsilon}, \qquad \varepsilon = 10^{-6}$$

It is robust to class imbalance (the breast silhouette is sparse against a large image) because it normalises by both foreground areas instead of treating every pixel equally.

### 8.2 Background Suppression Loss

Background pixels (where $M \le 0.5$) must have near-zero rendered opacity. Squared suppression:

$$\mathcal{L}_{\text{bg}} = \frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}} \hat M_i^2, \qquad \mathcal{B} = \{i : M_i \le 0.5\}$$

Without this, rays outside the breast silhouette accumulate spurious density ("floaters"). The weight ramps from 0 to `lambda_bg` linearly over the first 50 epochs so that geometry has a chance to form before the background is aggressively suppressed.

### 8.3 Thermal MSE Loss

Temperature is only supervised inside the breast region (foreground pixels):

$$\mathcal{L}_{\text{thermal}} = \frac{1}{|\mathcal{F}|}\sum_{i\in\mathcal{F}} \big(\hat T_i - \tilde I_i\big)^2, \qquad \mathcal{F} = \{i : M_i > 0.5\}$$

where $\tilde I_i$ is the min–max normalised thermal image value and $\hat T_i$ is the rendered normalised temperature. Computing it only on the foreground avoids penalising the background temperature, which has no physical meaning.

### 8.4 Total Variation Loss (3-D)

To regularise the volumetric density field against spatial noise, a TV penalty is computed on a coarse $32^3$ sub-grid sampled each step:

$$\mathcal{L}_{\text{tv}} = \underbrace{\frac{1}{N}\sum|\Delta_x \sigma|}_{\text{x-direction}} + \underbrace{\frac{1}{N}\sum|\Delta_y \sigma|}_{\text{y-direction}} + \underbrace{\frac{1}{N}\sum|\Delta_z \sigma|}_{\text{z-direction}}$$

The weight decays exponentially: `lambda_tv(epoch) = lambda_tv × max(0.5, exp(−0.015 × epoch))`, halving roughly every 46 epochs but flooring at half its initial value to keep edges smooth throughout training.

### 8.5 Entropy Loss

To snap semi-transparent voxels to either fully occupied or fully empty (preventing "comb" artefacts at the breast boundary):

$$\mathcal{L}_{\text{entropy}} = \frac{1}{N}\sum_i \hat M_i(1 - \hat M_i)$$

This is minimised when $\hat M_i \in \{0,1\}$. The weight ramps from 0 to `lambda_entropy` linearly over the first 100 epochs.

### 8.6 Combined loss

$$\mathcal{L} = \lambda_{\text{dice}}\,\mathcal{L}_{\text{Dice}} + \lambda_{\text{bg}}\,\mathcal{L}_{\text{bg}} + \lambda_{\text{thermal}}\,\mathcal{L}_{\text{thermal}} + \lambda_{\text{tv}}\,\mathcal{L}_{\text{tv}} + \lambda_{\text{entropy}}\,\mathcal{L}_{\text{entropy}}$$

| Weight | Value | Schedule |
|---|---|---|
| $\lambda_{\text{dice}}$ | 1.0 | Fixed |
| $\lambda_{\text{bg}}$ | 2.0 | Linearly ramps over epochs 1–50 |
| $\lambda_{\text{thermal}}$ | 20.0 | Fixed |
| $\lambda_{\text{tv}}$ | 0.01 → ≥0.005 | Exponential decay, floor at 50% |
| $\lambda_{\text{entropy}}$ | 0.1 | Linearly ramps over epochs 1–100 |

`lambda_thermal = 20.0` is large relative to the other terms because thermal MSE naturally sits in the range ~0.05 whereas Dice loss sits near ~0.1; the high coefficient keeps thermal gradient magnitudes comparable to the geometry gradient magnitudes.

---

## 9. Volume Rendering

### 9.1 Ray sampling

For each of the five views at angle $\theta_k$, rays are cast from a parallel-projection camera. The camera direction is $\mathbf{d} = (-\sin\theta_k,\ 0,\ \cos\theta_k)^T$. The origin of each ray is its pixel's 2-D position in the camera plane, expressed in the rotated frame. During training, `n_rays = 3072` rays are **randomly sampled** per view per step (not the full $128\times128 = 16384$), with stratified noise added to $t$ values to avoid aliasing.

### 9.2 Transmittance and weights

Along each ray, $n_{\text{samples}} = 256$ quadrature points are placed uniformly in $[{-1}, 1]$ (normalised space). For sample $i$ with interval $\delta_i$:

$$\alpha_i = 1 - \exp\!\big(-\sigma_i \cdot s \cdot \delta_i\big), \qquad s = \text{density\_scale} = 10$$

$$T_i = \prod_{j < i} (1 - \alpha_j), \qquad w_i = \alpha_i\, T_i$$

### 9.3 Rendered outputs

$$\hat M = \sum_i w_i \qquad \text{(rendered opacity / mask)}$$

$$\hat T = \frac{\sum_i w_i^{\,\text{sg}} \cdot T_i^{\text{field}}}{\sum_i w_i^{\,\text{sg}} + \varepsilon} \cdot \mathbb{1}\!\left[\hat M^{\text{sg}} > 0.05\right]$$

where superscript ${\text{sg}}$ denotes `detach()` (stop-gradient). The rendered temperature is masked to zero outside the breast silhouette.

### 9.4 Detached weights for thermal (critical design decision)

In `volume_render`, the occupancy weights are **detached before multiplying the temperature field**:

```python
rendered_temp = (weights.detach() * T_field).sum(dim=1)
```

This is intentional. If the weights were not detached, the thermal loss would also back-propagate through $\sigma$ (which controls the weights), corrupting the geometry signal with purely thermal gradients. By detaching, the thermal loss only updates the $T$ branch of the MLP, while the $\sigma$ branch is updated only by Dice + BG + TV + Entropy. The two heads are thus **trained independently** despite sharing the MLP trunk.

---

## 10. View-Specific Depth Cropping

### 10.1 The ribcage problem

The human ribcage is curved. From a strict 90° side view (Right Lateral or Left Lateral), the *far* breast protrudes slightly behind the *near* breast along the viewing axis. Because the UNet GT masks only annotate the **near** breast in lateral views, the NeRF would otherwise be penalised for correctly predicting the visible outline of the far breast.

### 10.2 Implementation

A simple X-coordinate mask zeros the density field for the unwanted breast before the volume integral:

```python
if angle <= -80:   # RL — camera is to the patient's right, far breast is left (X > 0)
    sigma = sigma.masked_fill(pts_flat[..., 0:1] > 0.1, 0.0)
elif angle >= 80:  # LL — camera is to the patient's left, far breast is right (X < 0)
    sigma = sigma.masked_fill(pts_flat[..., 0:1] < -0.1, 0.0)
```

The threshold $|X| = 0.1$ (normalised units) is a small buffer around the mid-sagittal plane. This masking is applied **during both training and inference** (`render_view`) to keep the rendered silhouette consistent with the GT masks.

---

## 11. Optimiser and Scheduler

### 11.1 Adam + Cosine Annealing

```
Optimiser: Adam(lr=5e-4, betas=(0.9, 0.999))
Scheduler: CosineAnnealingLR(T_max=1000, eta_min=lr*0.05 = 2.5e-5)
```

The cosine schedule halves the learning rate smoothly over the full 1 000 epochs, reaching 2.5×10⁻⁵ at epoch 1 000. This prevents the late-stage oscillations of a fixed LR without the abrupt decay of step schedules.

Gradient clipping is applied before each step:

```python
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
```

### 11.2 Dynamic lambda scheduling

Three loss weights are not constant — they change each epoch via `cfg_step`:

| Weight | Schedule formula | Rationale |
|---|---|---|
| `lambda_tv` | $\lambda_{\text{tv}} \times \max(0.5,\, e^{-0.015\cdot\text{epoch}})$ | Strong smoothing early; tails off but never disappears |
| `lambda_bg` | $\lambda_{\text{bg}} \times \min(1.0,\, \text{epoch}/50)$ | Ramps over 50 epochs so geometry forms before background is aggressively crushed |
| `lambda_entropy` | $\lambda_{\text{entropy}} \times \min(1.0,\, \text{epoch}/100)$ | Slow ramp prevents comb artefacts from forming at epoch 1 |

### 11.3 Coarse-to-fine frequency annealing

The positional encoding windowing parameter $\alpha$ increases linearly from 0 to $L=8$ over the first 50 epochs:

$$\alpha(\text{epoch}) = \min\!\left(\frac{L \cdot \text{epoch}}{\text{warmup\_epochs}},\ L\right)$$

Each Fourier band $\ell$ is gated by a cosine window:

$$w_\ell = \frac{1}{2}\left(1 - \cos\!\left(\pi\,\text{clamp}(\alpha - \ell,\, 0,\, 1)\right)\right)$$

At epoch 0 only the zeroth band (lowest frequency, coarsest) is active. Bands unlock one by one, allowing the network to first learn the global breast shape before committing to fine texture. This avoids early entrapment in the "solid block" failure mode.

---

## 12. Validation and Checkpointing

### 12.1 Joint metric

Validation runs on rank 0 after every training epoch. The **joint metric** is:

$$J_{\text{val}} = \mathcal{L}_{\text{Dice}}^{\text{val}} + \frac{\lambda_{\text{thermal}}}{10}\,\mathcal{L}_{\text{thermal}}^{\text{val}}$$

The scaling by $\lambda_{\text{thermal}}/10 = 2.0$ keeps the thermal contribution proportional to its weight in the training objective without dominating. A **lower** joint metric is better. This is the quantity monitored for early stopping / best-checkpoint selection.

From the joint metric, Dice score and IoU are also reported for interpretability:

$$\text{Dice} = 1 - \mathcal{L}_{\text{Dice}}, \qquad \text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$$

### 12.2 Checkpoint strategy

Two checkpoints are saved each epoch on rank 0:

| File | Condition | Purpose |
|---|---|---|
| `thermamnerf_best.pth` | `J_val < best seen so far` | The model loaded for inference and Stage 4 |
| `thermamnerf_latest.pth` | Every epoch | Recovery from crashes |

Both contain `{'encoder': state_dict, 'mlp': state_dict}`. DDP wrapper keys (`module.`) are stripped by `encoder_base` / `mlp_base` references before saving.

Resume is controlled by the `RESUME` flag and `START_EPOCH`; the scheduler is fast-forwarded by replaying `scheduler.step()` for the skipped epochs to restore the correct learning rate.

---

## 13. Post-training Projection Audit

After the training loop, the script automatically reloads `thermamnerf_best.pth` and runs a **full-cohort projection audit** (train + val combined). For each patient it:

1. Renders the mask and temperature for all five views using the final $\alpha = L = 8$ (fully annealed).
2. Computes per-view binary Dice and IoU against the GT UNet mask.
3. Saves a `3×5` figure (`audit_{patient_id}.png`) to `thermamnerf_outputs2.9/`:
   - **Row 1:** GT masks (5 views)
   - **Row 2:** Rendered opacity (Dice / IoU in title, green if >0.85, red otherwise)
   - **Row 3:** Rendered thermal (normalised)
4. Aggregates all per-patient scores and saves `cohort_performance_plot.png` (side-by-side Dice and IoU bars for every patient).

The audit is wrapped in a `try/except` so training curve saving and process cleanup still run if it fails (e.g., insufficient GPU memory at audit time).

**Typical output directory layout:**

```
TherMAM-NeRF/thermamnerf_outputs2.9/
  thermamnerf_best.pth          ← loaded by mukhmetov_recover.py / run_cohort.py
  thermamnerf_latest.pth
  sanity_check_sample.png       ← first-patient data check (saved at startup)
  training_curves.png           ← 2×2 plot: train loss, val Dice, joint metric, thermal MSE
  audit_Patient_1.png
  audit_Patient_2.png
  ...
  cohort_performance_plot.png
```
