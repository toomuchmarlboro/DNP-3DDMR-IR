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
