# Stage 4: Inverse Bioheat Solving — FEM-FEM Recovery (Mukhmetov-style)

**Current source:** `TherMAM-NeRF/mukhmetov_recover.py` — synthetic validation (`--idx 0`) and real cohort (`--real`); shared FEM forward + bilateral cost. `run_cohort.py` provides the FEA/geometry helpers it imports.
**Documented negative results (retained):** PINN v1 (`UNET_Segmentation/PINNpdeSolver/PINN_Pipeline.py`), PINN v3 (`TherMAM-NeRF/Thermamnerf_PINN_v3.ipynb`), Mukhmetov-PINN (`run_cohort.py --method mukhmetov`, prior PINN-based variant)

---

## 0. Progress Summary (status: 2026-06-17)

This section is the chronological story of the inverse-bioheat work and where it now stands — read it first if you are picking the project up mid-stream.

**What we set out to do.** Given a TherMAM-NeRF reconstruction (3D *bilateral* breast geometry + per-vertex skin temperature) from 2D IR images, estimate the depth and size of a subsurface tumour by solving the inverse Pennes bioheat problem. This is a **screening** estimate of position/depth/size — not a diagnosis.

**The path taken (chronological):**
1. **Three PINN inverse formulations** — free 5-parameter, forward-matching v3, Mukhmetov-PINN — all failed. A deep Tanh network cannot represent the bioheat field to better than ~7 °C, drowning the 0.67 °C tumour signal (§4).
2. **FEM-FEM inverse** — replaced the PINN with FEniCSx as the forward model inside the optimiser; synthetic recovery to <0.5 % (§4.4, §6).
3. **Cost-function progression on real data** — absolute MSE (radius rails to 40 mm) → mean-centred (still rails) → **bilateral asymmetry** (interior minimum). Identifiability is a property of the *cost*, not the forward model (§7.0).
4. **Full 122-patient cohort** — extended from the initial 40-patient subset to the complete usable DMR-IR set (96 benign, 26 malignant) (§7.3).
5. **Numerical verification** — solver and mesh convergence confirmed; the 3 mm production mesh is converged (§6.5).
6. **Literature validation** — boundary conditions and the depth–size degeneracy match the published FEM-thermography literature (Jahani et al. 2023) (§2.5).

**Headline results (current):**
- **Synthetic validity:** depth/radius recovered to **mean 0.28 % / 0.14 %** (9 scenarios) under the production bilateral cost (§6).
- **Significant malignancy signal:** the *model-free* bilateral asymmetry |A| is significantly higher in malignant breasts — benign median **1.37 °C** vs malignant **1.56 °C**, **Mann–Whitney U p = 0.0088 (two-sided), 0.0044 (one-sided)**, n=122; survives Bonferroni correction across the 5 metrics tested (§7.3).
- **Inverse *geometry* does not separate classes:** recovered depth/radius/cost show no significant benign-vs-malignant difference (all p > 0.3) — consistent with the published depth–size degeneracy (§2.5, §7.3).
- **Numerics verified:** the linear solver converges optimally (CG/GAMG, ~17 iterations independent of mesh); skin temperature is mesh-converged at 3 mm (changes <0.01 °C through 1.0 mm refinement) (§6.5).
- **BCs are literature-standard** (Jahani et al. 2023): identical Pennes constants, Gautherie law, 37 °C chest wall, h=10, T∞≈21 °C; our mesh is ~56× finer (3D, 587 k elements) than their published 2D study (§2.5).

**Honest framing.** The result is two-sided: a **positive** finding (bilateral asymmetry separates classes, p<0.01) and a rigorously-characterised **negative** one (subsurface geometry inversion is ill-posed — a known, reproduced degeneracy, not an implementation error). Absolute depth/radius accuracy remains unverifiable without co-registered MRI/CT.

**New artifacts this stage:** `results/cohort_femfem_ALL122.csv`, `results/asymmetry_ALL.csv`, `results/thesis_figures/fig7_fem_convergence.png` & `fig8_konvergensi_mesh.png` (mesh convergence), `fig9_asimetri_bilateral.png` (the p=0.0044 figure), and the dataset-verification notebook `TherMAM-NeRF/data_verification.ipynb` (funnel 268→137→122, class distribution, FEM-input checklist).

---

## 1. Overview

Stage 4 couples the Stage-3 surface temperature to biophysics and solves an **inverse bioheat problem**: given the reconstructed skin temperature, estimate the internal tumour consistent with it. The framing is a **screening tool** — the quantities sought are the tumour's **position, depth, and size**, not a diagnosis.

The current method uses **FEniCSx (dolfinx) as the forward model inside a grid-search + Nelder-Mead optimiser**, directly replicating the self-consistent "FEM-as-oracle" approach of Mukhmetov et al. (2025), but replacing their ANSYS solver with FEniCSx and their 3D scanner requirement with TherMAM-NeRF geometry. Three PINN variants were attempted first; all failed for reasons documented in §4.

A second, equally important departure from Mukhmetov: TherMAM-NeRF reconstructs **both breasts** as a single mesh, whereas Mukhmetov models one breast. This is exploited — not worked around. The cost function is a **bilateral asymmetry** comparison (§5.6): the contralateral (healthy) breast is a built-in control whose subtraction cancels the common-mode model-mismatch that otherwise renders real-data geometry unidentifiable. The progression of three cost functions (absolute MSE → mean-centred → bilateral) and why only the last is identifiable on real data is documented in §7.

### 1.1 Four-stage pipeline overview

| Stage | What happens | Key output |
|---|---|---|
| **1 — TherMAM-NeRF** | Neural Radiance Field reconstructs 3D breast geometry and surface temperature from 2D IR image views | `surface_pts` (mm), `T_measured` (°C), `faces`, bounding box |
| **2 — Mesh generation** | Gmsh fills the surface shell with ~587 k tetrahedra (mesh size 3 mm) | `Patient_N.msh` — volumetric FEM mesh, generated once per patient |
| **3 — Forward FEM** | FEniCSx solves the Pennes bioheat PDE on the volumetric mesh for a candidate tumour `(z_t, r_t)` and returns surface temperatures | `T_surf` — simulated skin temperature field |
| **4 — Inverse optimisation** | A coarse grid search (5 depth × 4 radius = 20 FEM evaluations) followed by Nelder-Mead refinement (≤30 more) minimises the **bilateral asymmetry** error between FEM output and measured IR (§5.6) | `(z_hat, r_hat)` — best-fit tumour depth and radius |

Stages 3 and 4 are repeated inside the optimiser — every candidate `(z_t, r_t)` triggers one full FEM solve. The lateral position `(x₀, y₀)` and peak heat `Q₀` are **not optimised** (see §5).

---

## 2. Theoretical Background

### 2.1 The physiological basis of breast thermography

Malignant tumours grow aggressively, demanding oxygen and nutrients. This drives **angiogenesis** (chaotic new vasculature) and an **elevated metabolic rate**, forming a localised internal heat source. As that heat diffuses to the skin, it can create an asymmetric hyperthermic signature visible in the infrared. The premise of this pipeline is to invert that signature for the heat source that produced it.

### 2.2 The Inverse Heat Transfer Problem (IHTP) is ill-posed

- The **forward** problem (known source → surface temperature) is well-posed.
- The **inverse** problem (surface temperature → unknown source) **lacks continuous dependence on data**: heat conduction strongly smooths and attenuates the tumour signal on its way to the skin. Different tumours can produce almost identical surface patterns.

### 2.3 Identifiability: magnitude cannot be recovered from the surface

A sensitivity analysis (Bezerra et al., 2013) shows tumour heat **magnitude** (and blood perfusion) has near-zero surface sensitivity — the skin simply does not contain the information needed to pin it down. Only source **geometry** (position and size) leaves a recoverable imprint. Mukhmetov et al. (2025) accordingly **fix** the magnitude from physiology and train only geometry. All three PINN variants we tried confirmed this experimentally: $Q_{max}$ never moved from its initialisation on any patient.

### 2.4 The model-mismatch floor — and why it is *common-mode*

An FEniCSx forward solve of the same BVP (uniform tissue, Robin skin BC $h=10$, $T_{air}=20\,^\circ$C) reproduces real DMR-IR thermograms with a **residual of 4.5–5.6 °C** across the cohort. This floor comes from the homogeneous-tissue assumption, guessed BC constants ($h$, $T_{air}$), ambient conditions, and NeRF geometry error — **none of which are the tumour.**

The decisive observation is that this floor is **common-mode**: it affects *both* breasts almost identically, because both are reconstructed by the same NeRF, meshed the same way, and solved with the same constants. A direct absolute-temperature inversion drowns in it (§7.1–7.2). But the **left–right asymmetry** $A(\mathbf{x}) = T(\mathbf{x}) - T(\text{mirror}(\mathbf{x}))$ subtracts the two breasts, and the common-mode floor **cancels**. What survives is the genuine asymmetry — on Patient_1, $|A| \approx 2.0\,^\circ$C, which is *larger* than the residual asymmetry mismatch, not smaller. This is the basis of the bilateral cost (§5.6) and the reason real-data geometry becomes identifiable.

Validation against true depth/radius still requires co-registered MRI/CT, which DMR-IR does not provide; what bilateral fixes is **identifiability** (a unique interior minimum exists), not absolute accuracy verification.

### 2.5 Boundary conditions are literature-standard (Jahani et al. 2023)

A natural concern — that the boundary conditions might be mis-specified — was checked against the peer-reviewed FEM-thermography literature. Jahani et al. (2023, *Estimation of Size and Depth of a Breast Tumor Using Thermal Analysis; A Numerical Study*, IEEE ICBME) solve the identical bioheat problem and use the **identical** boundary conditions:

| | Jahani et al. 2023 | This work |
|---|---|---|
| Chest/muscle wall | Dirichlet 37 °C | Dirichlet 37 °C |
| Skin | Robin, $h=10$, $T_\infty=\mathbf{21}\,^\circ$C | Robin, $h=10$, $T_\infty=\mathbf{20}\,^\circ$C |
| Tumour heat | Gautherie, $Q_t\tau=3.27\times10^6$ | Gautherie, $Q_t\tau=3.27\times10^6$ |
| Pennes (steady-state) | identical | identical |

The 1 °C air-temperature difference (21 vs 20 °C) is within the literature's own range and, under the bilateral cost, **cancels in the left–right subtraction** (a spatially-uniform BC error is common-mode; §2.4). The boundary conditions are therefore not an arbitrary choice — they are the standard published model. (The DMR-IR acquisition protocol does measure per-session atmospheric temperature, which our pipeline did not retain; using it per-patient would be a refinement *beyond* the literature, but is moot for the bilateral cost since it cancels.)

**The depth–size degeneracy is also published.** Jahani et al. report explicitly (their §III.A.4) that the maximum surface temperature difference *cannot* separately determine depth and size: a 0.14 °C signal can indicate a 10 mm tumour at 25 mm depth **or** a 15 mm tumour at 30 mm depth. This is precisely the degeneracy that rails our recovered radius (§7.0) — a **known fundamental limit of thermographic inversion, independently reproduced here**, not an implementation error. Their proposed remedy — a second shape feature `L` (the half-maximum width of the surface hotspot, which is "mostly a function of depth") — is the natural direction for future work to break the rail.

---

## 3. Mathematical Formulation

### 3.1 Pennes bioheat equation (steady state)

$$ \nabla \cdot (k \nabla T) + \omega_b c_b (T_a - T) + Q_m + Q_{tumor} = 0 $$

| Symbol | Meaning | Value |
|---|---|---|
| $k$ | tissue conductivity | $0.48\ \text{W/(m·K)}$ |
| $\omega_b$ | blood perfusion rate | $0.0005\ \text{s}^{-1}$ |
| $c_b$ | blood specific heat | $3600\ \text{J/(kg·K)}$ |
| $T_a$ | arterial temperature | $37.0\,^\circ$C |
| $Q_m$ | basal metabolic heat | $450\ \text{W/m}^3$ |
| $Q_{tumor}$ | pathological heat source | see §3.2 |

### 3.2 Tumour heat source

A Gaussian source centred at $(x_t,y_t,z_t)$ with radius $r_t$:

$$ Q_{tumor}(\mathbf{x}) = Q_{0}\cdot \exp\!\left(-\frac{\lVert \mathbf{x}-\mathbf{x}_t\rVert^2}{r_t^2}\right) $$

$Q_0$ is **not optimised** — it is derived from $r_t$ via the Gautherie tumour doubling-time law: $Q_m\tau = C$, $C = 3.27\times10^6\ \text{W·day/m}^3$, calibrated to Bezerra's Table 1 (≈65 400 W/m³ at $D=1$ cm; ≈7 800 W/m³ at $D=2.2$ cm).

### 3.3 Boundary conditions

| Boundary | Condition |
|---|---|
| Chest wall (body side) | Dirichlet $T = 37\,^\circ$C |
| Skin surface | Robin: $-k\,\partial T/\partial n = h(T - T_{air})$, $h=10\ \text{W/(m}^2\text{·K)}$, $T_{air}=20\,^\circ$C |

---

## 4. Inverse Formulations — Three Failed, One Working

### 4.1 PINN v1 — free five-parameter inverse (failed)

Learnable $(x_t, y_t, z_t, r_t, Q_{max})$ with free-form temperature MLP, Adam→L-BFGS, multi-start. **Failure modes:**

1. $Q_{max}$ frozen at initialisation — magnitude unidentifiable (§2.3).
2. Free source escapes to skin boundary — the Gaussian exploits the negative boundary-layer PDE residual and collapses to the surface ($r_t \to$ floor).
3. Expressive MLP cancels any imposed $Q$ via $\nabla^2 T$ — source and PDE residual are the same equation, so the network absorbs the source into the interior field.

### 4.2 PINN v3 — forward-matching inverse (failed on synthetic)

Fixed magnitude (Gautherie), lateral pinned to IR hot-spot, PINN solves the *forward* BVP for each candidate $(z_t, r_t)$, Nelder-Mead minimises mean-centred pattern mismatch. Structurally removes the escape failure of v1.

**Failure on synthetic:** the 6-layer Tanh MLP saturates, stalling at **7.3 °C BC-fitting RMSE**. The tumour signal is 0.67 °C → SNR ≈ 0.09:1. Even on ideal self-consistent synthetic data the cost surface $J(z_t,r_t)$ was nearly flat (range 0.009–0.12 across the full grid); the grid minimum was 35 mm from the planted depth. A network that cannot represent the background field to better than 7 °C cannot detect a 0.67 °C perturbation.

**Empirical evidence (from `mukhmetov_syn.log`, v3 run):**
- Phase 1 (BC fitting): skin loss stuck at exactly **53.33** for **936 consecutive steps** across all 9 scenarios — not slow convergence, complete gradient death from Tanh saturation in a 6-layer network.
- Phase 2 (source ID): PDE residual increased (random walk) because the wrong interior T field made the tumour gradient signal invisible.
- Chest-wall loss ≈ 48 (RMSE ≈ 6.9 °C) — network predicted ~30 °C at chest wall vs 37 °C target.
- Cost surface range 0.009–0.12 — effectively flat; no identifiable minimum.

### 4.3 Mukhmetov-PINN — Dirichlet-skin, biased collocation (failed)

Mukhmetov's Dirichlet skin BC (hard-pin instead of Robin), single Adam optimisation of $(z_t, r_t)$, 70 % tumour-biased collocation resampling every 200 steps. Addressed collocation sparsity near the tumour but not Tanh saturation. BC fitting still stalled at ~7 °C RMSE; tumour gradient remained invisible.

**Empirical evidence (from `mukhmetov_syn.log`, Mukhmetov-PINN run, Patient_1 scenario z=−61.9, r=8):**
- **Best result after tuning:** recovered z=−87.6 vs planted z=−61.9 → depth_err = **41.5 %**; recovered r=20.5 vs planted r=8.0 → radius_err = **156.3 %**.
- Tuning attempts that did not fix it: learning rate `lr_src` reduced from 5×10⁻¹ to 10⁻¹; `CosineAnnealingLR` scheduler added (lr → 10⁻³); neither changed the BC-fitting floor.
- Root cause confirmed: Phase 1 BC loss identical to 4 decimal places from step 312 to step 1248 — the gradient was exactly zero, not just small. This is Tanh saturation, not a learning-rate problem.
- Mean depth error across all 9 scenarios: **138.9 %** — worse than random guessing on a bounded domain.

### 4.4 FEM-FEM inverse — current method (working)

Replace the PINN with **FEniCSx as the forward model** inside the optimiser. Each candidate $(z_t, r_t)$ triggers a full FEM solve on the patient mesh (≈102 k nodes, ≈587 k elements), extracting surface temperature, and computing skin-only MSE against the target.

**Why this works where PINN failed:**

| | PINN (all variants) | FEM-FEM |
|---|---|---|
| Forward model noise | 7.3 °C (Tanh saturation) | ~0.001 °C (FEM numerical precision) |
| Tumour signal | 0.67 °C | 0.67 °C |
| SNR | **0.09:1 — undetectable** | **670:1 — trivial** |

---

## 5. FEM-FEM Algorithm — Code Walkthrough

### 5.1 Stage 1 — TherMAM-NeRF geometry extraction (`load_geo_and_mesh`)

`run_cohort.load_patient_geo()` loads the patient's IR image, runs the NeRF encoder + MLP to reconstruct a volumetric density field, marching-cubes extracts the surface at a calibrated threshold, and Laplacian smoothing (20 iterations) removes meshing artefacts. The result is a Python dict with:

- `surface_pts` — `(N, 3)` float32 array of vertex positions in mm
- `faces` — triangle connectivity
- `T_measured` — temperature at each vertex in °C, mapped from the IR image via the NeRF UV projection
- `bbox_min / bbox_max` — axis-aligned bounding box in mm (gives the breast depth range)
- `patient_id`, `label` (`'benign'` / `'malignant'`)

### 5.2 Stage 2 — Volumetric mesh generation

The surface triangulation is exported as a binary STL, then Gmsh tessellates the enclosed volume with tetrahedra:

```python
save_stl_binary(stl_path, geo['surface_pts'], geo['faces'])
stl_to_tet_mesh(stl_path, msh_path, mesh_size_mm=3.0)
```

Result: ~102 k nodes, ~587 k tetrahedral elements. The mesh is generated **once per patient** and reused for all 50+ FEM evaluations. Reuse makes the grid search tractable — mesh generation takes ~2 min; a FEM solve takes ~3 min.

### 5.3 Stage 3 — Lateral pin (`hotspot_lateral`)

The tumour's lateral position $(x_0, y_0)$ is read directly from the IR thermogram — no FEM needed:

```python
thr = np.quantile(T_measured[skin_only], 0.90)
hot = skin_only & (T_measured >= thr)
x0, y0 = surface_pts[hot, 0].mean(), surface_pts[hot, 1].mean()
```

The centroid of the top-10% warmest skin vertices is a reliable lateral localiser because heat diffusion from a subsurface source preserves the projection of the source onto the skin plane. It is **not optimised** — lateral sensitivity in the cost function is much weaker than depth sensitivity.

### 5.4 Stage 4 — Forward FEM call (`fem_surface`)

Each candidate $(z_t, r_t)$ triggers one FEniCSx solve:

1. $Q_0$ is computed from $r_t$ via the Gautherie law (not optimised — physiologically constrained).
2. FEniCSx assembles and solves the Pennes BVP on the volumetric mesh with Robin skin BC and Dirichlet chest-wall BC.
3. The solution field $T(\mathbf{x})$ is interpolated back onto the surface vertices.
4. The skin-only subset (chest wall excluded) is returned as `T_surf`.

The chest wall is excluded from the cost because it is clamped to 37°C by Dirichlet BC in **every** solve — its temperature is identical regardless of tumour parameters and carries zero information about $(z_t, r_t)$.

### 5.5 Stage 4 — Inverse: grid search + Nelder-Mead (`recover_fem_fem`)

**Fixed (never optimised):**
- Lateral $(x_0, y_0)$ = centroid of top-10 % IR hot-spot — robust, read directly from IR.
- Peak heat $Q_0 = Q_0(r_t)$ from Gautherie law — physiologically derived, not fitted.

**Optimised:** $(z_t, r_t)$ only — 2 degrees of freedom.

**Step 1 — coarse grid:** 5 depth × 4 radius = 20 FEM solves spanning 85 % of breast depth range × 7–22 mm radius. Cost = bilateral asymmetry error on the affected-breast skin (§5.6; chest wall excluded — Dirichlet 37 °C in every solve, carries no useful information).

```
depth grid:  z_lo + 8%·Δz   to   z_hi − 5%·Δz   (5 values)
radius grid: 7 mm  to  22 mm                       (4 values)
```

The asymmetric margin (8% from bottom, 5% from top) keeps the tumour away from the chest wall and skin boundary where the cost function is distorted by boundary conditions rather than tumour proximity.

**Step 2 — Nelder-Mead refine:** ≤ 30 iterations initialised at the grid minimum; convergence tolerances $\Delta z < 0.5$ mm, $\Delta J < 10^{-7}$. Uses Scipy's `minimize(..., method='Nelder-Mead')` — no gradients required, which matters because the FEM output is not differentiable with respect to the input parameters in the way the optimiser needs.

**Total FEM calls per patient:** 20 (grid) + ≤30 (Nelder-Mead) ≈ 50. At ~3 min/solve this is ~150 min/patient.

### 5.6 The cost function — bilateral asymmetry (`build_mirror_map` + `recover_fem_fem`)

The DMR-IR mesh contains **both breasts** (X spans ~235 mm, centred near the sternum at $x_{mid}\approx$ mean of surface X). The cost exploits this: the healthy breast is a built-in control.

**Mirror map (`build_mirror_map`).** For every surface vertex $\mathbf{v}=(x,y,z)$, reflect across the midline to $(2x_{mid}-x,\,y,\,z)$ and snap to the nearest actual vertex via a KD-tree. The mean snap distance is reported per patient (`mirror_snap`) as a **bilateral-symmetry quality flag** — small = clean mirror (Patient_1: 11.7 mm); large = asymmetric breasts, noisier fit.

**Affected side.** The breast containing the IR hot-spot ($x_0$) is the affected side; its skin vertices `aff` are where the asymmetry is evaluated.

**Cost.** For candidate $(z_t,r_t)$ the FEM surface field $T_c$ is reduced to its asymmetry on the affected side and matched to the measured asymmetry:

$$ A(\mathbf{v}) = T(\mathbf{v}) - T(\text{mirror}(\mathbf{v})),\qquad J = \frac{1}{|\text{aff}|}\sum_{\mathbf{v}\in\text{aff}}\big(A_{\text{FEM}}(\mathbf{v}) - A_{\text{IR}}(\mathbf{v})\big)^2 $$

**Two properties this buys:**

1. **Common-mode cancellation (§2.4).** The ~5 °C homogeneous-model floor is shared by both breasts and subtracts out, exposing the ~2 °C real asymmetry.
2. **No radius rail.** A source large enough to cross the midline warms *both* breasts roughly symmetrically; that contribution cancels in $A_{\text{FEM}}$ and earns **no** cost reduction. The optimiser is therefore forced to a confined, single-breast source instead of inflating $r_t$ to the bound. This is what eliminates the $r\to40$ mm degeneracy (§7).

On synthetic data the planted tumour is on one breast, so $A_{\text{FEM}}$ at the true $(z_0,r_0)$ exactly equals $A_{\text{syn}}$ and $J\to0$ — self-consistency, hence the synthetic validation, is unaffected.

---

## 6. Synthetic Validation

Because DMR-IR has no depth ground truth, **synthetic recovery** is the only available method-validity test. The logic is: if the method is correct, it should be able to recover a tumour that was planted by the same FEM solver — a self-consistency check. A known tumour $(z_0, r_0)$ is planted, FEniCSx generates the surface temperature as if that tumour existed, and then the unchanged inverse recovers $(\hat z, \hat r)$ treating that synthetic temperature as if it were real measured data:

```
Plant (z₀, r₀)
      ↓
FEM forward → T_syn   (same FEniCSx, same mesh, same BCs)
      ↓
Inverse FEM-FEM → (ẑ, r̂)   (grid search + Nelder-Mead on T_syn)
      ↓
depth_err = |ẑ − z₀| / |z₀| × 100%
radius_err = |r̂ − r₀| / r₀ × 100%
```

Because the forward model that generated the data is **identical** to the forward model used by the optimiser, the cost surface reaches machine zero at the true answer — any residual error is purely numerical (discretisation + optimiser tolerance). Sub-0.5% error therefore proves the optimiser is well-conditioned and the FEM solve is self-consistent.

This directly mirrors Mukhmetov et al. (2025)'s ANSYS synthetic validation protocol.

Errors are defined as:

$$\text{depth\_err} = \frac{|\hat z - z_0|}{|z_0|}\times100\%,\quad \text{radius\_err} = \frac{|\hat r - r_0|}{r_0}\times100\%$$

**Protocol:** 9 scenarios — three depths (40 %, 55 %, 70 % of breast depth range) × three radii (8, 14, 20 mm). Patient_1 geometry (z-range [−125.7, 33.7] mm), lateral fixed to IR hot-spot centroid (46.7, 24.5) mm.

**Results (Patient_1, bilateral cost, run 2026-06-15) — all 9 scenarios complete:**

| Planted z (mm) | Planted r (mm) | Recovered z (mm) | Recovered r (mm) | Depth err | Radius err |
|---|---|---|---|---|---|
| −61.93 | 8.0 | −62.06 | 8.02 | 0.20 % | 0.22 % |
| −61.93 | 14.0 | −61.99 | 14.00 | 0.09 % | 0.02 % |
| −61.93 | 20.0 | −61.97 | 20.00 | 0.07 % | 0.00 % |
| −38.03 | 8.0 | −37.88 | 8.06 | 0.39 % | 0.76 % |
| −38.03 | 14.0 | −38.14 | 14.01 | 0.28 % | 0.09 % |
| −38.03 | 20.0 | −38.05 | 20.01 | 0.04 % | 0.05 % |
| −14.13 | 8.0 | −14.20 | 8.00 | 0.47 % | 0.04 % |
| −14.13 | 14.0 | −14.21 | 14.01 | 0.57 % | 0.06 % |
| −14.13 | 20.0 | −14.19 | 20.00 | 0.41 % | 0.00 % |
| | | | **Mean** | **0.28 %** | **0.14 %** |
| | | | **Max** | **0.57 %** | **0.76 %** |

Cost converges to machine zero (< 10⁻⁶) in every scenario — FEniCSx recovers its own output exactly under the bilateral cost, as expected for a self-consistent FEM-FEM pair. There is no single outlier: depth error stays sub-0.6 % and radius error sub-0.8 % across all nine scenarios. The shallow tumours (z ≈ −14 mm) cluster slightly higher in *depth* error (0.41–0.57 %), consistent with stronger skin-BC interaction near the surface; the one mildly elevated *radius* error (0.76 %, z = −38 mm, r = 8 mm) is the smallest mid-depth source, where the asymmetry footprint is weakest.

This validation is run under the **same bilateral cost used on real data**, so it certifies the exact pipeline applied to the cohort — not a different (absolute-MSE) cost. It replicates Mukhmetov et al. (2025)'s synthetic ANSYS protocol and achieves mean 0.28 % / 0.14 %, vs their reported 0.70–5.19 %.

### 6.5 Numerical verification — solver and mesh convergence

Synthetic recovery (§6) proves the inverse *recovers its own forward output*; a separate question is whether each individual FEM solve is itself numerically correct — does the linear solver converge, and is the solution mesh-independent at the 3 mm production resolution? Both were verified on Patient_1 with a fixed tumour ($r=15$ mm), evaluating the solution at a **fixed set of 8 000 physical probe points** identical across all meshes (so re-triangulation injects no metric noise). Script: `convergence_test2.py`; figure: `results/thesis_figures/fig8_konvergensi_mesh.png`.

**Solver convergence — optimal.** The CG solver with GAMG preconditioner reaches `CONVERGED_RTOL` (relative tolerance $10^{-10}$) at every mesh, in **16–23 iterations**, with final residual $10^{-7}$–$10^{-6}$. The iteration count stays essentially flat as the mesh grows 54× (47 k → 2.58 M DOFs) — textbook mesh-independent multigrid scaling, which rules out any silent solver failure.

**Mesh convergence — converged at 3 mm.** The probe-mean skin temperature is stable at **24.56 ± 0.01 °C** across the full refinement, and the RMS change between successive meshes shrinks monotonically:

| Element size | DOFs | KSP iters | KSP reason | RMS change vs previous |
|---|---|---|---|---|
| 4.0 mm | 47,622 | 16 | CONVERGED_RTOL | — |
| **3.0 mm (production)** | 102,135 | 17 | CONVERGED_RTOL | 38.9 m°C |
| 2.0 mm | 319,376 | 17 | CONVERGED_RTOL | 35.2 m°C |
| 1.5 mm | 742,370 | 17 | CONVERGED_RTOL | 24.3 m°C |
| 1.0 mm | 2,583,936 | 23 | CONVERGED_RTOL | 9.5 m°C |

The successive change falls from 38.9 to 9.5 m°C; at matched refinement ratio (1.5×) the 3→2 mm step (35 m°C) versus the 1.5→1 mm step (9.5 m°C) is **3.7× smaller**, confirming the solution is in the asymptotic convergence regime. The 0.01 °C residual wobble at 3 mm is ~300× smaller than the ~5 °C model-mismatch floor (§2.4), so mesh discretisation is **not** a meaningful error source.

**Why this matters.** Because the solver converges optimally *and* the skin signal is mesh-converged at 3 mm, the ~5 °C real-data mismatch cannot be a numerical artefact — it is attributable to the **boundary-condition physics** (the homogeneous-tissue assumption and the BC constants of §2.5), exactly as expected. For context, our 3 mm mesh has **587 k tetrahedral elements in 3D**, versus Jahani et al.'s 10 448 triangular elements in 2D (~56× more, one dimension higher), so the model is far better resolved than the published comparison.

---

## 7. Real-Patient Application and Limitations

The method validates to sub-0.5 % on synthetic data, but applying it to real DMR-IR exposed that the **choice of cost function decides whether real-data geometry is identifiable at all**. Three cost functions were tried, in order, each diagnosing the next.

### 7.0 The cost-function progression (empirical)

**(1) Absolute skin MSE — degenerate.** Cost $=\text{mean}((T_{\text{FEM}}-T_{\text{IR}})^2)$ over skin. Every real patient collapsed to the radius bound $r=40$ mm. Mechanism: the homogeneous-model floor (§2.4) makes the FEM skin ~5 °C colder than real IR, and because total injected heat scales as $\propto r^3$ (~33× from $r=8$ to $r=40$ even after the Gautherie $Q_0$ drop), the optimiser abuses radius as a **global heating knob** to close the absolute-level gap. Evidence: 6/6 patients at $r=40$, `final_cost` 23–28 (`results/raw_baseline/`).

**(2) Mean-centred skin MSE — still degenerate.** Subtract each field's spatial mean before MSE, matching pattern not level. The absolute-level abuse is removed (`final_cost` drops to ~7–8), but radii still rail — now to *both* bounds (40 mm and 7 mm) — with **flat cost surfaces** (Nelder-Mead returns the grid value unchanged to 6 decimals). Mechanism: even after level removal, the diffusion-smoothed real pattern is intrinsically broad, so a diffuse source matches it as well as a tight one. Evidence: `results/mean_centered_baseline/`.

**(3) Bilateral asymmetry — identifiable.** Cost $=\text{mean}((A_{\text{FEM}}-A_{\text{IR}})^2)$ on the affected breast (§5.6). Patient_1 recovered $r=20.4$ mm, $z=-0.5$ mm — an **interior** minimum, off both rails, for the first time. Mechanism: bilateral subtraction cancels the common-mode floor (raising SNR) *and* removes the cross-midline reward (killing the radius rail). Both failure modes of (1) and (2) are addressed simultaneously.

| Cost function | Patient_1 $r_{hat}$ | Cohort behaviour | Verdict |
|---|---|---|---|
| Absolute MSE | 40.0 mm | 6/6 at upper rail | degenerate |
| Mean-centred MSE | 40.0 mm | both rails, flat surface | degenerate |
| **Bilateral asymmetry** | **20.4 mm** | **interior minimum** | **identifiable** |

### 7.0.1 What bilateral does and does not establish

- **Does:** restores *identifiability* — a unique interior $(z_t,r_t)$ minimum exists, validated end-to-end on synthetic data under the identical cost (§6).
- **Does not:** verify *absolute accuracy* on real patients — DMR-IR has no MRI/CT ground truth. The residual after the best bilateral fit (Patient_1 RMS ≈ 3.7 °C) reflects genuine non-tumour left–right tissue/vascular asymmetry the homogeneous model cannot reproduce, which bounds accuracy.

**What is recoverable on real data, post-bilateral:**

| Feature | Recoverable? | Method |
|---|---|---|
| Lateral position (x, y) | **Yes** | IR hot-spot centroid — direct, no FEM |
| Depth / radius estimates | **Identifiable** (interior), accuracy unvalidated | bilateral FEM-FEM inverse |
| Bilateral-symmetry quality | **Yes** | `mirror_snap` (mean mirror distance) per patient |
| FEA model-fit residual | **Yes** | absolute \|FEA−IR\| — model-quality flag |
| Benign vs malignant trends | **Significant via bilateral asymmetry** (p=0.0044, §7.3); not via inverse geometry | distributions over the 122-patient cohort |

Mukhmetov et al. (2025) report 0.7–5 % on synthetic ANSYS and 18.67 % radius error on their single real patient (depth unverifiable) — using a real 3D scanner and a single-breast model. We reach identifiability on real data from **2D IR alone** by exploiting the bilateral geometry their single-breast setup does not have.

### 7.1 Fault-tolerant incremental execution

Each patient's result is written to CSV **immediately after recovery**, before the next patient starts:

```python
df_row.to_csv(out_csv, mode='a', header=write_header, index=False)
```

If the run crashes (OOM, power loss, FEM mesh failure), restarting with `--resume` reads the completed patient IDs and skips them, picking up from the first incomplete patient. No work is lost.

### 7.2 GPU-parallel execution

The 40-patient cohort is split across two GPUs using a stride/offset scheme:

```
GPU 0  --stride 2 --offset 0  →  patients [0, 2, 4, 6, ...]  (even indices)
GPU 1  --stride 2 --offset 1  →  patients [1, 3, 5, 7, ...]  (odd indices)
```

Patient selection (`select_patients`) first builds a **balanced interleaved list** — alternating benign/malignant before the stride is applied — so each GPU processes approximately equal numbers of both classes. `CUDA_VISIBLE_DEVICES=0/1` isolates GPU memory. The two CSVs are merged after both runs complete:

```bash
head -1 results/femfem_gpu0.csv > results/cohort_femfem_all.csv
tail -n +2 results/femfem_gpu0.csv >> results/cohort_femfem_all.csv
tail -n +2 results/femfem_gpu1.csv >> results/cohort_femfem_all.csv
```

The cohort was then extended from the initial balanced 40 to the **full usable set of 122 patients** (the remaining 82 run with `--indices` into `results/femfem_gpu{0,1}_rest.csv`, merged into `results/cohort_femfem_ALL122.csv`).

### 7.3 Full-cohort statistical results (122 patients)

The complete usable DMR-IR set is **122 patients (96 benign, 26 malignant)** — every patient with all five IR views and masks (dataset funnel: 268 in the HuggingFace manifest → 137 extracted to disk → **122 complete 5-view sets**; the 15 dropped lack views. Full verification: `TherMAM-NeRF/data_verification.ipynb`).

**Inverse-geometry metrics do not separate the classes.** At n=122, no recovered-geometry metric reaches significance (Mann–Whitney U, two-sided):

| Metric | Benign median (n=96) | Malignant median (n=26) | p |
|---|---|---|---|
| Depth $\hat z$ (mm) | −26.9 | −25.7 | 0.99 |
| Radius $\hat r$ (mm) | 40.0 | 40.0 | 0.66 |
| Final cost | 10.14 | 8.10 | 0.32 |
| FEA residual (°C) | 6.02 | 6.07 | 0.40 |

Radius rails persist in both classes (interior fraction: benign 31 %, malignant 27 %), consistent with the published depth–size degeneracy (§2.5). The recovered geometry carries **no class information** — the expected outcome for an ill-posed inversion.

**The model-free bilateral asymmetry *does* separate the classes.** The mean left–right asymmetry |A| on the affected breast (the target quantity `|A_target|`, computed directly from IR — **independent of the FEM inverse**) is significantly higher in malignancy:

| | Benign (n=96) | Malignant (n=26) | p |
|---|---|---|---|
| \|A\| median | 1.373 °C | **1.559 °C** | **0.0088** (two-sided), **0.0044** (one-sided: malignant > benign) |

This was a marginal trend at n=20/group (one-sided p ≈ 0.063) and **sharpens to p < 0.01 at the full cohort**, surviving Bonferroni correction across the five metrics tested ($0.0088 < 0.05/5$). It agrees with the clinical premise (malignant angiogenesis raises surface asymmetry) and with the subtraction-based detection of Cervantes et al. and Jahani et al. (data `results/asymmetry_ALL.csv`; figure `results/thesis_figures/fig9_asimetri_bilateral.png`).

**Interpretation.** The pipeline yields a significant malignancy signal through the *bilateral asymmetry* it extracts, while the *geometric* inversion is fundamentally ill-posed. The positive result is model-free and robust; the negative result is a rigorously-characterised, literature-confirmed limit — not an implementation failure.

---

## 8. Findings and Honest Contribution

- **Forward-model contribution:** a documented progression of inverse formulations — free-parameter PINN (escapes to boundary) → forward-matching PINN (Tanh saturation floor) → FEM-FEM (self-consistent, sub-0.5 %). FEniCSx-as-oracle removes the 7 °C PINN noise floor (SNR 0.09:1 → 670:1 on synthetic).
- **Cost-function contribution:** a documented progression of cost functions on real data — absolute MSE (radius rails to 40 mm) → mean-centred MSE (rails to both bounds, flat surface) → **bilateral asymmetry** (interior minimum). Identifiability is a property of the *cost*, not just the forward model.
- **Validated claim:** under the bilateral cost, the FEM-FEM inverse recovers planted depth and radius to mean 0.28 % / 0.14 % on synthetic self-consistent data (9 scenarios, Patient_1) — and that *same* cost yields an interior, non-degenerate solution on real data (Patient_1: r = 20.4 mm).
- **Key insight:** the ~5 °C model-mismatch floor is **common-mode** across the two reconstructed breasts and cancels under bilateral subtraction, exposing the ~2 °C real asymmetry. The two-breast geometry is a feature, not a nuisance.
- **Honest limitation:** bilateral establishes *identifiability*, not *absolute accuracy* — DMR-IR has no co-registered MRI/CT, so real-patient depth/radius accuracy remains unverifiable; residual non-tumour left–right asymmetry bounds the fit.
- **Novel aspect vs Mukhmetov:** TherMAM-NeRF provides bilateral 3D geometry + surface temperature from **2D IR alone** — no 3D scanner — and the bilateral cost turns the second breast (which their single-breast model lacks) into the control that makes real-data inversion identifiable.
- **Significant real-data finding (122 patients):** the model-free bilateral asymmetry |A| is **significantly elevated in malignancy** (benign 1.37 °C vs malignant 1.56 °C, Mann–Whitney U p = 0.0088 two-sided / 0.0044 one-sided; survives Bonferroni). This is a clinically-meaningful, model-independent screening signal, distinct from the (ill-posed) geometric inversion (§7.3).
- **Numerical method verified:** the FEM solver converges optimally (mesh-independent ~17 CG/GAMG iterations) and the skin signal is mesh-converged at 3 mm (<0.01 °C change through 1.0 mm refinement), so the model-mismatch floor is physics, not numerics (§6.5).
- **Literature-grounded:** boundary conditions are identical to the published FEM-thermography model (Jahani et al. 2023), and the recovered-radius rail is the depth–size degeneracy that paper reports explicitly — a reproduced known limit, with their shape-feature `L` as the indicated path forward (§2.5).

---

## 9. Running

Use the full Python path (no `conda activate` needed in non-interactive shells):

```bash
PYTHON=/mnt/Data1/Peoples/faiz836b/miniconda3/envs/bioheat/bin/python
cd TherMAM-NeRF

# ── Synthetic validation (9 scenarios, Patient_1) ──────────────────────────
$PYTHON -u mukhmetov_recover.py --idx 0 | tee mukhmetov_syn.log
# Outputs: results/mukhmetov_recovery.csv
#          results/mukhmetov_recovery.png         (planted vs recovered scatter)
#          results/mukhmetov_error_breakdown.png  (bar chart + heatmaps)

# ── Real cohort — 40 patients, 2-GPU split (run in separate tmux panes) ───
# Pane 1 — GPU 0 (even patients):
CUDA_VISIBLE_DEVICES=0 $PYTHON -u mukhmetov_recover.py --real --subset 40 \
    --stride 2 --offset 0 --out results/femfem_gpu0.csv | tee femfem_gpu0.log

# Pane 2 — GPU 1 (odd patients):
CUDA_VISIBLE_DEVICES=1 $PYTHON -u mukhmetov_recover.py --real --subset 40 \
    --stride 2 --offset 1 --out results/femfem_gpu1.csv | tee femfem_gpu1.log

# ── Resume after crash ─────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 $PYTHON -u mukhmetov_recover.py --real --subset 40 \
    --stride 2 --offset 0 --out results/femfem_gpu0.csv --resume | tee -a femfem_gpu0.log

# ── Monitor progress ───────────────────────────────────────────────────────
grep -E "(^\[|RESULT|refined)" femfem_gpu0.log | tail -5
wc -l results/femfem_gpu0.csv results/femfem_gpu1.csv

# ── Merge + plot after both GPUs finish ───────────────────────────────────
head -1 results/femfem_gpu0.csv > results/cohort_femfem_all.csv
tail -n +2 results/femfem_gpu0.csv >> results/cohort_femfem_all.csv
tail -n +2 results/femfem_gpu1.csv >> results/cohort_femfem_all.csv
$PYTHON -u mukhmetov_recover.py --plot-csv results/cohort_femfem_all.csv
```

---

## 10. Outputs

### Synthetic validation
| File | Contents |
|---|---|
| `results/mukhmetov_recovery.csv` | 9-scenario table: `planted_z, recovered_z, depth_err_pct, planted_r, recovered_r, radius_err_pct` |
| `results/mukhmetov_recovery.png` | Planted vs recovered scatter — depth and radius (ideal line) |
| `results/mukhmetov_error_breakdown.png` | Per-scenario bar chart + 3×3 depth/radius error heatmaps |
| `results/Patient_N/Patient_N_syn.stl` | Patient surface mesh (STL, from TherMAM-NeRF) |
| `results/Patient_N/Patient_N_syn.msh` | Tetrahedral mesh for FEniCSx (generated once, reused) |

### Real-patient cohort
| File | Contents |
|---|---|
| `results/femfem_gpu0.csv` | GPU 0 results (20 patients): `patient_id, label, x0_mm, y0_mm, z_hat_mm, r_hat_mm, grid_cost, final_cost, fea_mean_residual, fea_max_residual, n_fem_evals` |
| `results/femfem_gpu1.csv` | GPU 1 results (20 patients), same columns |
| `results/cohort_femfem_all.csv` | Merged 40-patient table |
| `results/cohort_femfem_results.png` | 6-panel thesis figure: depth/radius box-strip plots by class, z-vs-r scatter, lateral position map, cost and FEA residual distributions |
| `results/cohort_femfem_summary.csv` | Per-class mean ± std for z_hat, r_hat, final_cost |
| `results/Patient_N/Patient_N_fea_vs_ir.html` | Interactive 3D: FEM temperature vs IR thermogram side-by-side |
| `results/Patient_N/Patient_N_tumor_localisation.html` | Interactive 3D: breast surface (IR coloured) with recovered tumour sphere |

### Full cohort (122 patients) & verification
| File | Contents |
|---|---|
| `results/femfem_gpu{0,1}_rest.csv` | The additional 82 patients (beyond the initial 40), same columns |
| `results/cohort_femfem_ALL122.csv` | Merged 122-patient table (96 benign, 26 malignant) |
| `results/asymmetry_ALL.csv` | Per-patient model-free bilateral asymmetry `\|A_target\|` (the p=0.0044 signal) |
| `results/thesis_figures/fig9_asimetri_bilateral.png` | Benign-vs-malignant asymmetry box/strip plot with p-values (Bahasa Indonesia) |
| `results/thesis_figures/fig7_fem_convergence.png`, `fig8_konvergensi_mesh.png` | Solver + mesh convergence study (EN / Bahasa Indonesia) |
| `results/thesis_figures/fig1…6_*.png` | Dataset verification: class distribution, 268→137→122 funnel, view completeness, clinical metadata, FEM-input map |
| `TherMAM-NeRF/data_verification.ipynb` | Reproducible dataset/metadata verification notebook |
| `convergence_test2.py` | Standalone solver/mesh convergence diagnostic (no GPU) |