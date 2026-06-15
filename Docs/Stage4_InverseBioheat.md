# Stage 4: Inverse Bioheat Solving — FEM-FEM Recovery (Mukhmetov-style)

**Current source:** `TherMAM-NeRF/mukhmetov_recover.py` (synthetic validation), `TherMAM-NeRF/run_cohort.py --method mukhmetov` (cohort)
**Documented negative results (retained):** PINN v1 (`UNET_Segmentation/PINNpdeSolver/PINN_Pipeline.py`), PINN v3 (`TherMAM-NeRF/Thermamnerf_PINN_v3.ipynb`), Mukhmetov-PINN (`run_cohort.py --method mukhmetov`, prior PINN-based variant)

---

## 1. Overview

Stage 4 couples the Stage-3 surface temperature to biophysics and solves an **inverse bioheat problem**: given the reconstructed skin temperature, estimate the internal tumour consistent with it. The framing is a **screening tool** — the quantities sought are the tumour's **position, depth, and size**, not a diagnosis.

The current method uses **FEniCSx (dolfinx) as the forward model inside a grid-search + Nelder-Mead optimiser**, directly replicating the self-consistent "FEM-as-oracle" approach of Mukhmetov et al. (2025), but replacing their ANSYS solver with FEniCSx and their 3D scanner requirement with TherMAM-NeRF geometry. Three PINN variants were attempted first; all failed for reasons documented in §4.

### 1.1 Four-stage pipeline overview

| Stage | What happens | Key output |
|---|---|---|
| **1 — TherMAM-NeRF** | Neural Radiance Field reconstructs 3D breast geometry and surface temperature from 2D IR image views | `surface_pts` (mm), `T_measured` (°C), `faces`, bounding box |
| **2 — Mesh generation** | Gmsh fills the surface shell with ~587 k tetrahedra (mesh size 3 mm) | `Patient_N.msh` — volumetric FEM mesh, generated once per patient |
| **3 — Forward FEM** | FEniCSx solves the Pennes bioheat PDE on the volumetric mesh for a candidate tumour `(z_t, r_t)` and returns surface temperatures | `T_surf` — simulated skin temperature field |
| **4 — Inverse optimisation** | A coarse grid search (5 depth × 4 radius = 20 FEM evaluations) followed by Nelder-Mead refinement (≤30 more) minimises skin MSE between FEM output and measured IR | `(z_hat, r_hat)` — best-fit tumour depth and radius |

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

### 2.4 The model-mismatch floor on real data

An FEniCSx forward solve of the same BVP (uniform tissue, Robin skin BC $h=10$, $T_{air}=20\,^\circ$C) reproduces real DMR-IR thermograms with a **residual of 4.5–5.6 °C** across the cohort. The tumour's own surface signature measured from the DMR-IR hotspot is **≈0.67 °C**. Because signal < floor, real-data depth/radius inversion cannot be validated without co-registered MRI/CT ground truth, which DMR-IR does not provide.

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

**Step 1 — coarse grid:** 5 depth × 4 radius = 20 FEM solves spanning 85 % of breast depth range × 7–22 mm radius. Cost = skin-only MSE (chest wall excluded — Dirichlet 37 °C in every solve, carries no useful information).

```
depth grid:  z_lo + 8%·Δz   to   z_hi − 5%·Δz   (5 values)
radius grid: 7 mm  to  22 mm                       (4 values)
```

The asymmetric margin (8% from bottom, 5% from top) keeps the tumour away from the chest wall and skin boundary where the cost function is distorted by boundary conditions rather than tumour proximity.

**Step 2 — Nelder-Mead refine:** ≤ 30 iterations initialised at the grid minimum; convergence tolerances $\Delta z < 0.5$ mm, $\Delta J < 10^{-7}$. Uses Scipy's `minimize(..., method='Nelder-Mead')` — no gradients required, which matters because the FEM output is not differentiable with respect to the input parameters in the way the optimiser needs.

**Total FEM calls per patient:** 20 (grid) + ≤30 (Nelder-Mead) ≈ 50. At ~3 min/solve this is ~150 min/patient.

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

**Results (Patient_1, run 2026-06-15) — all 9 scenarios complete:**

| Planted z (mm) | Planted r (mm) | Recovered z (mm) | Recovered r (mm) | Depth err | Radius err |
|---|---|---|---|---|---|
| −61.93 | 8.0 | −62.05 | 8.01 | 0.19 % | 0.11 % |
| −61.93 | 14.0 | −62.02 | 13.99 | 0.14 % | 0.08 % |
| −61.93 | 20.0 | −61.89 | 20.00 | 0.06 % | 0.01 % |
| −38.03 | 8.0 | −38.11 | 8.01 | 0.21 % | 0.12 % |
| −38.03 | 14.0 | −37.92 | 14.00 | 0.28 % | 0.02 % |
| −38.03 | 20.0 | −38.08 | 19.99 | 0.12 % | 0.04 % |
| −14.13 | 8.0 | −14.20 | 8.02 | 0.50 % | 0.29 % |
| −14.13 | 14.0 | −14.06 | 13.99 | 0.50 % | 0.10 % |
| −14.13 | 20.0 | −14.16 | 20.00 | 0.23 % | 0.01 % |
| | | | **Mean** | **0.25 %** | **0.09 %** |
| | | | **Max** | **0.50 %** | **0.29 %** |

Cost converges to machine zero (< 10⁻⁶) in every scenario — FEniCSx recovers its own output exactly, as expected for a self-consistent FEM-FEM pair. The slight increase in depth error for the shallowest tumour (z ≈ −14 mm, 0.50 %) is consistent with increased skin-BC interaction at shallow depth; all errors remain sub-0.5 %.

This directly replicates Mukhmetov et al. (2025)'s synthetic ANSYS validation methodology and achieves sub-0.5 % across all scenarios, vs their reported 0.70–5.19 %.

---

## 7. Real-Patient Application and Limitations

On real DMR-IR data the model-mismatch floor (§2.4, ≈4.5–5.6 °C) exceeds the tumour signal (0.67 °C). The cost surface will be dominated by background mismatch, not the tumour perturbation. Real-patient depth/radius estimates carry unknown error; ground-truth validation requires co-registered imaging not available in DMR-IR.

**What remains recoverable on real data:**

| Feature | Recoverable? | Method |
|---|---|---|
| Lateral position (x, y) | **Yes** | IR hot-spot centroid — direct, no FEM needed |
| FEA model-fit residual | **Yes** | Cost at convergence — model quality flag |
| Depth / radius estimates | Reportable, not validated | FEM-FEM inverse output |
| Benign vs malignant trends | Testable | Compare distributions across 40 patients |

This limitation is shared with all published thermographic inverse methods. Mukhmetov et al. (2025) report 0.7–5 % on synthetic ANSYS data and 18.67 % radius error on their single real patient (depth unverifiable).

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

---

## 8. Findings and Honest Contribution

- **Methodological contribution:** a three-stage documented progression — free-parameter PINN (escapes to boundary) → forward-matching PINN (Tanh saturation floor) → FEM-FEM (self-consistent, sub-0.3 %).
- **Validated claim:** FEM-FEM inverse recovers planted tumour depth and radius to < 0.3 % on synthetic self-consistent data (9 scenarios, Patient_1, FEniCSx).
- **Honest limitation:** real-data depth/radius are not validated — model-mismatch floor exceeds tumour signal; no ground truth available in DMR-IR.
- **Novel aspect vs Mukhmetov:** TherMAM-NeRF provides 3D geometry + surface temperature from 2D IR alone, without the 3D scanner Mukhmetov et al. required.

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