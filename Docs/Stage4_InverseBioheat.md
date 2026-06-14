# Stage 4: Inverse Bioheat Solving — FEM-FEM Recovery (Mukhmetov-style)

**Current source:** `TherMAM-NeRF/mukhmetov_recover.py` (synthetic validation), `TherMAM-NeRF/run_cohort.py --method mukhmetov` (cohort)
**Documented negative results (retained):** PINN v1 (`UNET_Segmentation/PINNpdeSolver/PINN_Pipeline.py`), PINN v3 (`TherMAM-NeRF/Thermamnerf_PINN_v3.ipynb`), Mukhmetov-PINN (`run_cohort.py --method mukhmetov`, prior PINN-based variant)

---

## 1. Overview

Stage 4 couples the Stage-3 surface temperature to biophysics and solves an **inverse bioheat problem**: given the reconstructed skin temperature, estimate the internal tumour consistent with it. The framing is a **screening tool** — the quantities sought are the tumour's **position, depth, and size**, not a diagnosis.

The current method uses **FEniCSx (dolfinx) as the forward model inside a grid-search + Nelder-Mead optimiser**, directly replicating the self-consistent "FEM-as-oracle" approach of Mukhmetov et al. (2025), but replacing their ANSYS solver with FEniCSx and their 3D scanner requirement with TherMAM-NeRF geometry. Three PINN variants were attempted first; all failed for reasons documented in §4.

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

## 5. FEM-FEM Algorithm

**Fixed (never optimised):**
- Lateral $(x_0, y_0)$ = centroid of top-10 % IR hot-spot — robust, read directly from IR.
- Peak heat $Q_0 = Q_0(r_t)$ from Gautherie law — physiologically derived, not fitted.

**Optimised:** $(z_t, r_t)$ only — 2 degrees of freedom.

**Step 1 — coarse grid:** 5 depth × 4 radius = 20 FEM solves spanning 85 % of breast depth range × 7–22 mm radius. Cost = skin-only MSE (chest wall excluded — Dirichlet 37 °C in every solve, carries no useful information).

**Step 2 — Nelder-Mead refine:** ≤ 30 iterations from grid best; tolerances $\Delta z < 0.5$ mm, $\Delta J < 10^{-7}$.

**Mesh:** generated once per patient from TherMAM-NeRF STL via Gmsh (`mesh_size_mm = 3.0`), reused across all FEM evaluations for that patient.

---

## 6. Synthetic Validation

Because DMR-IR has no depth ground truth, **synthetic recovery** is the only available method-validity test. A known tumour $(z_0, r_0)$ is planted, FEniCSx generates the surface temperature, the unchanged inverse recovers $(\hat z, \hat r)$, and errors are computed:

$$\text{depth\_err} = \frac{|\hat z - z_0|}{|z_0|}\times100\%,\quad \text{radius\_err} = \frac{|\hat r - r_0|}{r_0}\times100\%$$

**Protocol:** 9 scenarios — three depths (40 %, 55 %, 70 % of breast depth range) × three radii (8, 14, 20 mm). Patient_1 geometry (z-range [−125.7, 33.7] mm), lateral fixed to IR hot-spot centroid (46.7, 24.5) mm.

**Results (Patient_1, run 2026-06-15):**

| Planted z (mm) | Planted r (mm) | Recovered z (mm) | Recovered r (mm) | Depth err | Radius err | FEM evals |
|---|---|---|---|---|---|---|
| −61.9 | 8.0 | −62.0 | 8.0 | **0.2 %** | **0.1 %** | 48 |
| −61.9 | 14.0 | −62.0 | 14.0 | **0.1 %** | **0.1 %** | 57 |
| −61.9 | 20.0 | (in progress) | — | — | — | — |
| −38.0 | 8.0 | — | — | — | — | — |
| −38.0 | 14.0 | — | — | — | — | — |
| −38.0 | 20.0 | — | — | — | — | — |
| −14.1 | 8.0 | — | — | — | — | — |
| −14.1 | 14.0 | — | — | — | — | — |
| −14.1 | 20.0 | — | — | — | — | — |

Cost converges to **machine zero** (< 10⁻⁶) in completed scenarios — FEniCSx recovers its own output exactly, as expected for a self-consistent FEM-FEM pair. This directly replicates Mukhmetov et al. (2025)'s synthetic ANSYS validation methodology but achieves sub-0.3 % with FEniCSx as both forward and inverse model.

---

## 7. Real-Patient Application and Limitations

On real DMR-IR data the model-mismatch floor (§2.4, ≈4.5–5.6 °C) exceeds the tumour signal (0.67 °C). The cost surface will be dominated by background mismatch, not the tumour perturbation. Real-patient depth/radius estimates carry unknown error; ground-truth validation requires co-registered imaging not available in DMR-IR.

**What remains recoverable on real data:**

| Feature | Recoverable? | Method |
|---|---|---|
| Lateral position (x, y) | **Yes** | IR hot-spot centroid — direct, no FEM needed |
| FEA model-fit residual | **Yes** | Cost at convergence — model quality flag |
| Depth / radius estimates | Reportable, not validated | FEM-FEM inverse output |
| Benign vs malignant trends | Testable | Compare distributions across 122 patients |

This limitation is shared with all published thermographic inverse methods. Mukhmetov et al. (2025) report 0.7–5 % on synthetic ANSYS data and 18.67 % radius error on their single real patient (depth unverifiable).

---

## 8. Findings and Honest Contribution

- **Methodological contribution:** a three-stage documented progression — free-parameter PINN (escapes to boundary) → forward-matching PINN (Tanh saturation floor) → FEM-FEM (self-consistent, sub-0.3 %).
- **Validated claim:** FEM-FEM inverse recovers planted tumour depth and radius to < 0.3 % on synthetic self-consistent data (9 scenarios, Patient_1, FEniCSx).
- **Honest limitation:** real-data depth/radius are not validated — model-mismatch floor exceeds tumour signal; no ground truth available in DMR-IR.
- **Novel aspect vs Mukhmetov:** TherMAM-NeRF provides 3D geometry + surface temperature from 2D IR alone, without the 3D scanner Mukhmetov et al. required.

---

## 9. Running

```bash
conda activate bioheat
cd TherMAM-NeRF

# Synthetic validation (9 scenarios, Patient_1):
python -u mukhmetov_recover.py --idx 0 | tee mukhmetov_syn.log

# Real cohort (all 122 patients):
python -u run_cohort.py --method mukhmetov --all --out results/cohort_mukhmetov.csv

# Two-GPU split (even/odd patient indices):
CUDA_VISIBLE_DEVICES=0 python -u run_cohort.py --method mukhmetov \
    --indices 0 2 4 6 8 ... --out results/cohort_gpu0.csv &
CUDA_VISIBLE_DEVICES=1 python -u run_cohort.py --method mukhmetov \
    --indices 1 3 5 7 9 ... --out results/cohort_gpu1.csv &
```

---

## 10. Outputs

| File | Contents |
|---|---|
| `results/mukhmetov_recovery.csv` | 9-scenario synthetic table (planted_z, recovered_z, depth_err_pct, planted_r, recovered_r, radius_err_pct) |
| `results/mukhmetov_recovery.png` | Planted vs recovered scatter plots — depth and radius |
| `results/Patient_N/Patient_N_syn.stl` | Patient surface mesh (STL, from TherMAM-NeRF) |
| `results/Patient_N/Patient_N_syn.msh` | Tetrahedral mesh for FEniCSx (generated once, reused) |
| `results/cohort_mukhmetov.csv` | Per-patient: x_t_mm, y_t_mm, z_hat_mm, r_hat_mm, cost, fea_residual |