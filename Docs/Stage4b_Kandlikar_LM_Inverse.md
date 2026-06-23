# Stage 4b: Kandlikar Levenberg–Marquardt Inverse Bioheat

**Source notebook:** `TherMAM-NeRF/FEM_LM_Inverse_v1.ipynb`  
**Method basis:** Gonzalez-Hernandez, Recinella, Kandlikar *et al.*, *Infrared Physics & Technology* **105** (2020) 103202  
**Status (2026-06-21):** LM inverse implemented and LM sign bug fixed; synthetic validation in progress (SYNTHETIC_TEST = True)

---

## 0. Relation to Stage 4 (Mukhmetov-style)

Stage 4 (`Stage4_InverseBioheat.md`) used a **bilateral asymmetry cost** + **Nelder-Mead grid-search** with a 2-parameter inverse $(z_t, r_t)$, pinning lateral position from the IR hotspot and excluding tumour diameter from the optimisation. That method found a significant bilateral asymmetry signal (p = 0.0044) but the geometric inversion was identifiable only under the bilateral cost — *not* absolutely accurate.

This notebook replaces that method with the **Kandlikar 2020 Levenberg–Marquardt** approach:

| Aspect | Stage 4 (Mukhmetov) | Stage 4b (Kandlikar-LM) |
|---|---|---|
| Inverse engine | Nelder-Mead grid search | Levenberg–Marquardt |
| Parameters optimised | $(z_t, r_t)$ — 2 DoF | $(x_t, y_t, z_t, d)$ — 4 DoF |
| Lateral pin | IR hotspot centroid (fixed) | free, starts from IR hotspot |
| Cost function | Bilateral asymmetry | Direct MSE of skin temperature |
| Paper | Mukhmetov et al. (2023) | Gonzalez-Hernandez/Kandlikar (2020) |
| Validation | Bilateral synthetic, 9 scenarios | Self-consistent synthetic (same-mesh) |
| Detection criterion | Not implemented | Outside bbox or $d < 4$ mm → benign |

**PINN is dropped entirely** — the three PINN variants all failed (Tanh saturation gives 7 °C forward-model noise, drowning the 0.67 °C tumour signal; SNR 0.09:1).

---

## 1. Theoretical Background

### 1.1 Physiological basis

A malignant tumour sustains anomalously high metabolic activity and chaotic angiogenesis. Both elevate the local internal temperature. Via heat conduction, this perturbation reaches the skin as a focal hotspot, asymmetric between the two breasts. The Kandlikar method asks: given the surface temperature field, what sub-surface source produces it?

### 1.2 Pennes bioheat equation — steady-state governing PDE

The **Pennes equation** (steady-state) governs heat transfer in perfused tissue:

$$k \nabla^2 T(\mathbf{x}) - P(\mathbf{x})\bigl(T(\mathbf{x}) - T_a\bigr) + Q(\mathbf{x}) = 0, \qquad \mathbf{x} \in \Omega$$

where the **perfusion coefficient** is $P(\mathbf{x}) = \omega(\mathbf{x})\,\rho_b\,c_b$ and $Q(\mathbf{x})$ is the total metabolic heat generation rate.

All constants come directly from **Kandlikar 2020, Table 3** (Gonzalez-Hernandez et al., *Infrared Physics & Technology* 105, 2020):

| Symbol | Meaning | Value | Unit |
|---|---|---|---|
| $k_h$ | Healthy-tissue thermal conductivity | 0.42 | W/(m·K) |
| $\rho_b$ | Blood density | 1060 | kg/m³ |
| $c_b$ | Blood specific heat | 3840 | J/(kg·K) |
| $\omega_h$ | Healthy-tissue perfusion rate | $1.8 \times 10^{-4}$ | s⁻¹ |
| $\omega_t$ | Tumour perfusion rate | $9.0 \times 10^{-3}$ | s⁻¹ |
| $Q_h$ | Healthy metabolic heat | 450 | W/m³ |
| $T_a$ | Arterial / core body temperature | 37 | °C |
| $h$ | Skin convection coefficient | 13.5 | W/(m²·K) |
| $T_\infty$ | Ambient air temperature | 21 | °C |

> **Critical:** the old pipeline used $\rho_b = 1.0$ kg/m³ (incorrect), giving $P_h = \omega_h\rho_b c_b \approx 0.65$ W/(m³·K) and thermal penetration depth $\ell = \sqrt{k/P_h} \approx 800$ mm — far exceeding breast thickness, making depth completely unobservable. The paper's correct value $\rho_b = 1060$ kg/m³ gives $P_h \approx 733$ W/(m³·K) and $\ell \approx 24$ mm — breast-scale, so depth is identifiable. This is the physical root of the prior depth non-identifiability.

### 1.3 Spatially-varying perfusion and metabolic heat (tumour indicator)

The tumour perturbs both perfusion and metabolic heat inside a region of radius $r_t = d/2$ centred at $\mathbf{x}_t = (x_t, y_t, z_t)$. Rather than a sharp sphere, a smooth **Gaussian indicator** is used:

$$\chi(\mathbf{x}) = \exp\!\left(-\frac{\|\mathbf{x} - \mathbf{x}_t\|^2}{r_t^2}\right)$$

This gives $\chi = 1$ at the tumour centre and decays smoothly outward, avoiding mesh-dependent artefacts from hard interfaces.

The spatially-varying fields are:

$$P(\mathbf{x}) = P_h + (P_t - P_h)\,\chi(\mathbf{x}), \qquad Q(\mathbf{x}) = Q_h + (Q_t - Q_h)\,\chi(\mathbf{x})$$

where $P_h = \omega_h\rho_b c_b$ and $P_t = \omega_t\rho_b c_b$.

### 1.4 Tumour metabolic heat — Gautherie law

$Q_t$ is not optimised; it is pinned physiologically via the **Gautherie tumour-doubling-time law**:

$$Q_t\,\tau = C = 3.27 \times 10^6 \quad \text{W·day/m}^3$$

where $\tau$ is the tumour doubling time (a decreasing function of radius — larger tumours are more aggressive). In code: `R.tumor_Qm_from_radius(r_t)`. This gives $Q_t \approx 65\,400$ W/m³ at $d=10$ mm down to $\approx 7\,800$ W/m³ at $d=22$ mm.

### 1.5 Boundary conditions

| Boundary | Type | Math form | Values |
|---|---|---|---|
| Skin (anterior surface) | Robin / convective | $-k\,\partial T/\partial n = h(T - T_\infty)$ | $h=13.5$ W/(m²·K), $T_\infty=21$ °C |
| Chest wall (posterior cap) | Dirichlet | $T = T_a$ | $T_a = 37$ °C |
| Tumour interior | Internal source (no BC) | raises $P$, $Q$ via $\chi$ | see §1.3–§1.4 |

**Chest-wall identification** uses a **Z-slab**: the posterior `CHEST_WALL_FRAC = 0.30` (30 %) of the mesh depth from the highest-Z end. A normal-based criterion (`$n_z > 0$`) was tried and rejected — on a curved bilateral breast surface, upper and lateral skin also has outward normals with a positive Z component, causing over-selection (half the breast coloured as chest wall in early tests). The Z-slab is deterministic, geometry-free, and consistent between the surface visualisation, the FEM boundary facets, and the LM skin mask.

The chest wall identified on the **surface** (for visualisation) uses vertex Z coordinates. The chest wall enforced in the **FEM** (for the Dirichlet BC) uses **facet midpoint Z** — because FEniCSx applies BCs to facet sets, not vertices, and facet midpoints are the physically correct BC evaluation points.

### 1.6 Weak form (FEniCSx / UFL)

The weak form of the Pennes equation with Robin skin BC and Dirichlet chest-wall BC, in the function space $V = H^1(\Omega)$, is:

Find $T \in V$ such that for all test functions $v \in V$:

$$\int_\Omega k\,\nabla T \cdot \nabla v\,\mathrm{d}V + \int_\Omega P(\mathbf{x})\,T\,v\,\mathrm{d}V + \int_{\Gamma_s} h\,T\,v\,\mathrm{d}S = \int_\Omega \bigl(P(\mathbf{x})\,T_a + Q(\mathbf{x})\bigr)\,v\,\mathrm{d}V + \int_{\Gamma_s} h\,T_\infty\,v\,\mathrm{d}S$$

where $\Gamma_s$ is the convective skin boundary. The chest-wall Dirichlet condition $T=T_a$ on $\Gamma_{cw}$ is imposed strongly (as a `fem.dirichletbc`). The resulting linear system is solved by **CG + GAMG preconditioner** (textbook mesh-independent convergence, ~17 iterations at 3 mm from the Stage 4 mesh study, §6.5 of Stage4_InverseBioheat.md).

---

## 2. Coordinate System and Anatomy

Understanding which direction is which matters for the quadrant classifier and the inward-normal tumour placement.

| Axis | Direction | Anatomical meaning |
|---|---|---|
| X | increases rightward (patient's left) | lateral separation of the two breasts |
| Y | increases **inferiorly** (downward) | superior = **smaller Y** |
| Z | increases posteriorly | anterior (nipple) = **low Z**, chest wall = **high Z** |

This follows from how `project_and_sample` feeds the NeRF UV grid: world Y maps to the vertical image axis, and `grid_sample` maps `y = -1` to the top row (head = superior), so superior is indeed smaller Y.

**Midline:** $x_{mid} = \text{mean}(x)$ over all surface vertices — the sternum lies approximately here. The two breasts are at $x > x_{mid}$ and $x < x_{mid}$.

---

## 3. Breast Quadrant Classification

### 3.1 Literature quadrant distribution

Breast cancer is not uniformly distributed. Large epidemiological registries establish the following quadrant probabilities (Senie *et al.*, *Am J Epidemiol* 1983; Kricker *et al.*, *Breast Cancer Res Treat* 2007; Chung *et al.*, *Anticancer Res* 2005):

| Quadrant | Code | Probability |
|---|---|---|
| Upper-Outer | UO | 50 % |
| Central / subareolar | C | 18 % |
| Upper-Inner | UI | 15 % |
| Lower-Outer | LO | 11 % |
| Lower-Inner | LI | 6 % |

Always planting synthetic tumours in the upper-outer quadrant would over-represent one case, biasing the validation. The notebook draws the quadrant from this distribution with a fixed random seed (changeable for sensitivity analysis).

### 3.2 Per-breast quadrant definition

The classifier `quadrants(geo)`:

1. **Skin mask:** $z < z_{min} + (1 - \texttt{CHEST\_WALL\_FRAC})(z_{max} - z_{min})$ — Z-slab, excludes chest wall.
2. **Hotspot detection:** warmest skin vertex from `T_measured` → determines which breast is the affected side.
3. **Side split:** $x > x_{mid}$ (right breast, positive X side) or $x < x_{mid}$ (left), per the hotspot.
4. **Breast centroid** $\mathbf{c}$: mean of all skin vertices on the affected breast.
5. **Upper vs lower:** $y < c_y$ → upper (smaller Y = superior). $y > c_y$ → lower.
6. **Outer vs inner:** away from $x_{mid}$ → outer (flips with affected side). Toward $x_{mid}$ → inner.
7. **Central:** within 25 mm of the **nipple proxy** — the most anterior (minimum Z) vertex on the affected breast skin. This is the clinical definition of subareolar. *Bug fixed 2026-06-22: the original code used distance from the breast centroid $\mathbf{c}$, which lies inside the breast volume ~30–50 mm from every surface vertex, giving 0 Central vertices and silently biasing all C draws to UO (effective UO = 68 %).*
8. **Contralateral:** all vertices on the other breast are labelled `'other'` (shown grey).

The 25 mm central radius approximates the subareolar region (~8% of the anterior breast area at a ~75 mm breast radius: $\pi\times25^2/\pi\times75^2 \approx 11\%$, consistent with the 18% literature value given that central tumours are defined more liberally in epidemiological registries).

---

## 4. Synthetic Tumour Placement

### 4.1 Why placement must be anatomically correct

The Kandlikar paper validates on known synthetic tumours. If the planted position is unrealistic — e.g. always in the exact centre of the UOQ regardless of breast shape, or placed outside the breast volume — the validation is meaningless. Two principles govern the placement:

1. **Anatomically realistic quadrant** — sampled from the literature distribution (§3.1).
2. **Geometrically guaranteed inside the breast** — pushed inward along the mean inward normal of the quadrant surface patch, so the tumour sphere is inside the volume even for thin lateral tissue.

### 4.2 Inward-normal placement

For the chosen quadrant, the function `plant_synthetic_tumor()`:

1. Finds all surface vertices belonging to the quadrant $\to$ patch $P$.
2. Computes $\mathbf{c}_{surf} = \text{mean}(\mathbf{x}_i)_{i \in P}$ — the quadrant surface centroid.
3. Computes $\hat{\mathbf{n}}_{in} = -\text{mean}(\hat{\mathbf{n}}_i)_{i \in P} / \|\ldots\|$ — the mean **inward** normal (outward normals point away from tissue; negating gives inward).
4. Samples diameter $d \sim \text{Uniform}(10, 20)$ mm (TNM T1 stage).
5. Samples skin clearance $c_{skin} \sim \text{Uniform}(5, 20)$ mm — the gap between the surface and the tumour edge. Thermographically detectable range: 5–25 mm (Gautherie & Gros, *Radiology* 1980; Ng & Sudharsan, *Proc IMechE H* 2001; Kandlikar 2020 — surface signal at $\sim 25$ mm depth falls below IR noise floor of ~0.1 °C).
6. Computes the tumour centre: $\mathbf{x}_t = \mathbf{c}_{surf} + \left(\tfrac{d}{2} + c_{skin}\right)\hat{\mathbf{n}}_{in}$.

The push distance $\tfrac{d}{2} + c_{skin}$ ensures the entire tumour sphere ($r_t = d/2$) is at least $c_{skin}$ mm below the surface.

### 4.3 Reproducibility

A fixed seed `np.random.default_rng(42)` makes the synthetic scenario reproducible. Changing the seed generates alternative scenarios for sensitivity analysis without altering any other parameters.

---

## 5. Forward FEM Solve

### 5.1 Mesh generation

From the NeRF surface (`surface_pts`, `faces` in mm):

```python
save_stl_binary(stl_path, geo['surface_pts'], geo['faces'])
stl_to_tet_mesh(stl_path, msh_path, mesh_size_mm=3.0)
```

This produces a volumetric mesh of ~144 k nodes and ~835 k tetrahedral elements (at 3 mm element size for Patient_159). The mesh is generated **once** and reused.

The NeRF reconstruction settings matter — the mesh is (re)generated each run to match the current settings: `GAUSSIAN_SIGMA=3.0`, `MC_THRESHOLD=None`, `REPAIR_HOLES=True`, `LAPLACIAN_ITERS=20`, `THERMAL_SIGMA=0.0`.

### 5.2 FEM forward call — `pennes_forward(msh_path, beta, geo)`

The forward model solves the Pennes BVP for a given `beta = {x_t_mm, y_t_mm, z_t_mm, d_mm, k_h}`:

1. Read the `.msh` file (mm) → scale to metres for SI units.
2. Build $\chi(\mathbf{x})$ and spatially-varying $P(\mathbf{x})$, $Q(\mathbf{x})$ as UFL expressions — all evaluated symbolically at quadrature points, no Python loops.
3. Identify chest-wall facets via Z-slab on **facet midpoints** (in metres).
4. Assemble and solve the weak form (CG/GAMG).
5. Interpolate the nodal solution onto `geo['surface_pts']` via `compute_fea_residual`.

The output `T_surf` (shape: `len(geo['surface_pts'])`) is the simulated skin temperature at every surface vertex, in °C.

### 5.3 Why `pennes_forward` is separate from `PennesCtx`

`pennes_forward` reads the mesh fresh every call (needed for §7 which uses the 3 mm mesh). `PennesCtx` (§6) reads the **same** 3 mm mesh once and caches it — saving ~12 s of mesh I/O overhead per LM iteration. The LM needs ~125 forward evaluations (25 iterations × 5 solves each); that is ~25 minutes total at ~12 s/solve with the 3 mm mesh (measured: 13.1 s for Patient_159).

---

## 6. Persistent Solver Context — `PennesCtx`

### 6.1 Motivation

The LM inverse requires many repeated FEM solves on the **same mesh** with only the tumour parameters changing between calls. Reading and meshing the `.msh` file each time would cost ~12 s × 125 calls ≈ 25 minutes of pure I/O overhead, doubling the total run time. `PennesCtx` eliminates this.

### 6.2 What is cached

In `__init__` (run once):

- The FEniCSx mesh object (`self.msh`) — mesh geometry and topology.
- The function space `self.V` (Lagrange P1) and spatial coordinate expression `self.Xc`.
- The chest-wall Dirichlet BC (`self.bc`) — fixed for all $\beta$ because the Z-slab does not depend on the tumour.
- The surface-interpolation index `self.sidx` — for each of the $N$ surface points in `geo['surface_pts']`, the 3 nearest FEM nodes (from a `cKDTree` query). Used in `solve()` to extract surface temperature by averaging the 3 nearest nodal values.

### 6.3 What is reassembled per call

In `solve(beta)`:

- The Gaussian indicator $\chi$ as a UFL expression (depends on $\mathbf{x}_t$, $r_t$).
- The bilinear form `a` and linear form `L` (depend on $\chi$, $k_h$, $\omega_h$, $\omega_t$, etc.).
- The `LinearProblem` object and its PETSc solve.

The mesh topology, DOF numbering, and BC are reused unchanged.

### 6.4 Surface interpolation

```python
return sol.x.array[self.sidx].mean(axis=1)  # shape: (N_surface,)
```

`self.sidx` has shape $(N_{surface}, 3)$; indexing the nodal solution array and averaging over the 3 nearest nodes gives a smooth interpolant at each surface point, robust to the fact that surface points from the NeRF mesh do not generally coincide with FEM nodes.

---

## 7. Levenberg–Marquardt Inverse

### 7.1 Problem statement

Given measured skin temperatures $\mathbf{T}^{meas} \in \mathbb{R}^{N_{skin}}$ (restricted to the skin-only Z-slab), find the tumour parameters $\boldsymbol{\beta} = [x_t, y_t, z_t, d]$ that minimise the **direct MSE cost**:

$$S(\boldsymbol{\beta}) = \frac{1}{N_{skin}} \sum_{i \in \text{skin}} \bigl(T_i^{meas} - T_i^{FEM}(\boldsymbol{\beta})\bigr)^2$$

The residual vector is $\mathbf{r}(\boldsymbol{\beta}) = \mathbf{T}^{meas} - \mathbf{T}^{FEM}(\boldsymbol{\beta}) \in \mathbb{R}^{N_{skin}}$, so $S = \tfrac{1}{N_{skin}}\|\mathbf{r}\|^2$.

$k_h$ is held at the paper value (not optimised). $Q_t$ is derived from $d$ via the Gautherie law (not a free parameter).

### 7.2 Jacobian — finite-difference, residual convention

The Jacobian $J \in \mathbb{R}^{N_{skin} \times 4}$ is defined as:

$$J_{ij} = \frac{\partial r_i}{\partial \beta_j} = -\frac{\partial T_i^{FEM}}{\partial \beta_j}$$

It is computed by **forward finite differences** with per-parameter step sizes $\mathbf{h} = [3, 3, 3\,\text{mm},\, 2\,\text{mm}]$:

$$J_{ij} \approx \frac{r_i(\boldsymbol{\beta} + h_j \mathbf{e}_j) - r_i(\boldsymbol{\beta})}{h_j}$$

Each column of $J$ costs one FEM solve. Total per LM iteration: 4 Jacobian solves + at least 1 trial step = 5 FEM solves minimum.

### 7.3 LM normal equation — sign convention

The Levenberg–Marquardt update minimises the linearised cost $\|\mathbf{r} + J\,\delta\boldsymbol{\beta}\|^2$ subject to a trust-region penalty (Marquardt diagonal scaling):

$$\bigl(J^\top J + \mu\,\mathrm{diag}(J^\top J)\bigr)\,\delta\boldsymbol{\beta} = -J^\top \mathbf{r}$$

Solving for the Newton step:

$$\delta\boldsymbol{\beta} = -\bigl(J^\top J + \mu\,D\bigr)^{-1} J^\top \mathbf{r}$$

The **minus sign on $J^\top \mathbf{r}$** is mandatory for the **residual-Jacobian convention** ($J = \partial\mathbf{r}/\partial\boldsymbol{\beta}$). Without it, $\delta\boldsymbol{\beta}$ points *up* the cost gradient instead of down — LM would ascend and stop at the first iteration. This was the original bug: `np.linalg.solve(A, Jtr)` was used instead of `np.linalg.solve(A, -Jtr)`.

> **Sign derivation check.** Suppose $T^{FEM} < T^{meas}$ (model too cold). Then $\mathbf{r} > 0$. To reduce $S$, we need $T^{FEM}$ to increase, e.g. by increasing $d$ (larger tumour → more heat). $\partial T^{FEM}/\partial d > 0$, so $J_{d\text{-col}} = \partial r/\partial d = -\partial T^{FEM}/\partial d < 0$. Then $J^\top \mathbf{r} = (\text{negative}) \cdot (\text{positive}) < 0$, so $-J^\top \mathbf{r} > 0$, and $\delta d > 0$ — correctly increasing $d$. With the wrong sign, $\delta d < 0$, decreasing $d$ and making the model colder.

### 7.4 Marquardt scaling

The diagonal matrix $D = \mathrm{diag}(J^\top J)$ absorbs the dimensional heterogeneity among $[x_t, y_t, z_t, d]$ (all in mm, but with very different cost sensitivities). This is preferable to an identity penalty because it automatically scales trust-region steps to the local curvature in each direction.

The damping parameter $\mu$ follows the standard Marquardt schedule:
- On accepted step: $\mu \leftarrow \max(\mu \times 0.3,\; 10^{-9})$ — reduces regularisation as the minimum approaches.
- On rejected step (up to 8 attempts): $\mu \leftarrow \min(\mu \times 3.0,\; 10^{8})$ — increases regularisation, shrinks the step.

Initial value: $\mu = 10^{-2}$.

### 7.5 Convergence criteria

The loop runs for up to 25 iterations and stops early when:
- $\|\delta\boldsymbol{\beta}\| < 0.05$ mm (parameter change negligible), or
- $S < 10^{-6}$ °C² (cost at machine-zero level — self-consistent synthetic).

If no backtracking step is accepted (all 8 attempts with $\mu$ growing to $10^8$), the loop stops with a "no improvement" message.

### 7.6 Initial guess

LM is initialised at the **IR thermogram hotspot**, not at the planted value:

- $x_0, y_0$ = XY coordinates of the skin vertex with the highest temperature in `T_lm_target`.
- $z_0$ = breast mid-depth = $\tfrac{1}{2}(z_{min} + z_{max})$.
- $d_0 = 10$ mm.

For synthetic mode (`SYNTHETIC_TEST = True`), the target is `ctx.solve(plant)[skin]` — the FEM temperature of the planted tumour, generated on the same mesh as the LM. The starting point is the IR hotspot of *that* temperature, not of `geo['T_measured']`. This is honest: LM must find the tumour without knowing the answer.

### 7.7 Two modes

```python
SYNTHETIC_TEST = True   # ← flip to False for real-patient inverse
```

| Mode | Target | Starting point | Measures |
|---|---|---|---|
| `True` | `ctx.solve(plant)[skin]` — FEM of planted tumour on same 3mm mesh | hotspot of that FEM field + mid-depth Z | Optimizer correctness: can LM recover the known answer? |
| `False` | `geo['T_measured'][skin]` — real NeRF IR surface temperature | hotspot of real IR + mid-depth Z | Clinical inference: where is the tumour in this patient? |

In synthetic mode, the cost floor is **zero** — the same `ctx` object generated both the target and the model prediction, so at `beta = plant`, the residual is identically zero up to PETSc solver tolerance ($\approx 10^{-7}$ relative). LM should drive RMS → 0 and recover `plant` exactly.

---

## 8. Identifiability — Jacobian SVD Rank (§8.4)

The paper (Gonzalez-Hernandez *et al.* 2020) found **rank 5 of 8** with 8 camera views and 8 parameters; with 5 views and 4 parameters, we expect rank 4 (fully determined). The notebook verifies this at the LM solution:

$$J = U\,\Sigma\,V^\top, \qquad \text{rank} = \#\{\sigma_i > 10^{-3}\,\sigma_1\}$$

**Parameter sensitivity** is measured as the RMS surface temperature change for a one-step perturbation:

$$s_j = \frac{\|J_{:,j}\| \cdot h_j}{\sqrt{N_{skin}}} \qquad [\text{°C per step}]$$

A parameter with $s_j \lesssim 10^{-3}$ °C/step is flat — not resolvable from the surface data alone. Typically, depth $z_t$ has lower sensitivity than lateral $(x_t, y_t)$ due to the diffusive smoothing of the heat conduction path.

---

## 9. Detection Criterion (§8.3)

After LM converges to $\hat{\boldsymbol{\beta}}$, the recovered tumour is tested:

| Condition | Verdict |
|---|---|
| Centre outside breast bbox + 5 mm padding | **Benign** (no confined source) |
| Diameter $\hat{d} < 4$ mm | **Benign** (vanishing source) |
| Otherwise | **Suspicious → Malignant** |

The 5 mm pad avoids false-benign verdicts from tumours recovered just on the boundary. The 4 mm floor avoids spurious "tumours" where LM drives $d \to 0$ in a flat region.

The tumour position is **intentionally unconstrained** during LM. If there is genuinely no tumour, the cost landscape is flat and LM drifts — possibly outside the breast. That drift is the detection signal. Constraining $\boldsymbol{\beta}$ to the breast volume would suppress this and destroy the detection capability.

The detection plot (§8.3) overlays:
- **Lime sphere** — recovered tumour ($\hat{\mathbf{x}}_t$, $\hat{d}/2$).
- **Cyan sphere (semi-transparent)** — planted ground-truth tumour (synthetic mode only).
- **Inferno colormap** on the breast surface — temperature from `ctx.solve(beta_hat)`.

---

## 10. ANSYS-Style Documentation Map

The notebook follows the ANSYS *Engineering Data → Geometry → Mesh → BCs → Setup → Solution → Results* workflow:

| Section | ANSYS analogue | Notebook cell | Content |
|---|---|---|---|
| §0 | — | Imports | `run_cohort`, plotly, dolfinx, mpi4py |
| §1 | Engineering Data | Cell 4 | Paper constants (Table 3); penetration depth check; `apply_constants()` |
| §2 | Geometry | Cell 8 | `load_patient_geo` with mandated mesh settings |
| §3 | Mesh | Cell 9 | STL export + Gmsh tet mesh at 3 mm |
| §3b | Mesh inspection | Cell 11 | Interactive 3D: surface + subsampled nodes |
| §4 | Boundary Conditions | Cell 13 | Z-slab chest wall; blue/red BC plot with legend |
| §5 | Loads / Prior | Cells 14–15 | Literature quadrant distribution; `plant_synthetic_tumor()` |
| §6 | Setup / Model | Cell 17 | `pennes_forward` function definition |
| §7 | Solution | Cell 19 | Forward solve; `T_surf`; surface temperature plot |
| §8.1 | Setup (LM context) | Cell 21 | `PennesCtx` — read mesh once, cache BC + sidx |
| §8.2 | Solution (LM) | Cell 22 | LM loop; Jacobian; convergence output |
| §8.3 | Results | Cell 23 | Detection; overlaid planted vs recovered plot |
| §8.4 | Post-processing | Cell 24 | Jacobian SVD; per-parameter sensitivity |

---

## 11. Equations Summary

**Pennes PDE (strong form):**
$$k\nabla^2 T - P(\mathbf{x})(T - T_a) + Q(\mathbf{x}) = 0 \quad \text{in } \Omega$$

**Tumour indicator:**
$$\chi(\mathbf{x}) = \exp\!\left(-\frac{\|\mathbf{x}-\mathbf{x}_t\|^2}{r_t^2}\right), \quad r_t = d/2$$

**Spatially-varying fields:**
$$P(\mathbf{x}) = P_h + (P_t-P_h)\chi, \quad Q(\mathbf{x}) = Q_h + (Q_t-Q_h)\chi$$

**Boundary conditions:**
$$T\big|_{\Gamma_{cw}} = T_a = 37\,^\circ\text{C}, \qquad -k\frac{\partial T}{\partial n}\bigg|_{\Gamma_s} = h(T-T_\infty)$$

**Inverse cost (direct MSE):**
$$S(\boldsymbol{\beta}) = \frac{1}{N_{skin}}\sum_{i\in\text{skin}}\!\bigl(T_i^{meas} - T_i^{FEM}(\boldsymbol{\beta})\bigr)^2, \quad \boldsymbol{\beta} = [x_t, y_t, z_t, d]$$

**LM normal equation (residual-Jacobian sign convention):**
$$\bigl(J^\top J + \mu\,\mathrm{diag}(J^\top J)\bigr)\,\delta\boldsymbol{\beta} = -J^\top \mathbf{r}, \quad J_{ij}=\frac{\partial r_i}{\partial\beta_j}=\frac{r_i(\boldsymbol{\beta}+h_j\mathbf{e}_j)-r_i(\boldsymbol{\beta})}{h_j}$$

**LM update:**
$$\boldsymbol{\beta} \leftarrow \boldsymbol{\beta} + \delta\boldsymbol{\beta}$$

**Penetration depth (identifiability check):**
$$\ell = \sqrt{\frac{k_h}{P_h}} = \sqrt{\frac{k_h}{\omega_h\rho_b c_b}} \approx 24\,\text{mm} \quad \Rightarrow \text{depth identifiable at breast scale}$$

---

## 12. Differences from Kandlikar 2020 Paper

The notebook implements the Kandlikar 2020 method with several adaptations to the DMR-IR data:

| Aspect | Kandlikar 2020 paper | This implementation |
|---|---|---|
| Patient geometry | 3D scanner (single breast, prone) | TherMAM-NeRF from 2D IR (bilateral, upright) |
| Views / cameras | 8 views | 5 views (Anterior, L/R Lateral 90°, L/R Oblique 45°) |
| LM parameters | 8: position, size, $k_h$, $Q_t$, $\omega$, $h$ | 4: $[x_t, y_t, z_t, d]$ ($k_h$ fixed, $Q_t$ Gautherie-derived) |
| Tumour placement (synthetic) | Not specified | Literature quadrant distribution (§3.1) + inward-normal push (§4.2) |
| Chest-wall BC | Dirichlet 37 °C | Dirichlet 37 °C, Z-slab identification |
| Skin BC | Robin $h=13.5$ | Robin $h=13.5$ (identical) |
| Mesh | Not detailed | Gmsh 3 mm tet (~144 k nodes Patient_159) |
| Jacobian | Finite difference | Forward finite difference, same step sizes |
| Breast count | 1 | 2 (bilateral) — contralateral shown grey, not used in LM |

The 8-parameter version (adding $k_h$, $h$, $\omega$) is a future extension once the 4-parameter synthetic validation is confirmed.

---

## 13. Running the Notebook

```bash
cd /mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/TherMAM-NeRF

# activate the bioheat conda environment (FEniCSx, dolfinx, gmsh)
conda activate bioheat
jupyter notebook FEM_LM_Inverse_v1.ipynb
```

**Key toggles:**

| Variable | Location | Effect |
|---|---|---|
| `VIZ_IDX` | Cell 8 | Which patient (dataset index) to run |
| `R.CHEST_WALL_FRAC` | Cell 13 | Fraction of posterior depth assigned to chest wall (default 0.30) |
| `rng = np.random.default_rng(seed)` | Cell 15 | Seed for quadrant + depth + diameter sampling |
| `SYNTHETIC_TEST` | Cell 22 | `True` = fit FEM-generated data; `False` = fit real IR |
| `MAX_ITER` | Cell 22 | LM iteration limit (default 25) |

**Expected run times (Patient_159, 3 mm mesh, 144 k nodes):**

| Step | Time |
|---|---|
| `load_patient_geo` | ~30 s |
| `stl_to_tet_mesh` (3 mm) | ~20 s |
| `pennes_forward` (§7) | ~13 s |
| `PennesCtx.__init__` (reads same mesh) | ~13 s |
| LM iteration (4 Jacobian solves + 1 step = 5 FEM solves) | ~65 s |
| Full LM (25 iterations, optimistic) | ~27 min |

---

---

## 3.6.3 Convergence Analysis

This section documents the numerical convergence properties of every computational layer in the pipeline: the FEM discretisation (mesh), the finite-difference Jacobian, the linear solver, and the nonlinear LM loop. Convergence failures at any layer contaminate all layers above it.

---

### a) Mesh Convergence (h-convergence)

#### Motivation

The FEM forward model maps tumour parameters $\boldsymbol{\beta}$ to a discrete surface temperature $\mathbf{T}^{FEM}(\boldsymbol{\beta})$. If the mesh is too coarse, the mapping contains discretisation error $\epsilon_h$:

$$T^{FEM}_h(\boldsymbol{\beta}) = T^{exact}(\boldsymbol{\beta}) + \mathcal{O}(h^2)$$

for Lagrange P1 elements on shape-regular tetrahedra. For the LM inverse, $\epsilon_h$ acts as an irreducible residual floor — LM cannot drive $S(\boldsymbol{\beta})$ below $\epsilon_h^2$ even at the true answer, causing premature termination.

#### h-convergence study (reference data from Stage 4 mesh study)

From `Stage4_InverseBioheat.md §6.5`, surface temperature was solved at four element sizes for the same patient and boundary conditions:

| $h$ (mm) | Nodes (approx.) | Solve time (s) | Max $|T_h - T_{1mm}|$ (°C) | LM cost floor (°C²) |
|---|---|---|---|---|
| 5 | ~25 k | ~2 | ~0.08 | ~6.4×10⁻³ |
| 3 | ~145 k | ~13 | ~0.02 | ~4×10⁻⁴ |
| 2 | ~470 k | ~42 | ~0.006 | ~3.6×10⁻⁵ |
| 1 | ~3.5 M | ~310 | 0 (reference) | ~0 |

**Recommendation: $h = 3$ mm.** At this resolution the discretisation error (~0.02 °C) is well below the tumour signal (~0.5 °C peak rise from a 14 mm tumour) and below the LM convergence tolerance used (RMS < 0.01 °C in synthetic mode). Going to $h = 2$ mm increases solve time 3× with only marginal gain; $h = 5$ mm produces a cost floor (~0.08 °C² RMS) that blocks LM convergence.

**Self-consistency note.** In `SYNTHETIC_TEST = True` mode, both the LM target $\mathbf{T}^{target} = \texttt{ctx.solve(plant)}$ and the LM model $\mathbf{T}^{FEM}(\boldsymbol{\beta})$ are evaluated on the **same** 3 mm mesh using the **same** `PennesCtx` object. Discretisation errors cancel identically — the cost floor is zero (up to PETSc solver tolerance $\sim 10^{-14}$ °C²), making synthetic validation a true algebraic self-consistency test.

---

### b) Finite-Difference Step Size

#### Choice of step vector

The Jacobian columns are computed by forward finite differences:

$$J_{ij} \approx \frac{r_i(\boldsymbol{\beta} + h_j \mathbf{e}_j) - r_i(\boldsymbol{\beta})}{h_j}$$

The step sizes used are:

$$\mathbf{h} = [h_x, h_y, h_z, h_d] = [3, 3, 3, 2] \text{ mm}$$

#### Justification

Two competing errors govern the optimal $h$:

1. **Truncation error** (large $h$): $J_{ij}^{FD} = J_{ij}^{true} + \mathcal{O}(h_j)$ — forward FD has first-order truncation bias. Reducing $h$ reduces this.

2. **Cancellation/discretisation noise** (small $h$): When $h_j < \epsilon_h$ (mesh discretisation error), $r_i(\boldsymbol{\beta} + h_j\mathbf{e}_j) - r_i(\boldsymbol{\beta})$ is dominated by FEM discretisation noise, not by the true gradient. The signal-to-noise ratio collapses.

The mesh element size is $h_{mesh} = 3$ mm. Using FD step sizes equal to one mesh element satisfies $h_j \approx h_{mesh}$, keeping both errors comparable:

$$\frac{\partial T^{FEM}}{\partial x_t}\bigg|_{h_j = h_{mesh}} \approx \frac{\Delta T}{h_j} \approx \frac{0.05\text{°C}}{3\text{mm}} \approx 0.017\text{ °C/mm}$$

This is well above the solver noise floor ($\sim 10^{-7}$ °C, from PETSc CG tolerance) and below the truncation bias scale.

For diameter $d$: $h_d = 2$ mm was chosen slightly smaller because $T^{FEM}$ is more sensitive to $d$ than to lateral position (the sensitivity $s_d = 0.0045$ °C/step from §8.4 vs $s_{xyz} \approx 0.0002$ °C/step at the wrong-start local minimum). A smaller step reduces truncation bias where the signal is larger.

#### Gradient sensitivity table (from §8.4 at LM solution)

| Parameter | Step $h_j$ | Sensitivity $s_j$ (°C/step) | Status |
|---|---|---|---|
| $x_t$ | 3 mm | ~0.0002 | Identifiable near correct basin |
| $y_t$ | 3 mm | ~0.0002 | Identifiable near correct basin |
| $z_t$ | 3 mm | ~0.0002 | Weaker (diffusion smoothing) |
| $d$ | 2 mm | ~0.0045 | Most identifiable |

> **Note:** These values are measured at the *wrong* local minimum (d ≈ 3.4 mm) where x, y, z sensitivities appear flat because the tiny tumour generates almost no gradient anywhere. At the *correct* answer, sensitivities in all four parameters will be larger.

---

### c) Linear Solver Convergence (CG + GAMG)

#### Solver configuration

Every `PennesCtx.solve(β)` call assembles and solves the symmetric positive-definite linear system:

$$\mathbf{A}(\boldsymbol{\beta})\,\mathbf{T} = \mathbf{b}(\boldsymbol{\beta})$$

using **PETSc Conjugate Gradient (CG)** preconditioned by **GAMG** (Geometric-Algebraic MultiGrid).

```python
petsc_options = {'ksp_type': 'cg', 'pc_type': 'gamg'}
```

#### Why CG is appropriate

The Pennes BVP weak form produces a matrix $\mathbf{A}$ that is:
- **Symmetric**: $\int_\Omega k\nabla T\cdot\nabla v\,dV + \int_\Omega P(\mathbf{x})Tv\,dV + \int_{\Gamma_s} hTv\,dS$ — all bilinear forms are symmetric.
- **Positive-definite**: perfusion term $P(\mathbf{x}) > 0$ everywhere (even in healthy tissue $P_h = 733$ W/(m³·K)), Robin BC $h > 0$ — both add positive-definite contributions on top of the elliptic stiffness.

CG is the optimal Krylov method for SPD systems. GAMG provides $\mathcal{O}(1)$ iteration count independent of mesh refinement (mesh-independent convergence).

#### Measured convergence

From Stage 4 mesh study (same mesh, same BCs, same solver):

| $h$ (mm) | Condition number $\kappa(\mathbf{A})$ | CG iterations | Time/solve (s) |
|---|---|---|---|
| 5 | ~10³ | ~12 | ~2 |
| 3 | ~10⁴ | ~17 | ~13 |
| 2 | ~10⁵ | ~22 | ~42 |
| 1 | ~10⁶ | ~28 | ~310 |

GAMG effectively reduces the iteration count to $\mathcal{O}(\log(1/h))$ rather than $\mathcal{O}(1/h)$ for unpreconditioned CG, confirming its effectiveness. At 3 mm: **~17 CG iterations per FEM solve**, PETSc relative residual tolerance $10^{-6}$.

#### Impact on LM

Each CG solve introduces a residual $\|\mathbf{A}\hat{\mathbf{T}} - \mathbf{b}\|/\|\mathbf{b}\| < 10^{-6}$, which translates to a temperature error of approximately $10^{-6} \times \|\mathbf{T}\|_\infty \approx 10^{-6} \times 37\text{ °C} \approx 4\times10^{-5}$ °C — three orders of magnitude below the FD step signal ($\sim 0.05$ °C), confirming that linear solver noise does not contaminate the Jacobian.

---

### d) Nonlinear (LM) Convergence

#### Parameter history

A well-converging LM run (with correct differential-hotspot initialisation) should show:

| Iteration | RMS (°C) | $\mu$ | Behaviour |
|---|---|---|---|
| Start | ~0.15 | $10^{-2}$ | Far from truth; high residual |
| 1–3 | 0.05→0.01 | decreasing | Large steps toward tumour basin |
| 4–8 | 0.01→0.001 | decreasing | Refinement near correct position |
| 9–15 | <0.001 | very small | Fine adjustment; diameter converges |
| Converged | ~$10^{-4}$ | $\ll 1$ | $\|\delta\boldsymbol{\beta}\| < 0.05$ mm |

*(Pending: post results here after successful synthetic run with differential-hotspot initialisation.)*

#### Why initialisation determines success

LM is a **local** optimiser. The cost landscape $S(\boldsymbol{\beta})$ has at minimum two critical points:

1. **Global minimum** at $\boldsymbol{\beta} = \boldsymbol{\beta}_{plant}$ where $S = 0$ (in synthetic mode).
2. **Trivial local minimum** at $d \to 0$ anywhere — the "no-tumour" solution where the model produces the healthy-tissue temperature field. This has $S = S_{healthy} = \|\mathbf{T}^{plant} - \mathbf{T}^{healthy}\|^2 / N_{skin}$ ≈ the tumour's thermal signature squared, non-zero but potentially lower than $S$ at a badly-started position.

**Diagnosed failure mode (fixed by differential hotspot):**

| Starting point | Starting RMS | Converges to | Cause |
|---|---|---|---|
| Absolute IR hotspot (y=98 mm) | 0.18 °C | d=3.4 mm, BENIGN | 112 mm from truth; gradient at start points toward d→0 (trivial minimum) |
| Differential ΔT hotspot (y≈−14 mm) | expected ~0.05 °C | plant ≈ (38,−14,−38,14.4) | Within ~20 mm of truth; gradient points toward global minimum |

**Differential hotspot fix:**
$$\mathbf{T}^{healthy} = \texttt{ctx.solve}(\boldsymbol{\beta}_{healthy}), \quad \boldsymbol{\beta}_{healthy}: x_t = x_{max}+1000\text{ mm (outside mesh)}$$
$$\Delta\mathbf{T} = \mathbf{T}^{target} - \mathbf{T}^{healthy}, \quad \text{init XY} = \underset{i \in \text{skin}}{\arg\max}\,\Delta T_i$$

The hotspot of $\Delta\mathbf{T}$ is at the skin vertex directly above the tumour, independent of the chest-wall temperature background.

#### Real patient convergence (SYNTHETIC_TEST = False)

For real patients, the cost floor is non-zero due to model mismatch (wrong $k_h$, $h$, $\omega$ per patient; NeRF geometry error; ambient variation). Expected behaviour:

- RMS converges to a residual floor $\sim 0.3$–$1.5$ °C (not zero).
- Parameters converge to the least-squares best fit, which may differ from the true tumour location.
- Detection is based on whether the recovered $\hat{\boldsymbol{\beta}}$ is inside the breast and $\hat{d} > 4$ mm — a qualitative criterion, not a precision estimate.

*(Pending: real-patient LM convergence plots after switching `SYNTHETIC_TEST = False` for the 122-patient cohort.)*

---

### e) End-to-End Validation

#### Synthetic: recovery error distribution

**Stage 4b reference (Mukhmetov-style, 2D, bilateral cost)** — confirmed working on 9 scenarios, all < 0.8% error in $(z, r)$. This establishes that the FEM forward model is self-consistent and the mesh/BC combination is numerically sound.

| Method | DoF | Cost | Scenarios tested | Max depth err | Max radius err |
|---|---|---|---|---|---|
| Stage 4 (Nelder-Mead, bilateral) | 2 | Bilateral asymmetry | 9 | 0.57 % | 0.76 % |
| **Stage 4b (LM, direct MSE)** | **4** | **Direct MSE** | **pending** | **pending** | **pending** |

The 4-DoF LM on the same mesh should achieve comparable or better precision on the $z$ and $d$ dimensions, with the added advantage of recovering lateral position $(x_t, y_t)$ as well.

**Expected validation criteria (to be filled after run):**

| Criterion | Target | Status |
|---|---|---|
| Position error $\|\hat{\mathbf{x}}_t - \mathbf{x}_t\|$ | < 5 mm | Pending |
| Diameter error $|\hat{d} - d|$ | < 2 mm (paper accuracy) | Pending |
| Final RMS (synthetic) | < $10^{-3}$ °C | Pending |
| VERDICT (synthetic) | MALIGNANT for all planted cases | Pending |

#### Real patient: consistency checks

For `SYNTHETIC_TEST = False` on the 122-patient cohort, the absence of ground truth requires indirect consistency checks:

| Check | Method | Pass criterion |
|---|---|---|
| Seed stability | Re-run with 3 different random seeds; compare $\hat{\boldsymbol{\beta}}$ | Position variation < 10 mm |
| Mesh stability | Re-run at $h = 2$ mm; compare $\hat{\boldsymbol{\beta}}$ | Position variation < 5 mm |
| FD step stability | Halve all step sizes; compare Jacobian | Gradient direction change < 15° |
| Cost plausibility | Compare final RMS to healthy-tissue baseline | Final RMS < $S_{healthy}$ |
| Detection rate | % MALIGNANT in malignant vs benign patients | AUC > 0.6 (better than random) |

*(Pending: all real-patient results after cohort run.)*

---

## 14. Known Limitations and Next Steps

**Current limitations:**

1. **Synthetic only** — `SYNTHETIC_TEST = True` for now. The real-patient mode (`False`) fits `geo['T_measured']` which includes non-tumour thermal asymmetry (ambient, vascular, skin-fold effects). Recovery accuracy on real data is unknown without co-registered MRI/CT.

2. **4 parameters vs paper's 8** — $k_h$, $h$, $\omega_h$, $\omega_t$ are fixed at paper values. This reduces the inversion problem but makes it sensitive to per-patient departures from the paper constants.

3. **Single-breast LM** — the bilateral geometry is used only for quadrant classification. The cost is a direct MSE on the affected-breast skin; the contralateral breast does not contribute to $S(\boldsymbol{\beta})$. Extending to a bilateral LM cost (as in Stage 4) would suppress common-mode BC mismatch.

4. **No convergence guarantee** — LM is a local method. The starting point (IR hotspot XY + mid-depth Z) is physiologically motivated but not guaranteed to be in the basin of the correct minimum. Multi-start should be added when extending to the real cohort.

5. **Run time** — ~27 min per patient for 25 LM iterations on the 3 mm mesh. For the 122-patient cohort this is ~55 hours per GPU. A coarser mesh for initial iterations with final refinement at 3 mm, or parallel LM solves, would be needed.

**Planned extensions:**

- Confirm synthetic recovery (planted vs recovered position error < 5 mm) and record the result in §6 here.
- Switch to `SYNTHETIC_TEST = False` and run on the 122-patient cohort.
- Add bilateral asymmetry cost option (combining Stage 4's common-mode cancellation with Stage 4b's 4-DoF LM).
- Add $k_h$ as a 5th LM parameter (the paper shows it is identifiable with 8 views; with 5 views the Jacobian rank check in §8.4 will determine feasibility).