# 3D Breast Thermography Reconstruction & Inverse Bioheat

This repository documents a multi-stage research pipeline that reconstructs breast **shape and surface temperature** from multi-view thermographic images, and then estimates deep-tissue **tumour parameters** (position, depth, size) from that reconstruction by solving the inverse Pennes bioheat problem.

This README is written as a handoff document for future contributors, especially Zifa and Jessie. It explains what the **current** pipeline does, what **previous** approaches existed and **why each was superseded**, and how the pieces connect.

The project is guided by the canonical five-view acquisition protocol:

* Frontal: 0°
* Right oblique: +45°
* Left oblique: -45°
* Right lateral: +90°
* Left lateral: -90°

The sign convention above is used throughout the repository and matches the paper-aligned registration notes in the reconstruction code.

---

## Current Pipeline at a Glance

```
16-bit thermal TIFFs
        │
        ▼
 U-Net breast-region segmentation  ──►  binary masks
        │
        ▼
 TherMAM-NeRF (thermal neural field)  ──►  watertight geometry (STL)  +  per-vertex surface temperature
        │
        ▼
 FEM-FEM inverse bioheat (FEniCSx grid search + Nelder-Mead)  ──►  tumour depth z_t, radius r_t
        │
        ▼
 Lateral pin from IR hot-spot centroid  ──►  tumour lateral (x, y) position
```

The **active** components are: preprocessing + U-Net (masks), TherMAM-NeRF (reconstruction), and the FEM-FEM inverse bioheat (`mukhmetov_recover.py`, `run_cohort.py`). PINN variants (v1, v3, Mukhmetov-PINN) are under "Previous Work" as documented negative results — each entry explains why it was superseded.

---

## Foundational Stages (still used)

### 1. Preprocessing — [watershed_background_removal.py](watershed_background_removal.py)

Removes the background from organized thermograms via watershed segmentation, producing binary masks and optional radiometric masked arrays that preserve thermal values while zeroing the background. All later stages assume the breast region is isolated from room background and imaging artifacts.

### 2. U-Net breast-region segmentation

A **semantic segmentation** U-Net delineates the breast region from single-channel thermal images. Its binary masks are the backbone input for everything downstream: they define the **silhouette contours** the reconstruction consumes, and they isolate the **region of interest**, removing background thermal noise before bioheat analysis. Source notebook: [UNET_Segmentation_newest.ipynb](<UNET_Segmentation/Masking and Segmentation/UNET_Segmentation_newest.ipynb>); trained checkpoints in [UNET_Segmentation/HybridPlan/](UNET_Segmentation/HybridPlan/). Full write-up: [Docs/Stage2_UNet_Segmentation.md](Docs/Stage2_UNet_Segmentation.md).

* **Input / normalisation:** 16-bit calibrated TIFFs resized to $256\times256$ (area interpolation to preserve thermal energy), min–max normalised $m_{i,j} = (P_{i,j}-\min P)/(\max P-\min P)$ so kernels learn morphology invariant to the patient's absolute baseline temperature.
* **Ground truth:** manual polygon annotation (`MANUAL MASKING.ipynb`) with anatomical priors — exclude neck/upper torso, truncate at the lateral fold (no axillary leakage), trace the inframammary fold — rasterised with `cv2.fillPoly`. 162 samples split 78 % train / 22 % test.
* **Augmentation:** horizontal flip ($p=0.5$, applied to image+mask) and brightness jitter $\gamma\sim U[0.8,1.2]$ (image only, to simulate ambient thermal variation).
* **Architecture:** 4-level encoder–decoder U-Net with skip connections (contracting path for semantic context, expanding path for precise localisation), Kaiming/He initialisation.
* **Loss:** hybrid $\mathcal{L} = 0.6\,\mathcal{L}_{\text{BCE}} + 0.4\,\mathcal{L}_{\text{Dice}}$ — BCE for smooth pixel-level gradients, soft Dice for class-imbalance robustness (small breast regions in lateral views).
* **Training:** AdamW ($\eta_0=3\times10^{-4}$, weight decay $10^{-4}$), `ReduceLROnPlateau`, early stop ≈ epoch 40. **Test Dice $0.8935 \pm 0.0542$.**
* **Inference:** sigmoid + threshold 0.5, then keep the largest connected component (a breast is one continuous mass), and Canny edge detection isolates the silhouette boundary. The checkpoint is used **frozen** at runtime by the reconstruction stage to generate masks on demand.

---

## Current Work: TherMAM-NeRF Thermal Neural Field

Reconstruction has progressed from mask-based occupancy (BreastNet3D, see Previous Work) to **TherMAM-NeRF**, a neural radiance-style field that learns *both* breast geometry and a continuous surface-temperature field from the five aligned IR views. Code lives in [TherMAM-NeRF/](TherMAM-NeRF/); the current stable training script is [thermamnerf_v2.9.py](TherMAM-NeRF/thermamnerf_v2.9.py).

* A **Siamese encoder** fuses the five-view IR + mask inputs into per-view features; a **ThermamNeRF MLP** decodes a volumetric field giving, at each point, an occupancy density `sigma` and a temperature `T`.
* Geometry is extracted by **marching cubes** on the `sigma` grid (128³), repaired to a watertight mesh with trimesh, and exported as STL.
* The temperature field is sampled onto the surface vertices to produce a per-vertex **surface IR map** (`T_measured`).

**Why TherMAM-NeRF replaced BreastNet3D for this project:** the bioheat stage needs a per-surface-vertex *temperature*, not just a silhouette. BreastNet3D learns geometry from binary masks only and discards the thermal signal during reconstruction. TherMAM-NeRF jointly learns geometry and temperature, which is exactly the input the inverse-bioheat solver consumes.

**What the downstream solver trusts:** TherMAM-NeRF produces a *volumetric* temperature field, but only its **surface** values are used. A NeRF trained on outside-in IR views cannot observe the breast interior — its interior temperature is an unconstrained extrapolation. The bioheat solver therefore consumes the reliable **surface** output and lets physics determine the interior.

---

## Current Work: FEM-FEM Inverse Bioheat

The goal is a **screening** estimate of a tumour's **position, depth, and size** inside the reconstructed breast, by coupling the NeRF surface temperature to the **Pennes bioheat equation**:

$$ \nabla\cdot(k\nabla T) + \omega_b c_b (T_a - T) + Q_m + Q_{tumour} = 0 $$

The current implementation is [mukhmetov_recover.py](TherMAM-NeRF/mukhmetov_recover.py) (synthetic validation) and [run_cohort.py](TherMAM-NeRF/run_cohort.py) (real cohort). Full write-up: [Docs/Stage4_InverseBioheat.md](Docs/Stage4_InverseBioheat.md).

### Why PINN variants failed (documented negative results)

Three PINN formulations were tried before reaching the current approach:

1. **PINN v1 (free 5-param):** tumour source escapes to skin boundary; $Q_{max}$ frozen at init (unidentifiable from surface — Bezerra et al. 2013).
2. **PINN v3 (forward-matching):** 6-layer Tanh MLP saturates at **7.3 °C BC-fitting RMSE** on synthetic data. Tumour signal is 0.67 °C → SNR 0.09:1. Cost surface flat even on ideal self-consistent data; planted tumour unrecoverable.
3. **Mukhmetov-PINN (Dirichlet-skin):** same Tanh saturation floor; tumour gradient invisible.

All three share the same root cause: a deep Tanh network cannot represent the bioheat field accurately enough for a 0.67 °C tumour perturbation to be detected.

### The FEM-FEM fix

Replace the PINN with **FEniCSx itself** as the forward model inside the optimiser:

| | PINN (all variants) | FEM-FEM |
|---|---|---|
| Forward model noise | 7.3 °C (Tanh saturation) | ~0.001 °C (FEM precision) |
| Tumour signal | 0.67 °C | 0.67 °C |
| SNR | **0.09:1 — undetectable** | **670:1 — trivial** |

### Algorithm

* **Fixed:** lateral $(x_0,y_0)$ from IR hot-spot centroid (top 10 % warmest vertices); magnitude $Q_0(r_t)$ from Gautherie law.
* **Optimised:** $(z_t, r_t)$ only (2 DOF).
* **Step 1:** coarse 5×4 grid = 20 FEM solves; cost = skin-only MSE vs target surface T.
* **Step 2:** Nelder-Mead refine (≤30 iterations) from grid best.
* **Mesh:** generated once from TherMAM-NeRF STL via Gmsh, reused across all evaluations.

### Synthetic validation (Patient_1, in progress)

| Planted z (mm) | r (mm) | Recovered z (mm) | Depth err | Radius err |
|---|---|---|---|---|
| −61.9 | 8 | −62.0 | **0.2 %** | **0.1 %** |
| −61.9 | 14 | −62.0 | **0.1 %** | **0.1 %** |
| 7 more scenarios in progress … | | | | |

Cost reaches machine zero — FEniCSx recovers its own output exactly.

### Honest scope

The synthetic SNR is 670:1 → sub-0.3 % error. On **real DMR-IR data** the model-mismatch floor is 4.5–5.6 °C (real tissue heterogeneity vs homogeneous FEM), larger than the 0.67 °C tumour signal. Real-patient depth/radius estimates cannot be validated without co-registered MRI/CT — the same limitation Mukhmetov et al. (2025) report (18.67 % radius error, depth unverifiable, single real patient). Lateral position from the IR hot-spot centroid and per-patient FEA residuals remain reportable for all 122 patients.

### Running

```bash
conda activate bioheat
cd TherMAM-NeRF
python -u mukhmetov_recover.py --idx 0 | tee mukhmetov_syn.log   # synthetic validation
python -u run_cohort.py --method mukhmetov --all                  # real cohort
```

### Outputs (under `TherMAM-NeRF/PINNpdeSolver/results/`)

* `mukhmetov_recovery.csv` / `mukhmetov_recovery.png` — 9-scenario synthetic table + scatter plot
* `{Patient_ID}/{Patient_ID}_syn.stl` / `.msh` — patient surface + tetrahedral mesh
* `cohort_mukhmetov.csv` — per-patient `x_t_mm, y_t_mm, z_hat_mm, r_hat_mm, cost, fea_residual`

---

## Previous Work

These components shaped the project and are kept for reference, but are **not on the current path**. Each notes why.

### Classical 2.5D geometric reconstruction — [KPE_Current.ipynb](UNET_Segmentation/KPE_Current.ipynb)

A structured geometric pipeline: extract anatomical curves and anchor points from five-view masks, register them in a common coordinate system, and loft a breast surface from ordered geometry. It defines the project's anatomical logic (view interpretation, contour ordering, pivot mapping).

**Why superseded:** contour extraction and keypoint/curve matching are brittle across patients, views, and image quality — small segmentation errors or a mislabeled view break the registration. Learned reconstruction (below) was adopted to remove this fragility.

Phases, for the record:
1. **Segmentation & refinement** — U-Net silhouette + morphological cleanup.
2. **Inframammary fold extraction** — trace and order the lower boundary into a continuous polyline.
3. **Anatomical keypoint extraction** — anchors $P1,P2,P3$. Frontal/oblique: $P2=\arg\min y,\ P1=\arg\max x,\ P3=\arg\min x$; lateral views use the rotated-axis convention.
4. **3D geometric transformation** — rigid rotation about $Y$ relative to pivot $P2$: $x_{rot}=x'\cos\theta,\ z_{rot}=-x'\sin\theta$.
5. **B-spline fitting** — degree-4 smoothing of each lifted curve.
6. **Visualization & lofted surface** — assemble boundary curves into a surface.

### Exploratory representation learning — [CNNVAE_test.py](CNNVAE_test.py)

A convolutional variational autoencoder for thermal images with latent sampling and t-SNE tools.

**Why superseded:** this was a feasibility experiment to see whether thermal images support latent representation learning. It was never a production reconstruction component and led into, rather than being part of, the shape-reconstruction work.

### Learned 3D occupancy reconstruction — BreastNet3D (v4 → v5)

[breastnet3d_v4.ipynb](UNET_Segmentation/3DBreastnet/breastnet3d_v4.ipynb) and the upgraded [breastnet3d_v5.ipynb](UNET_Segmentation/3DBreastnet/breastnet3d_v5.ipynb) encode five-view binary masks into a latent vector, decode a $128^3$ occupancy volume, and render differentiable projections back into the five views for self-supervised training.

* **Differentiable projection:** $Projection(h,w) = 1 - \exp\left(-\sum_d V_{rot}(d,h,w)\right)$ over a rotated volume.
* **Loss:** Dice between projections and target masks, $\mathcal{L}_{Dice}(P,T) = 1 - \frac{2\sum (P\cdot T)}{\sum P^2 + \sum T^2 + \epsilon}$.
* **v5 improvements:** `DoubleConv3D` blocks, gradient checkpointing, strict patient curation (122 complete sets from 137). Reported val Dice ≈ 0.848, HD95 ≈ 12.46.
* It also included a post-hoc **thermal overlay** (Sec 2.2.5): view-angle estimation by Dice minimisation, ray-cast visibility, absolute-temperature sampling from RAW `.tiff`, KNN gap-filling, and Gaussian pre-smoothing before marching cubes.

**Why superseded:** BreastNet3D reconstructs *geometry from silhouettes only* and discards the IR signal during reconstruction; its temperature was a post-hoc projection, not a learned field. The bioheat stage needs a jointly-learned surface temperature, so reconstruction moved to **TherMAM-NeRF**. BreastNet3D remains the reference for the occupancy/projection formulation and the asymmetry features.

### Legacy inverse PINN v1 — [PINN_Pipeline.py](UNET_Segmentation/PINNpdeSolver/PINN_Pipeline.py)

A multi-start inverse PINN that tried to estimate $(x_t, r_t, Q_{max})$ jointly with an Adam→L-BFGS scheme, plus dolfinx FEM verification.

**Why superseded:** learns tumour magnitude $Q_{max}$ as a free parameter — unidentifiable from surface data (Bezerra et al. 2013, confirmed on DMR-IR). Free source also escapes to skin boundary. Replaced by PINN v3.

### Forward-matching PINN v3 — [Thermamnerf_PINN_v3.ipynb](TherMAM-NeRF/Thermamnerf_PINN_v3.ipynb)

PINN used as a *forward* solver inside a depth-radius grid search. Fixed magnitude (Gautherie), lateral pinned to IR hot-spot, Nelder-Mead minimising mean-centred skin pattern mismatch.

**Why superseded:** 6-layer Tanh MLP saturates at 7.3 °C BC-fitting RMSE on synthetic data. Tumour signal 0.67 °C → SNR 0.09:1. Cost surface nearly flat even on self-consistent synthetic data; planted tumour unrecoverable. Replaced by FEM-FEM inverse.

### Mukhmetov-PINN — [run_cohort.py](TherMAM-NeRF/run_cohort.py) `--method mukhmetov` (PINN variant)

Dirichlet skin BC, single Adam optimisation of $(z_t, r_t)$, 70 % tumour-biased collocation resampling. Addressed collocation sparsity but not Tanh saturation. Same 7.3 °C floor, same 0.09:1 SNR. Replaced by FEM-FEM inverse.

---

## Mathematical Notes

### Rotation matrix
Both the geometric and learned pipelines rotate about the $Y$ axis:

$$
R_y(\theta) =
\begin{bmatrix}
\cos\theta & 0 & \sin\theta & 0 \\
0 & 1 & 0 & 0 \\
-\sin\theta & 0 & \cos\theta & 0
\end{bmatrix}
$$

### Asymmetry feature
A left-right asymmetry score from the binary volume (used in the BreastNet3D feature export):

$$
Asymmetry_{LR} = \frac{|V_{left} - V_{right}|}{V_{left} + V_{right} + \epsilon}
$$

### Tumour magnitude vs. size (current bioheat stage)
The Gautherie doubling-time relation fixes the tumour metabolic heat from its size, removing the unidentifiable magnitude DOF: $Q_m\,\tau = C$ with $C = 3.27\times10^6$ W·day/m³, and $D = 0.01\,e^{[0.002134(\tau-50)]}$.

---

## Output Artifacts

From preprocessing & U-Net: binary masks (PNG, mirroring the TIFF hierarchy), masked radiometric arrays, dataset distribution plots, U-Net checkpoints in `UNET_Segmentation/HybridPlan/`.

From BreastNet3D (previous): `3dbreastnet_best.pth`, `training_history.png`, per-patient `.npy` volumes, angle `.json`, projection comparison images, asymmetry feature CSVs.

From TherMAM-NeRF + PINN v3 (current): see the "Outputs" list under PINN v3 above — watertight STL/MSH, trained weights, the $J(\text{depth},\text{radius})$ cost surface, interactive HTML viewers, and per-patient position/depth/size.

---

## Practical Notes for Future Contributors

Zifa and Jessie: the main things to preserve are the **view convention** and **patient grouping** logic — most reconstruction errors come from mislabeled views, incomplete patient sets, or changing the angle sign convention mid-workflow.

* Keep the five-view ordering fixed as RL, RO, F, LO, LL unless you update every dependent stage.
* Keep the angle convention consistent with the repository notes and paper-aligned registration.
* The U-Net masks are the foundation — if you change the segmentation checkpoint, re-verify that mask generation stays compatible with reconstruction (it is consumed frozen at runtime).
* For the bioheat stage, remember the rule that makes v3 work: **fix the tumour magnitude, estimate only geometry, and never let the source minimise the PDE residual freely** (it escapes to the boundary). If you revisit the inverse, read "Why v3 differs" first.
* The NeRF interior temperature is not trustworthy — only use its surface values downstream.

---

## Suggested Workflow

To run the **current** pipeline end-to-end:

1. Run [watershed_background_removal.py](watershed_background_removal.py) for background-cleaned inputs, and generate breast masks with the frozen U-Net ([UNET_Segmentation_newest.ipynb](<UNET_Segmentation/Masking and Segmentation/UNET_Segmentation_newest.ipynb>)).
2. Train / load the thermal neural field with [thermamnerf_v2.9.py](TherMAM-NeRF/thermamnerf_v2.9.py) to obtain geometry + surface temperature.
3. Run the FEM-FEM inverse: `python -u mukhmetov_recover.py --idx 0` for synthetic validation (9 scenarios, one patient), then `python -u run_cohort.py --method mukhmetov --all` for the real 122-patient cohort.

To understand the **history / methodology** (not the active path):

4. [KPE_Current.ipynb](UNET_Segmentation/KPE_Current.ipynb) — classical geometric reconstruction and the project's anatomical logic.
5. [breastnet3d_v5.ipynb](UNET_Segmentation/3DBreastnet/breastnet3d_v5.ipynb) — learned occupancy reconstruction and asymmetry features.
6. [CNNVAE_test.py](CNNVAE_test.py) — exploratory latent-space experiment.

---

## DMR-IR Dataset

Dataset verification is captured in [datasettest.ipynb](<Previous Works (VAE, legacy stuff, misc)/datasettest.ipynb>): streaming metadata from `SemilleroCV/DMR-IR` via the Hugging Face `datasets` library without downloading the full 5 GB, and programmatically extracting/verifying the `ClassLabel` categories for downstream classification/labeling.

---

## Selected References

* Olzhas Mukhmetov et al. (2025), *Non-Invasive Breast Cancer Detection Using Physics-Informed Neural Networks with Thermal Imaging and 3D Patient-Specific Breast Models* — the closest method to the v3 inverse (fixes magnitude, estimates geometry).
* Olzhas Mukhmetov et al. (2023), *Physics-Informed Neural Network for Fast Prediction of Temperature Distributions in Cancerous Breasts* — PINN as a forward bioheat solver.
* L.A. Bezerra (2013), *Estimation of breast tumor thermal properties using infrared images* — FEM-in-loop inverse + the sensitivity analysis showing magnitude/perfusion are unidentifiable from the surface.
* M. Gautherie (1980/1983), tumour doubling-time vs. metabolic-heat relation used to fix $Q$ from size.
* Arka Prabha Saha (2023), *3D-BreastNet: A Self-supervised Deep Learning Network for Reconstruction of 3D Breast Surface from 2D Thermal Images*.
* Gleidson M. Costa (2023), *Modeling the 3D Breast Surface Using Thermography*.
* Thaweesak Trongtirakul (2023), *Automated tumor segmentation in thermographic breast images*.
* AAT Standard of Breast Thermography; HIKMICRO Pocket Series 2 Datasheet.
* DMR-IR dataset — https://huggingface.co/datasets/SemilleroCV/DMR-IR

---

## Repository Files Worth Reading First

**Current path:**
* [mukhmetov_recover.py](TherMAM-NeRF/mukhmetov_recover.py) — FEM-FEM inverse bioheat, synthetic validation
* [run_cohort.py](TherMAM-NeRF/run_cohort.py) — real patient cohort runner (`--method mukhmetov`)
* [thermamnerf_v2.9.py](TherMAM-NeRF/thermamnerf_v2.9.py) — thermal neural field reconstruction
* [UNET_Segmentation_newest.ipynb](<UNET_Segmentation/Masking and Segmentation/UNET_Segmentation_newest.ipynb>) — breast-region segmentation (foundational)

**History / methodology:**
* [breastnet3d_v5.ipynb](UNET_Segmentation/3DBreastnet/breastnet3d_v5.ipynb) — learned occupancy reconstruction
* [KPE_Current.ipynb](UNET_Segmentation/KPE_Current.ipynb) — classical geometric reconstruction
* [CNNVAE_test.py](CNNVAE_test.py) — exploratory latent-space experiment
* [view_patient_tiffs.py](UNET_Segmentation/3DBreastnet/view_patient_tiffs.py) — CLI for viewing absolute per-patient temperatures
