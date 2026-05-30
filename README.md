# 3D Breast Thermography Reconstruction

This repository documents a multi-stage research pipeline for reconstructing breast shape information from multi-view thermographic images. The project has evolved through several phases: background removal, U-Net segmentation, 2.5D geometric reconstruction, exploratory representation learning, and the current BreastNet3D notebook that learns a 3D occupancy volume from five aligned views.

This README is written as a handoff document for future contributors, especially Zifa and Jessie. It explains what each stage does, why it exists, and how the pieces connect.

The project is guided by the canonical five-view acquisition protocol:

* Frontal: 0°
* Right oblique: +45°
* Left oblique: -45°
* Right lateral: +90°
* Left lateral: -90°

The sign convention above is the one used throughout the repository and matches the paper-aligned registration notes in the reconstruction code.

---

## Project Summary

The repository contains two broad families of work.

The first family is the earlier geometric pipeline captured in [KPE_Current.ipynb](UNET_Segmentation/KPE_Current.ipynb). That notebook focuses on contour extraction, keypoint detection, rigid registration, and lofting a 3D breast surface from ordered 2D curves. It follows the methodology inspired by Costa et al. and implements the project’s 2.5D geometric interpretation of the thermographic views.

The second family is the newer learned reconstruction path, centered on [breastnet3d_v4.ipynb](UNET_Segmentation/3DBreastnet/breastnet3d_v4.ipynb). This final version uses five-view binary masks as input to an encoder-decoder model, predicts a 128x128x128 3D occupancy volume, and then renders differentiable projections back into the five canonical views for self-supervised training.

In practical terms, the repository now covers three levels of work:

* preprocessing and masking
* classical geometric reconstruction
* learned 3D reconstruction and export

---

## Repository Evolution

### 1. Preprocessing and mask generation

The script [watershed_background_removal.py](watershed_background_removal.py) removes the background from organized thermograms using watershed segmentation. It produces binary masks and, optionally, radiometric masked arrays that preserve the original thermal values while setting the background to zero.

This step matters because all later stages assume that the breast region has already been isolated from the room background and imaging artifacts.

### 2. Exploratory representation learning

The file [CNNVAE_test.py](CNNVAE_test.py) is an exploratory experiment rather than a final production component. It implements a convolutional variational autoencoder for thermal images and includes feature-space analysis tools such as latent sampling and t-SNE visualization.

Its role in the repository is historical and methodological. It demonstrates that the project first explored whether thermal images could support latent representation learning before moving into shape reconstruction.

### 3. Classical 2.5D geometric reconstruction

The notebook [KPE_Current.ipynb](UNET_Segmentation/KPE_Current.ipynb) is the main record of the earlier geometric pipeline. It extracts anatomical curves and anchor points from five-view masks, aligns them in a consistent coordinate system, and reconstructs a breast surface from the ordered geometry.

This notebook is important because it defines the project’s anatomical logic: how the five views are interpreted, how the contour points are ordered, and how the pivot points are mapped across views.

### 4. Learned 3D reconstruction

The current BreastNet3D notebooks (specifically `breastnet3d_v4.ipynb`) extend the project into a robust learned reconstruction formulation. Instead of lofting a surface directly from curve geometry, the model learns a latent representation from the 5-view masks and decodes it into a 3D occupancy volume. The training pipeline resolves FP16 gradient overflow via forced FP32 rendering and achieves high geometric accuracy (~0.84 Dice score).

This is the current working direction of the project in the repository.

---

## Previous Work: KPE_Current.ipynb

The earlier notebook [KPE_Current.ipynb](UNET_Segmentation/KPE_Current.ipynb) is best understood as a structured geometric pipeline. It contains the following phases.

### Phase 1 and Phase 2: segmentation and refinement

The raw thermal input is segmented with a U-Net. The output is a binary silhouette of the breast region. Morphological cleanup is used to reduce noise and stabilize the contour.

If the input intensity image is denoted by $I(x,y)$ and the normalized output by $I_{norm}(x,y)$, the project uses the usual min-max scaling:

$$
I_{norm}(x,y) = \frac{I(x,y) - I_{min}}{I_{max} - I_{min} + \epsilon}
$$

### Phase 3: inframammary fold extraction

The goal is to isolate the lower boundary of the breast contour. The code traces the lower edge, removes the lateral vertical segments, and then orders the remaining points into a continuous polyline. That ordering step is important because later interpolation assumes an actual curve, not an unordered point cloud.

### Phase 4: anatomical keypoint extraction

The notebook extracts the anchor points $P1$, $P2$, and $P3$ used to register the contours across views.

For frontal and oblique views, the interpretation is:

$$
P2 = \arg\min_{(x,y)} y, \quad P1 = \arg\max_{(x,y)} x, \quad P3 = \arg\min_{(x,y)} x
$$

For lateral views, the convention changes because the image is effectively rotated relative to the body axis. The notebook keeps the paper-aligned interpretation of the sign convention and view-specific anchor assignment.

### Phase 5: 3D geometric transformation

The ordered 2D curves are lifted into a common 3D coordinate system using a rigid rotation around the $Y$ axis.

For a contour point $(x_c, y_c)$ and a local pivot $P2$, the transformed coordinates are:

$$
x' = x_c - P2_x, \quad y' = y_c - P2_y
$$

$$
x_{rot} = x' \cos(\theta), \quad y_{rot} = y', \quad z_{rot} = -x' \sin(\theta)
$$

The sign convention used here is important. In the repository notes, right oblique is $+45^\circ$, left oblique is $-45^\circ$, right lateral is $+90^\circ$, and left lateral is $-90^\circ$.

### Phase 6: B-spline fitting

Each lifted curve is smoothed with a degree-4 B-spline. This makes the curve easier to compare, loft, and visualize.

### Phase 7 and Phase 8: visualization and surface construction

The notebook visualizes the result and then constructs a lofted surface from the curve family. In practical terms, this converts a set of anatomical boundary curves into a continuous surface estimate that can be inspected and exported.

---

## Current Work: BreastNet3D (v4)

The current BreastNet3D notebook (`breastnet3d_v4.ipynb`) is the most important file for the next stage of the project. It is a stable, self-supervised learned reconstruction pipeline that starts from five-view masks and predicts a 3D volume.

### Core idea

The model takes the 5-view binary masks, encodes them into a latent vector, decodes that latent vector into a 3D occupancy grid, and compares rendered projections against the observed masks. In other words, it learns a 3D shape that remains consistent with all five 2D silhouettes.

The reconstruction objective is driven by a differentiable projection model:

$$
Projection(h,w) = 1 - \exp\left(-\sum_d V_{rot}(d,h,w)\right)
$$

where $V_{rot}$ is the rotated volume and the sum is taken over the depth dimension.

### Model structure

The current notebook uses the following components:

* a frozen U-Net segmentation model to dynamically obtain masks at runtime
* a patient grouper that selects only complete 5-view patients
* a 2D encoder that maps the 5-channel mask tensor to a 1000-dimensional latent vector
* a 3D decoder that expands the latent code into a $128 \times 128 \times 128$ occupancy volume
* a differentiable visual-hull style renderer for consistency training, strictly executing in `float32` to prevent NaN gradient collapse
* inference routines that recover the best per-view angles and map RAW temperature information onto the reconstructed volume using interactive WebGL (Plotly)

### 3D Temperature Overlay (Sec 2.2.5)

Following the canonical paper methodology, `breastnet3d_v4.ipynb` includes a rigorous implementation for mapping 2D thermal data onto the generated 3D silhouette:
1. **View Angle Estimation**: Minimizes the Dice loss between the 2D silhouette and projections of the 3D volume to find the precise estimated view angle (e.g., searching $\pm 20^\circ$ around base angles).
2. **Ray-Cast Visibility**: Rotates the volume vertices and calculates normal vectors to map only front-facing geometry, avoiding back-projection.
3. **Absolute Thermal Mapping**: Directly samples the absolute temperatures (°C) from the original RAW `.tiff` images (bypassing Min-Max normalized tensors).
4. **Missing Data Interpolation**: Resolves geometric self-occlusion ("black spots") by applying K-Nearest-Neighbor interpolation across the 3D surface.
5. **Smoothing**: Pre-processes the volume via 3D Gaussian smoothing prior to `marching_cubes` to eliminate voxel staircasing artifacts.

### Self-supervised training objective

The notebook uses Dice loss between the projected reconstruction and the input silhouette masks. The Dice loss is written as:

$$
\mathcal{L}_{Dice}(P,T) = 1 - \frac{2\sum (P \cdot T)}{\sum P^2 + \sum T^2 + \epsilon}
$$

This loss is a natural fit for binary silhouettes because it penalizes overlap errors directly and is less sensitive than pixelwise accuracy to foreground-background imbalance.

### Validation metrics

The notebook tracks:

* training loss
* validation loss
* validation Dice score
* validation HD95

The 95th-percentile Hausdorff distance is used because it is more robust than the raw maximum distance and better reflects practical contour mismatch.

### Why this matters

This learned version of the pipeline is a conceptual shift from pure geometric reconstruction to reconstruction constrained by learning. It is especially useful when the contour extraction and curve matching logic become too brittle across patients, views, or image quality.

---

## Data Flow

The current project pipeline can be summarized as follows.

1. Input thermographic views are organized by patient and class.
2. Background is removed or reduced with watershed preprocessing when needed.
3. Binary masks are generated or recovered using the frozen U-Net.
4. The masks are grouped into complete five-view patient sets.
5. The 5-view masks are encoded into a latent vector.
6. The latent vector is decoded into a 3D volume.
7. The volume is rendered back into the five canonical views.
8. The predicted projections are compared with the target masks.
9. The final volume is exported along with asymmetry features and quality checks.

The export stage writes patient-level outputs such as:

* binary 3D volume
* soft occupancy volume
* thermal volume overlay
* estimated view angles
* projection checks
* asymmetry feature tables

---

## Output Artifacts

The repository produces several types of outputs depending on the stage being run.

From preprocessing:

* binary masks
* masked radiometric arrays
* distribution plots of the dataset by view and class

From the geometric pipeline:

* ordered contours
* anchor points
* reconstructed curves
* lofted 3D visualization outputs

From BreastNet3D:

* `3dbreastnet_best.pth`
* `3dbreastnet_last.pth`
* `training_history.png`
* patient-level `.npy` volumes
* angle estimates in `.json`
* projection comparison images
* asymmetry feature CSV files

---

## Mathematical Notes

### Rotation matrix

The learned and geometric pipelines both rely on a rotation around the $Y$ axis:

$$
R_y(\theta) =
\begin{bmatrix}
\cos\theta & 0 & \sin\theta & 0 \\
0 & 1 & 0 & 0 \\
-\sin\theta & 0 & \cos\theta & 0
\end{bmatrix}
$$

### Asymmetry feature

The post-reconstruction feature extraction computes a left-right asymmetry score from the binary volume:

$$
Asymmetry_{LR} = \frac{|V_{left} - V_{right}|}{V_{left} + V_{right} + \epsilon}
$$

This is a simple but useful descriptor for comparing shape imbalance between the two sides of the breast.

---

## Practical Notes for Future Contributors

Zifa and Jessie: if you continue this project, the main thing to preserve is consistency in the view convention and patient grouping logic. Most reconstruction errors in this kind of pipeline come from mislabeled views, incomplete patient sets, or changing the angle sign convention halfway through the workflow.

When extending the code, keep the following in mind:

* Keep the five-view ordering fixed as RL, RO, F, LO, LL unless you update every dependent stage.
* Keep the angle convention consistent with the repository notes and the paper-aligned registration logic.
* Keep the preprocessing output format stable so older notebooks remain reproducible.
* If you change the segmentation checkpoint, confirm that the mask generation remains compatible with the reconstruction notebook.
* If you change the reconstruction volume size, update the encoder, decoder, renderer, and export routines together.

---

## Suggested Workflow

If you are resuming the project from scratch, a reasonable order is:

1. Run [watershed_background_removal.py](watershed_background_removal.py) if you need background-cleaned inputs.
2. Review [KPE_Current.ipynb](UNET_Segmentation/KPE_Current.ipynb) for the geometric logic and historical reconstruction assumptions.
3. Use [breastnet3d_v4.ipynb](UNET_Segmentation/3DBreastnet/breastnet3d_v4.ipynb) for the current learned 3D reconstruction path.
4. Consult [CNNVAE_test.py](CNNVAE_test.py) only if you need the exploratory latent-space experiment.

---

## Recent Updates: BreastNet3D (v5) and Dataset Integration

### BreastNet3D (v5)
The pipeline has been upgraded to its next iteration in [breastnet3d_v5.ipynb](UNET_Segmentation/3DBreastnet/breastnet3d_v5.ipynb), which builds upon the foundation of v4. Key findings and improvements in v5 include:
* **Architecture:** Maintains a 128x128x128 voxel reconstruction derived from a 1000-dimensional latent representation, but implements structural improvements like `DoubleConv3D` blocks and gradient checkpointing in the 3D Decoder for memory-efficient backpropagation.
* **Data Curation:** Explicitly filters incomplete patients (e.g., those with fewer than 5 valid views) to ensure strict alignment. In recent training runs, it isolated 122 complete patient sets from a pool of 137.
* **Validation Performance:** Showcases strong generalization and reconstruction quality, achieving average validation Dice score of approximately 0.8484 and average HD95 of 12.46.

### DMR-IR Dataset Verification
Exploratory dataset work is captured in [datasettest.ipynb](Previous Works (VAE, legacy stuff, misc)/datasettest.ipynb), where we:
* Utilized the Hugging Face `datasets` library to stream dataset metadata from `SemilleroCV/DMR-IR` without downloading the full 5GB contents.
* Programmatically extracted and verified all `ClassLabel` categorical features. This ensures that the dataset’s built-in labels are properly understood and mapped for any future classification or conditional generation tasks in the pipeline.

---

## Physics-Informed Neural Network (PINN) & FEM Bioheat Pipeline

We have fully implemented a clinical-grade, self-supervised pipeline for solving the **Inverse Pennes Bioheat Equation** to dynamically locate and characterize deep-tissue metabolic heat sources (tumors) inside the reconstructed 3D geometries.

The entire workflow is automated in [PINN_Pipeline.py](UNET_Segmentation/PINNpdeSolver/PINN_Pipeline.py) and visualized in [visualize_3d_temperatures.ipynb](UNET_Segmentation/PINNpdeSolver/visualize_3d_temperatures.ipynb).

### 🚀 Key Capabilities:
1. **Multi-Start Inverse PINN Solver:** Estimates tumor coordinates ($\mathbf{x}_t$), thermal radius ($r_t$), and metabolic heat generation ($Q_{max}$) natively in PyTorch. It leverages a hybrid optimization scheme (Adam with adaptive PDE weight re-normalization, followed by fine-tuning with the L-BFGS second-order optimizer).
2. **FEM Forward Verification:** Natively integrates a finite element solver using **dolfinx (FEniCSx)** and **Gmsh** to solve the forward bioheat problem on the patient's solid 3D tetrahedral mesh. It enforces convective air cooling on the outer skin ($h_{conv}=10\text{ W/m}^2\text{K}$, $T_{air}=20^\circ\text{C}$) and a core body temperature Dirichlet boundary condition ($T=37^\circ\text{C}$) at the flat chest wall boundary to ensure biophysical accuracy.
3. **Structured Database Export:** Automatically organizes outputs per-patient inside the `results/` folder:
   * `results/{Patient_ID}/{Patient_ID}.stl` and `.msh`: Registered geometry meshes.
   * `results/{Patient_ID}/{Patient_ID}_pinn.pth`: PyTorch model state weights.
   * `results/{Patient_ID}/{Patient_ID}_loss_convergence.png`: High-resolution training convergence plot.
   * `results/{Patient_ID}/{Patient_ID}_T_measured.npy`, `_surf_pts.npy`, `_T_fea.npy`: Grid arrays.
   * `results/pinn_fea_results.csv`: Master data spreadsheet summarizing parameters and residuals across the cohort of 122 patients.
4. **Interactive 3D Visualizer (`visualize_3d_temperatures.ipynb`):** A PyVista-based notebook that loads the registered tetrahedral meshes and the continuous PINN neural field. It allows users to rotate the breast in 3D and slice it vertically (sagittal plane) to inspect the deep tumor hyperthermia core in clinical dark mode.

### 🏃‍♂️ Running the Pipeline:
Navigate to the directory and run:
```bash
conda activate bioheat
python PINN_Pipeline.py
```

---

## Selected References

This repository is inspired by and should be read alongside the following literature:

* Arka Prabha Saha (2023), 3D-BreastNet: A Self-supervised Deep Learning Network for Reconstruction of 3D Breast Surface from 2D Thermal Images
* Gleidson M. Costa (2023), Modeling the 3D Breast Surface Using Thermography
* L.A. Bezerra (2013), Estimation of breast tumor thermal properties using infrared images
* Thaweesak Trongtirakul (2023), Automated tumor segmentation in thermographic breast images
* AAT Standard of Breast Thermography
* HIKMICRO Pocket Series 2 Datasheet
* DMR-IR dataset https://huggingface.co/datasets/SemilleroCV/DMR-IR

These are the key conceptual references behind the segmentation, masking, geometric contouring, and learned representation steps used in the repository.

---

## Repository Files Worth Reading First

* [README.md](README.md)
* [UNET_Segmentation_newest.ipynb](<UNET_Segmentation/Masking and Segmentation/UNET_Segmentation_newest.ipynb>)
* [KPE_Current.ipynb](UNET_Segmentation/KPE_Current.ipynb)
* [breastnet3d_v5.ipynb](UNET_Segmentation/3DBreastnet/breastnet3d_v5.ipynb) (Upgraded Reconstruction Pipeline)
* [CNNVAE_test.py](CNNVAE_test.py)
* [view_patient_tiffs.py](UNET_Segmentation/3DBreastnet/view_patient_tiffs.py) (CLI utility for viewing absolute temperature data per patient)

This is the best starting point for anyone taking over the project.
