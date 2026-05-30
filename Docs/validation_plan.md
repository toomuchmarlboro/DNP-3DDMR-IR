# Validation Plan: End-to-End Pipeline for High-Resolution 3D Thermal Breast Reconstruction

This document outlines the rigorous validation framework required to justify the synthesis of a 128³ 3D anatomical reconstruction and thermal mapping pipeline from multi-view 2D thermography, specifically operating on the DMR-IR dataset. Since clinical 3D ground truth (e.g., MRI/CT) is unavailable, validation relies on comprehensive 2D metric expansion, multi-view geometric reprojection, and photometric consistency checks.

## 1. Stage 2 Validation: U-Net Segmentation

**Objective:** Expand evaluation beyond simple average Dice scores to prove robustness across varying anatomies and camera angles.

### 1.1 Metrics Expansion
Run the full hold-out test set (36 samples) and compute the following for each prediction against the ground-truth mask:
*   **Dice Similarity Coefficient (DSC):** (Current baseline: ~0.89) Measures area overlap.
*   **Intersection over Union (Jaccard Index):** Provides a stricter penalty for false positives/negatives.
*   **Hausdorff Distance (HD95):** Measures the maximum boundary deviation between prediction and ground truth. A low HD95 is critical to prove the model accurately traces the Inframammary Fold (IMF) and axillary bounds.

### 1.2 Stratified Performance Analysis
Do not report only a single global average. Break down the metrics to demonstrate a deeper understanding of model behavior:
*   **Per-View Stratification:** Compare Frontal ($0^\circ$) vs. Lateral ($\pm 90^\circ$) performance. Lateral views are typically harder due to armpit occlusion. Quantifying this degradation proves an understanding of the data geometry.
*   **Clinical Stratification:** Compare Benign vs. Malignant subsets. Does the presence of a tumor alter morphology enough to degrade segmentation performance?

---

## 2. Stage 3 Validation: 3D Reconstruction (Geometric Validation)

**Objective:** Prove that the 3D voxel grid ($128^3$) is a mathematically faithful and tight bound (Visual Hull) of the 5 canonical camera views.

### 2.1 Multi-View Reprojection Error
1.  **Render Shadows:** Take the final predicted 3D volume. Using the differentiable volumetric renderer, project it back into 2D silhouettes at the canonical angles ($0^\circ, \pm 45^\circ, \pm 90^\circ$).
2.  **Compare to Input:** Calculate the Dice, Jaccard, and HD95 metrics between these reprojected shadows and the Stage 2 U-Net output masks.
3.  **Success Criteria:** High Dice/low HD95 scores confirm that the 3D geometry perfectly explains the 2D observations, fulfilling the Visual Hull requirement without needing 3D MRI scans.

---

## 3. Stage 3 Validation: Thermal Overlay (Photometric Validation)

**Objective:** Prove that the ray-casting and KNN interpolation algorithms map raw 2D temperatures onto the 3D surface without distortion or physical discontinuity.

### 3.1 Back-Projection Mean Absolute Error (MAE)
1.  Take the fully textured 3D mesh.
2.  Project the temperatures back onto a 2D plane at a specific canonical angle (e.g., Frontal $0^\circ$).
3.  Calculate the pixel-wise MAE between this 2D thermal projection and the original raw 16-bit `.tiff` image.
4.  **Success Criteria:** A low MAE mathematically proves the mapping algorithm preserved the original thermal intensities.

### 3.2 Physiological Plausibility
Validate that 100% of the surface voxels on the 3D mesh fall within the biologically possible range for skin thermography ($28^\circ\text{C}$ to $38^\circ\text{C}$). Values outside this range indicate a failure in the mapping logic.

### 3.3 Seam & Gradient Continuity
Compute the variance of the temperature gradient across the 3D surface. A successful K-Nearest Neighbor (KNN) interpolation and Lambertian blending will result in continuous thermal fields. Spikes in gradient variance (visible "seams") indicate a failure to properly merge overlapping camera views.
