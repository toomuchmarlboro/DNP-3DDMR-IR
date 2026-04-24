# 3D Breast Surface Reconstruction from Thermographic Images

This repository contains the implementation of the 3D breast surface reconstruction pipeline based on the methodology proposed in **Costa et al. (2023): "Modeling the 3D Breast Surface Using Thermography"**. 

The goal of this project is to take 2D thermographic images from 5 different views (0°, +45°, -45°, +90°, -90°) and reconstruct a 3D Non-Uniform Rational B-Spline (NURBS) surface representing the patient's breast anatomy.

---

## Pipeline Overview and Mathematical Formulation

The notebook `KPE_Current.ipynb` implements the complete pipeline in 8 phases. Below is a detailed mathematical and programmatic explanation of what happens in each phase.

### Phase 1 & 2: U-Net Segmentation & Refinement
*   **Input**: Raw 2D thermographic images.
*   **Process**: A Convolutional Neural Network (U-Net architecture) is used to segment the breast area from the background. The output is a binary mask. Morphological operations (dilation/erosion) are applied to clean noise and extract the continuous edge of the segmentation.

### Phase 3: Inframammary Fold Extraction
*   **Goal**: Isolate the lower boundary of the breast (the inframammary fold).
*   **Process**: 
    1. For every x-coordinate, find the lowest (max $y$) edge pixel.
    2. Subtract the purely vertical lateral boundaries (the extreme left and right vertical edges of the segmented mask).
    3. The remaining pixels form the "lower single edge."
    4. **Ordering**: An 8-connected neighbor traversal algorithm connects the raw pixels into an ordered, contiguous 2D curve $(x_i, y_i)$. This prevents the contour from being treated as random scatter in later interpolation steps.

### Phase 4: Anatomical Key Point Extraction (P1, P2, P3)
Key points are extracted based on Algorithm 1 (lines 10-20) of the paper to serve as pivot points for 3D alignment.

**For 0° (Frontal), 45° (Right Oblique), and -45° (Left Oblique) views:**
*   **P2 (Center Junction)**: The highest point of the curve mathematically defined as $P2 = \arg\min_{(x,y)} (y)$.
*   **P1 (Right Side)**: The patient's right-side inflection point, which corresponds to the maximum x-coordinate on the image: $P1 = \arg\max_{(x,y)} (x)$.
*   **P3 (Left Side)**: The patient's left-side inflection point, which corresponds to the minimum x-coordinate on the image: $P3 = \arg\min_{(x,y)} (x)$.

**For 90° (Right Lateral) and -90° (Left Lateral) views:**
*   **P2 (Lowest Point)**: $P2 = \arg\max_{(x,y)} (y)$.
*   **P1 (Toward Body Center) & P3 (Toward Arms)**:
    *   For 90°: Body center is on the left ($P1 = \arg\min x$), arms on the right ($P3 = \arg\max x$).
    *   For -90°: Body center is on the right ($P1 = \arg\max x$), arms on the left ($P3 = \arg\min x$).

### Phase 5: 3D Geometric Transformation (Generating the 9 Curves)
This phase projects the 2D edges into a unified 3D coordinate system (X=Horizontal, Y=Vertical Downward, Z=Depth, where $Z \le 0$ extends towards the camera).

#### 1. Frontal Curves (C1, C2, C3)
The 0° frontal view sits at $Z=0$.
*   **C1**: The segment of the ordered 0° contour extending to the right of $P1$ ($x \ge P1_x$).
*   **C2**: The segment of the ordered 0° contour extending to the left of $P3$ ($x \le P3_x$).
*   **C3**: An auxiliary vertical line constructed from $P2$ straight up to the top of the image boundary: $x = P2_x, y \in [y_{top}, P2_y]$.

#### 2. Rotated Side-View Curves (C4 - C9)
For the remaining views (+45°, -90° anchored to $P3_{global}$; -45°, +90° anchored to $P1_{global}$), a rigid geometric transformation is applied. 

For each point $(x_c, y_c)$ in a lateral/oblique contour:
1.  **Translate local $P2$ to origin**: 
    $$x' = x_c - P2_x, \quad y' = y_c - P2_y$$
2.  **Rotate around Y-axis by $\theta$** (where $\theta$ is the view angle):
    The rotation matrix $R_y(\theta)$ is applied:
    $$x_{rot} = x' \cos(\theta)$$
    $$y_{rot} = y'$$
    $$z_{rot} = -x' \sin(\theta)$$
3.  **Translate to Global Pivot**:
    Move the rotated curve to the corresponding anchor point ($P_{pivot} \in \{P1_{0°}, P3_{0°}\}$):
    $$X_f = x_{rot} + P_{pivot, x}$$
    $$Y_f = y_{rot} + P_{pivot, y}$$
    $$Z_f = z_{rot}$$
4.  **Z-Filtering & Contiguous Extraction**: 
    To remove overlapping back-facing segments (the body of the patient), points where $Z_f > 0$ are discarded. To prevent closed loops caused by internal contour gaps, the algorithm extracts the longest contiguous segment of $Z \le 0$ points, yielding an open 3D breast profile.

### Phase 6: B-Spline Curve Fitting
To normalize the point clouds and ensure smoothness, a **Degree-4 B-spline** is fitted to each of the 9 generated curves (C1-C9). 
*   `scipy.interpolate.splprep` computes a parametric representation of the curve.
*   The curve is then resampled using exactly 100 uniformly spaced parameters $u \in [0, 1]$ via `splev`. This ensures all 9 curves have the exact same dimensionality and smooth out pixelation noise.

### Phase 7 & Phase 8: 3D Visualization and NURBS Lofting
*   **Visualization**: `plotly.graph_objects` is used to plot the 9 3D curves interactively.
*   **NURBS Surface Generation**: To create a solid 3D mesh (surface) from the 9 skeleton curves, `scipy.interpolate.griddata` is used. It takes the $(X, Y)$ coordinates of the B-spline control points as inputs and interpolates a dense grid of $Z$ (depth) values, effectively lofting a continuous surface over the skeleton curves.

---

## What to Do Next for the Thesis

The geometric reconstruction pipeline is now mathematically sound and correctly implements Algorithm 1 from the paper. To finalize the system, you should complete the following steps:

1. **UV Texture Mapping (Algorithm 1, Line 42)**
   * Currently, the surface is a solid color mesh. You need to map the original 0° (frontal) raw thermographic image onto this 3D NURBS surface. 
   * **How to do it**: Since the $X, Y$ coordinates of the 3D surface grid perfectly align with the pixel coordinates of the frontal 0° image, you can sample pixel colors from the original 0° thermogram using the $(X, Y)$ coordinates of the surface grid and apply them as textures in the 3D plot.

2. **Refine Surface Lofting Boundaries**
   * The current `griddata` interpolation uses a rectangular convex hull, which sometimes causes artifacting at the extreme edges. You can apply a masking function or transition to a specialized CAD/NURBS library (like `geomdl`) to create a true lofted Non-Uniform Rational B-Spline surface bounded strictly by C1 and C2.

3. **Quantitative Validation**
   * As suggested in the "Future Work" section of Costa et al., visually inspecting the 3D shape isn't enough for clinical validation. 
   * **Action**: Compare the reconstructed point cloud against ground-truth data (e.g., from a 3D structured light scanner) using metrics like Root Mean Square Error (RMSE) or Hausdorff Distance.

4. **Batch Processing Automation**
   * Convert the pipeline inside the notebook into a Python class/script so it can iterate over the entire dataset (e.g., "Patient_13", "Patient_14", benign vs. malignant) and automatically save the 3D models (as `.obj` or `.stl` files) for all patients without manual intervention.
