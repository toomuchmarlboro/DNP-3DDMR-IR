# Stage 4: Physics-Informed Neural Network (PINN) for Inverse Bioheat Solving

## 1. Overview
Stage 4 bridges the anatomical 3D reconstruction from Stage 3 with biophysical modeling. The goal is to solve the **inverse bioheat problem**: given the measured 3D surface temperature distribution of a patient's breast, estimate the internal heat source parameters (location, size, and maximum metabolic heat generation) that produced it. This is achieved using a Physics-Informed Neural Network (PINN) coupled with Finite Element Analysis (FEA) for verification.

## 1.1 Theoretical Background

### The Physiological Basis of Breast Thermography
The premise of using breast thermography for cancer detection lies in the physiology of tumour growth. Malignant tumours exhibit highly aggressive growth patterns that demand significant nutrient and oxygen supply. This triggers **tumour angiogenesis**—the recruitment and formation of new, chaotic blood vessel networks. Furthermore, the cancer cells operate at a highly elevated metabolic rate compared to normal tissue. The combination of increased blood perfusion and elevated metabolic activity creates a localized internal heat source. As this heat diffuses through the breast tissue (via conduction and convection), it creates an asymmetrical hyperthermic anomaly on the skin surface, which is captured by the infrared camera. The objective of this pipeline is to mathematically isolate and quantify that elevated metabolic heat generation ($Q_{max}$).

### The Inverse Heat Transfer Problem (IHTP)
The goal of this stage is to solve an **Inverse Heat Transfer Problem**. 
- A **Forward Problem** takes a known internal heat source, domain geometry, and boundary conditions to calculate the resulting surface temperature. This is mathematically well-posed.
- An **Inverse Problem** takes the measured surface temperature and attempts to work backward to find the unknown internal heat source that caused it. 

The IHTP is notoriously **ill-posed** (specifically, it lacks continuous dependence on the data). Because heat diffusion heavily smooths and attenuates thermal signals as they travel from the tumour to the skin surface, small amounts of noise in the camera measurements can lead to wildly different estimations of the internal source. 

### Why Physics-Informed Neural Networks (PINNs)?
Traditional methods for solving the IHTP involve iterative Finite Element Analysis (FEA). These methods require generating a new computational mesh for every guess of the tumour location, solving the forward PDE, computing the residual against the surface measurements, and using gradient descent to update the guess. This is computationally expensive, highly sensitive to noise, and prone to failing due to mesh-dependency and complex geometries.

PINNs overcome these limitations by offering a **mesh-free surrogate solver**:
1. **Continuous Function Approximation:** The neural network learns the temperature field $T(x,y,z)$ as a continuous, infinitely differentiable function parameterized by its weights, completely avoiding discrete mesh limitations and artifacts.
2. **Simultaneous Optimization:** Instead of iterative forward solves, the PINN simultaneously learns the temperature field and the unknown tumour parameters by minimizing a joint loss function. It uses PyTorch's automatic differentiation (autograd) to exactly compute the spatial derivatives ($\nabla^2 T$) required by the Pennes bioheat equation without the truncation errors associated with numerical approximation methods.
3. **Regularization through Physics:** Deep neural networks are highly expressive and prone to overfitting noisy measurement data. By embedding the governing physical laws (the PDE) directly into the loss function, the network's hypothesis space is heavily restricted. It can only predict surface temperatures that physically obey the laws of thermodynamics, naturally regularizing the severely ill-posed inverse problem and preventing overfitting to camera noise.

## 2. Mathematical Formulation

### 2.1 Pennes Bioheat Transfer Equation
The steady-state temperature distribution $T(x,y,z)$ within the breast tissue is modeled using the Pennes bioheat equation:

$$ \nabla \cdot (k \nabla T) + \omega_b c_b (T_a - T) + Q_m + Q_{tumor} = 0 $$

Where:
- $k$: Thermal conductivity of tissue ($0.48 \, \text{W/(m}\cdot\text{K)}$)
- $\omega_b$: Blood perfusion rate ($0.0005 \, \text{s}^{-1}$)
- $c_b$: Specific heat of blood ($3600 \, \text{J/(kg}\cdot\text{K)}$)
- $T_a$: Arterial blood temperature ($37.0 \, ^\circ\text{C}$)
- $Q_m$: Basal metabolic heat generation ($450 \, \text{W/m}^3$)
- $Q_{tumor}$: Pathological heat generation from a potential tumour

### 2.2 Tumour Heat Source Modeling
The tumour is modeled as a spherical, Gaussian-distributed heat source parameterised by its center $(x_t, y_t, z_t)$, radius $r_t$, and maximum heat generation $Q_{max}$:

$$ Q_{tumor}(x, y, z) = Q_{max} \cdot \exp\left( - \frac{(x-x_t)^2 + (y-y_t)^2 + (z-z_t)^2}{r_t^2} \right) $$

These five parameters $(x_t, y_t, z_t, r_t, Q_{max})$ are **learnable parameters** optimized by the PINN simultaneously with the network weights.

## 3. PINN Architecture & Loss Functions

### 3.1 Network Structure
The PINN acts as a surrogate model approximating the temperature field $T(x,y,z)$.
- **Input**: 3D spatial coordinates $(x,y,z)$ normalized to $[-1, 1]$.
- **Architecture**: A Multi-Layer Perceptron (MLP) with 6 hidden layers, 256 neurons each, using `Tanh` activation functions (required for smooth second-order derivatives).
- **Output**: Predicted temperature $T_{pred}$ in $^\circ\text{C}$.

### 3.2 Loss Function
The total loss $\mathcal{L}$ combines data alignment (Dirichlet boundary conditions) and physics constraint satisfaction:

$$ \mathcal{L} = \lambda_{data} \mathcal{L}_{data} + \lambda_{pde} \mathcal{L}_{pde} $$

**1. Data Loss ($\mathcal{L}_{data}$):** Enforces agreement between PINN predictions and actual infrared measurements on the skin surface.
$$ \mathcal{L}_{data} = \frac{1}{N_{surf}} \sum_{i=1}^{N_{surf}} W_i \left( T_{pred}(x_i, y_i, z_i) - T_{measured}(x_i, y_i, z_i) \right)^2 $$
Where $W_i$ is a confidence weight derived from the camera projection angles.

**2. PDE Loss ($\mathcal{L}_{pde}$):** Evaluated at collocation points sampled inside the breast volume, enforcing the Pennes equation.
$$ \mathcal{L}_{pde} = \frac{1}{N_{int}} \sum_{j=1}^{N_{int}} \left( k \nabla^2 T_{pred, j} + \omega_b c_b (T_a - T_{pred, j}) + Q_m + Q_{tumor, j} \right)^2 $$
*Note on scaling:* Because the network inputs are normalized to $[-1,1]$, but the PDE requires physical dimensions, the Laplacian $\nabla^2 T$ is scaled by the inverse square of the coordinate scaling factor ($10^{-3}$ to convert mm to meters).

**3. Loss Normalization ($\lambda_{pde}$):**
The two losses exist on vastly different numerical scales (e.g., initial $\mathcal{L}_{data} \approx 800$, initial $\mathcal{L}_{pde} \approx 33,000,000$). Without normalization, the network would exclusively optimize the PDE by flattening the temperature field, completely ignoring the measured surface data. 

To balance these, a **static loss weighting** is calculated once at step 0:
$$ \lambda_{pde} = \left. \frac{\mathcal{L}_{data}}{\mathcal{L}_{pde} + \epsilon} \right|_{\text{step } 0} $$
*Why static instead of adaptive?* If $\lambda_{pde}$ is allowed to continuously update (e.g., rebalancing every 1,000 steps), as the network learns the physics and $\mathcal{L}_{pde}$ converges to a very small number, the adaptive $\lambda_{pde}$ multiplier spikes to an extremely large value. This massive coefficient causes the gradients to explode, instantly destabilizing the network, destroying data alignment, and driving the loss to NaNs. Holding the coefficient static from step 0 ensures perfectly stable, predictable convergence.

## 4. Data Preparation & Geometry Pipeline

Before the PINN can train, patient-specific geometry and thermal data must be extracted:
1. **Surface Extraction**: The 3D segmentation volume is smoothed and converted to a mesh using the Marching Cubes algorithm.
2. **Anatomical Registration**: The mesh is registered to a standard anatomical frame using the Inframammary Fold (IMF) and Superior Apex to ensure consistent spatial orientation.
3. **Thermal Overlay**: Absolute temperature values ($^\circ\text{C}$) from the 5 thermal views are projected onto the 3D surface using a Lambertian-weighted fusion algorithm.
4. **Interior Point Sampling**: 5,000 collocation points are uniformly sampled inside the convex hull of the breast mesh for evaluating the PDE loss.

## 5. Training Strategy (Optimization)

Solving the inverse problem is highly non-convex. The pipeline employs several strategies to ensure robust convergence:

### 5.1 The Dimensional Mismatch and Physical Space Optimization
A critical design feature is that the tumour parameters $(x_t, y_t, z_t, r_t)$ are optimized strictly in **physical space (millimeters)**, while the neural network weights are optimized over **normalized coordinates** $[-1, 1]$. 

**The Vanishing Gradient Problem:**
If $Q_{tumor}$ were evaluated using normalized coordinates $(x_{norm}, y_{norm}, z_{norm})$, a severe dimensional mismatch would occur. The normalized distance squared $d_{norm}^2$ between a point and the tumour center is strictly bounded (e.g., maximum distance squared $\sim 4.0$). If the tumour radius $r_t$ is initialized in physical millimeters (e.g., $10.0$ mm), the exponent in the heat source function becomes:
$$ \exp\left( - \frac{d_{norm}^2}{r_t^2} \right) \approx \exp\left( - \frac{\sim 4.0}{100.0} \right) \approx \exp(-0.04) \approx 0.96 $$
Because this exponential term is nearly constant ($1.0$) everywhere in the breast volume, the heat source is mathematically flat. A flat heat source provides **zero spatial gradient** ($\frac{\partial Q}{\partial x_t} \approx 0$). Without a spatial gradient, the optimizer cannot move the tumour coordinates, leaving them entirely frozen and unable to localize the source of the heat anomaly.

**The Solution:**
To resolve this, $Q_{tumor}$ is explicitly evaluated using un-normalized physical coordinates (in mm) sampled from the interior points. 
- The initial tumour center is randomly seeded by selecting an exact $(x, y, z)$ coordinate from the physical interior collocation points, guaranteeing it starts inside the breast geometry.
- By evaluating the distance $d_{physical}^2$ in mm, it correctly matches the mm scale of $r_t^2$. This restores the sharp Gaussian profile of the heat source, providing strong, non-vanishing spatial gradients that allow the Adam optimizer to actively pull the tumour coordinates toward the surface temperature anomalies.

### 5.2 Multi-Start Initialization
To avoid local minima (e.g., the network placing a weak tumour in the wrong quadrant), the PINN is trained from scratch $N=5$ times with random initial tumour positions. Only the run that yields the lowest final $\mathcal{L}_{data}$ is kept.

### 5.3 Two-Phase Optimization Architecture
Solving the non-convex PINN landscape requires navigating steep PDE gradients while fitting noisy surface data. This is achieved via a two-phase optimizer:

1. **Phase 1: Adam Optimizer (Global Exploration)** 
   - **Purpose:** Rapidly explore the parameter space, locate the general tumour position, and approximate the background temperature field.
   - **Duration:** Typically 5,000 to 10,000 steps.
   - **Differential Learning Rates:** The tumour parameters ($x_t, y_t, z_t, r_t, Q_{max}$) are assigned a learning rate of $10^{-2}$, which is $10\times$ higher than the neural network weights ($10^{-3}$). The PDE loss gradients flowing to the tumour are often much smaller than the data loss gradients; the higher learning rate ensures the tumour parameters remain highly mobile and do not get trapped by early network predictions.

2. **Phase 2: L-BFGS Optimizer (Local Fine-Tuning)** 
   - **Purpose:** A quasi-Newton second-order optimizer that excels at finding exact local minima. It strictly enforces the physical PDE constraints to smooth out the final temperature field and precisely dial in the $Q_{max}$ value.
   - **Duration:** 200 to 500 steps.
   - **Stability Guards:** Because L-BFGS utilizes a Hessian approximation, it can take massive, explosive steps if the PDE landscape is very steep. To prevent this, the code implements:
     - **Gradient Clipping:** $\|\nabla\|_{\max} = 1.0$ applied directly within the L-BFGS closure to bound step sizes.
     - **Adam Checkpointing (NaN Guard):** The state of the network at the exact end of the Adam phase is saved. If L-BFGS diverges, produces NaNs, or results in a loss $>10\times$ worse than Adam, the pipeline automatically aborts the L-BFGS result and reverts back to the safe Adam checkpoint.

## 6. FEA Forward Verification

Once the PINN estimates $(x_t, y_t, z_t, r_t, Q_{max})$, we must verify if these parameters actually produce the observed surface temperatures when plugged into a rigorous forward solver.
- The voxel mesh is converted to a tetrahedral mesh using `gmsh`.
- `dolfinx` (FEniCSx) solves the forward Pennes bioheat equation over the 3D domain.
- **Verification Metric**: The FEA-simulated surface temperatures are compared against the measured camera temperatures. A mean residual error $< 1.5^\circ\text{C}$ confirms that the PINN has found a valid biophysical solution.

## 7. Pipeline Inputs & Outputs

### 7.1 Input Tensors per Patient
| Tensor | Shape | Description |
|---|---|---|
| `surface_pts` | $[N_s, 3]$ | 3D coordinates of mesh vertices (mm) |
| `T_measured` | $[N_s]$ | Absolute temperature at each vertex ($^\circ\text{C}$) |
| `confidence` | $[N_s]$ | Visibility weights from projection |
| `interior_pts` | $[N_v, 3]$ | Random collocation points inside the breast (mm) |

### 7.2 Biophysical Constants
| Constant | Value | Unit |
|---|---|---|
| $K_{tissue}$ | $0.48$ | $\text{W/(m}\cdot\text{K)}$ |
| $\omega_b$ | $0.0005$ | $\text{s}^{-1}$ |
| $c_{blood}$ | $3600.0$ | $\text{J/(kg}\cdot\text{K)}$ |
| $T_{arterial}$ | $37.0$ | $^\circ\text{C}$ |
| $Q_{metab}$ | $450.0$ | $\text{W/m}^3$ |

### 7.3 Output Results
The pipeline ultimately produces a CSV containing the estimated parameters for each patient:
- `Q_max`: Peak metabolic heat generation (indicative of malignancy).
- `r_t_mm`: Estimated tumour radius.
- `volume_mm3`: Calculated tumour volume.
- `quadrant`: Anatomical quadrant of the tumour (e.g., Upper Outer).
- `fea_mean_residual`: Quality score of the inverse solution.
