# ThermalNeRF: Mathematical Foundations

## Notation

| Symbol | Meaning |
|---|---|
| $V = 5$ | Number of captured views |
| $\theta_v \in \{-90°, -45°, 0°, 45°, 90°\}$ | Canonical view angles |
| $I_v \in \mathbb{R}^{H \times W}$ | Radiometric TIFF (absolute °C) for view $v$ |
| $M_v \in \{0,1\}^{H \times W}$ | UNet binary segmentation mask for view $v$ |
| $\tilde{I}_v \in [0,1]^{H \times W}$ | Globally min-max normalised thermal image |
| $\mathbf{x} = (x, y, z) \in \mathbb{R}^3$ | 3D query point in normalised world space $[-1,1]^3$ |
| $\sigma(\mathbf{x}) \geq 0$ | Volume density (occupancy) at point $\mathbf{x}$ |
| $T(\mathbf{x}) \in [0,1]$ | Normalised temperature at point $\mathbf{x}$ |

---

## 1. Input Normalisation (Global Normalisation Fix)

Each patient's five thermal images are normalised using **Global Normalisation** across all views. This guarantees that a predicted scalar temperature $T(\mathbf{x}) = 0.8$ corresponds to the exact same absolute temperature regardless of which camera view it is projected into.

$$\tilde{I}_v(u,w) = \frac{I_v(u,w) - T_{\min}}{T_{\max} - T_{\min} + \epsilon}$$

where $T_{\min}$ and $T_{\max}$ are the global minimum and maximum temperatures across **all** pixels in **all** $V$ views for a given patient:
$T_{\min} = \min_{v,u,w} I_v(u,w)$ and $T_{\max} = \max_{v,u,w} I_v(u,w)$,
and $\epsilon = 10^{-6}$ prevents division by zero.

*(Decision Note: Per-view normalisation was explicitly rejected because it creates a mathematical contradiction where a single 3D point is forced to map to different normalised values depending on the view. Global normalisation resolves this multi-view thermal inconsistency.)*

The absolute values $T_{\min}, T_{\max}$ are stored per-patient so the final
3D temperature volume can be restored to °C at inference time:

$$T_{\text{abs}}(\mathbf{x}) = T(\mathbf{x}) \cdot (T_{\max} - T_{\min}) + T_{\min}$$

---

## 2. Siamese Feature Encoder

A shared-weight CNN $\mathcal{E}_\phi$ processes each view independently,
taking the two-channel stack $[\tilde{I}_v, M_v]$ as input:

$$\mathbf{F}_v = \mathcal{E}_\phi\!\left(\left[\tilde{I}_v \,\|\, M_v\right]\right) \in \mathbb{R}^{C \times H \times W}$$

Because $\mathcal{E}_\phi$ is **shared across all $V$ views**, it learns a
single feature space that is view-agnostic. The thermal channel provides
continuous gradient information; the mask channel localises the breast region.
Together they give the encoder both shape and temperature context at every pixel.

---

## 3. pixelNeRF Feature Projection

For a 3D query point $\mathbf{x}$, we need to condition the NeRF MLP on the
actual patient's thermal images. This is done by projecting $\mathbf{x}$ into
each view's image plane and sampling the corresponding feature vector.

### 3.1 Y-Axis Rotation

The imaging protocol rotates the patient about the vertical (Y) axis.
The rotation matrix for view $v$ with angle $\theta_v$ is:

$$R_y(\theta_v) = \begin{pmatrix}
\cos\theta_v & 0 & \sin\theta_v \\
0 & 1 & 0 \\
-\sin\theta_v & 0 & \cos\theta_v
\end{pmatrix}$$

Rotating $\mathbf{x}$ into view $v$'s camera frame:

$$\mathbf{x}^{(v)} = R_y(\theta_v)\,\mathbf{x}$$

The projected 2D pixel coordinate under **orthographic projection** is simply
the $(x, y)$ components of $\mathbf{x}^{(v)}$ (the $z$ component is depth
and is discarded for the projection):

$$\pi_v(\mathbf{x}) = \bigl(x^{(v)},\; y^{(v)}\bigr) \in [-1, 1]^2$$

### 3.2 Bilinear Feature Sampling (Zero-Padding Fix)

The feature vector at $\mathbf{x}$ from view $v$ is obtained by bilinear
interpolation on the feature map:

$$\mathbf{f}_v(\mathbf{x}) = \text{BilinearSample}\!\left(\mathbf{F}_v,\; \pi_v(\mathbf{x}),\; \text{padding}=\text{'zeros'}\right) \in \mathbb{R}^C$$

*(Decision Note: PyTorch's default 'border' padding was replaced with 'zeros'. Border padding caused 3D points outside the camera frustum to clone edge pixels, creating lateral hallucination artifacts. Zero-padding ensures that out-of-bounds queries correctly return zero.)*

### 3.3 Permutation-Invariant Aggregation

To aggregate features across all $V$ views in a way that is independent of
view ordering, we use **mean and variance pooling**:

$$\boldsymbol{\mu}(\mathbf{x}) = \frac{1}{V}\sum_{v=1}^{V} \mathbf{f}_v(\mathbf{x}) \in \mathbb{R}^C$$

$$\boldsymbol{\nu}(\mathbf{x}) = \frac{1}{V}\sum_{v=1}^{V} \left(\mathbf{f}_v(\mathbf{x}) - \boldsymbol{\mu}(\mathbf{x})\right)^2 \in \mathbb{R}^C$$

The aggregated feature is their concatenation:

$$\mathbf{g}(\mathbf{x}) = \bigl[\boldsymbol{\mu}(\mathbf{x}) \,\|\, \boldsymbol{\nu}(\mathbf{x})\bigr] \in \mathbb{R}^{2C}$$

The variance term captures **inter-view disagreement** at each point — high
variance means views are inconsistent, which implicitly signals uncertain geometry.

---

## 4. Positional Encoding with Coarse-to-Fine Schedule

Raw coordinates are poor inputs to MLPs because they bias toward low-frequency
solutions, producing the Gaussian blob artifacts seen in the baseline voxel model.
Fourier positional encoding lifts coordinates into a high-frequency embedding space.

### 4.1 Standard Fourier Encoding

For a coordinate vector $\mathbf{x} \in \mathbb{R}^3$ and $L$ frequency bands:

$$\gamma(\mathbf{x}) = \left[\sin(2^0 \pi \mathbf{x}),\; \cos(2^0 \pi \mathbf{x}),\; \sin(2^1 \pi \mathbf{x}),\; \cos(2^1 \pi \mathbf{x}),\; \ldots,\; \sin(2^{L-1} \pi \mathbf{x}),\; \cos(2^{L-1} \pi \mathbf{x})\right]$$

Output dimensionality: $3 \times 2L$.

### 4.2 Coarse-to-Fine Windowing

High frequencies early in training cause the network to latch onto noise.
A smooth cosine window $w_k(\alpha)$ controls which frequencies are active
as a function of training epoch, parameterised by $\alpha \in [0, L]$:

$$w_k(\alpha) = \begin{cases}
0 & \text{if } \alpha < k \\
\frac{1}{2}\left(1 - \cos(\pi(\alpha - k))\right) & \text{if } 0 \leq \alpha - k < 1 \\
1 & \text{if } \alpha - k \geq 1
\end{cases}$$

The windowed encoding is:

$$\gamma_\alpha(\mathbf{x}) = \left[w_k(\alpha) \cdot \sin(2^k \pi \mathbf{x}),\; w_k(\alpha) \cdot \cos(2^k \pi \mathbf{x})\right]_{k=0}^{L-1}$$

$\alpha$ is linearly ramped from $0 \to L$ over the first $N_{\text{warmup}}$ epochs,
then held constant at $L$ for the remainder of training.

This means the network first fits a smooth low-frequency shape, then progressively
recovers fine boundary detail — directly addressing the blob problem.

---

## 5. ThermalNeRF MLP

The core network $f_\psi$ maps the concatenated positional encoding and
aggregated features to density and temperature:

$$f_\psi : \mathbb{R}^{6L} \times \mathbb{R}^{2C} \;\longrightarrow\; \mathbb{R}^+ \times [0,1]$$

$$(\sigma(\mathbf{x}),\; T(\mathbf{x})) = f_\psi\!\left(\gamma_\alpha(\mathbf{x}),\; \mathbf{g}(\mathbf{x})\right)$$

The MLP has a **skip connection** at the midpoint layer: the input
$[\gamma_\alpha(\mathbf{x}) \| \mathbf{g}(\mathbf{x})]$ is concatenated back
into the activations halfway through, preventing vanishing gradients and
helping the network retain high-frequency positional information in deep layers.

Output activations:

$$\sigma = \text{Softplus}(\mathbf{h}_\sigma) = \log(1 + e^{\mathbf{h}_\sigma}) \geq 0$$

$$T = \text{Sigmoid}(\mathbf{h}_T) = \frac{1}{1 + e^{-\mathbf{h}_T}} \in (0, 1)$$

---

## 6. Differentiable Volume Rendering

To compare the 3D NeRF predictions against the 2D input images, we need a
differentiable rendering operation. For each pixel $(u, w)$ in view $v$,
we cast a ray $\mathbf{r}(t) = \mathbf{o} + t\,\mathbf{d}$ from the camera
through that pixel and integrate along it.

### 6.1 Ray Parameterisation

Under orthographic projection from view angle $\theta_v$, the camera direction
vector is:

$$\mathbf{d}_v = (-\sin\theta_v,\; 0,\; \cos\theta_v)$$

The ray origin for pixel $(u, w)$ mapped to normalised coordinates
$(p_x, p_y) \in [-1, 1]^2$ is:

$$\mathbf{o}_{u,w} = p_x \cdot \mathbf{r}_v + p_y \cdot \hat{\mathbf{y}}$$

where $\mathbf{r}_v = (\cos\theta_v, 0, \sin\theta_v)$ is the camera right
vector and $\hat{\mathbf{y}} = (0, 1, 0)$ is the vertical axis.

### 6.2 Stratified Sampling with Jitter

The ray is sampled at $N$ points using stratified sampling with random jitter
(applied during training only):

$$t_n = \frac{n}{N} + \epsilon_n, \quad \epsilon_n \sim \mathcal{U}\!\left(0, \frac{t_{\text{far}} - t_{\text{near}}}{N}\right)$$

$$\mathbf{x}_n = \mathbf{o} + t_n \cdot \mathbf{d}$$

The jitter acts as a regulariser: it prevents the network from overfitting
to a fixed discrete grid of sample positions.

Step sizes between consecutive samples:

$$\delta_n = t_{n+1} - t_n, \quad \delta_N = 10^{-3}$$

### 6.3 Alpha Compositing via Transmittance

The opacity (alpha) contributed by sample $n$ along the ray is:

$$\alpha_n = 1 - \exp\!\left(-\,k\,\sigma(\mathbf{x}_n)\,\delta_n\right)$$

where $k$ is the **density scale** hyperparameter (sharpness multiplier).
Higher $k$ produces harder boundaries; the paper's visual hull renderer
corresponds to the limit $k \to \infty$.

The transmittance — probability that the ray has not been blocked by any
earlier sample — is:

$$\mathcal{T}_n = \prod_{j=1}^{n-1}(1 - \alpha_j + \epsilon)$$

This is the **exclusive cumulative product**: $\mathcal{T}_1 = 1$ (unobstructed at start).

The weight assigned to sample $n$ is:

$$w_n = \alpha_n \cdot \mathcal{T}_n$$

These weights satisfy $\sum_n w_n \leq 1$ and represent how much each sample
contributes to the final pixel value.

### 6.4 Rendered Outputs (Denominator Gate Fix)

**Rendered occupancy mask** (analogous to the visual hull projection in the paper):

$$\hat{M}(u,w) = \sum_{n=1}^{N} w_n \in [0, 1]$$

**Rendered temperature** (expected temperature under the weight distribution):
To compute the expected temperature, we divide by the accumulated opacity. However, on completely empty background rays, both numerator and denominator approach zero, causing severe numeric instability (purple grid artifacts).

We solve this using an opacity gate $H(\hat{M}(u,w) > 0.05)$:

$$\hat{T}(u,w) = \left( \frac{\displaystyle\sum_{n=1}^{N} w_n \cdot T(\mathbf{x}_n)}{\hat{M}(u,w) + \epsilon} \right) \cdot H(\hat{M}(u,w) > 0.05)$$

*(Decision Note: By gating the temperature output, we force empty background rays to exactly 0.0, completely removing numerical instability from the thermal loss.)*

The transmittance formulation directly solves the **z-axis inflation problem**
of the baseline voxel model: a ray that encounters occupied voxels early along
its path accumulates high weight there and low weight beyond. The network
is therefore penalised for placing density at incorrect depths, not just for
getting the 2D projection wrong.

---

## 7. Loss Function

The total loss has four terms:

$$\mathcal{L} = \lambda_{\text{dice}}\,\mathcal{L}_{\text{dice}} + \lambda_{\text{bg}}\,\mathcal{L}_{\text{bg}} + \lambda_{\text{thermal}}\,\mathcal{L}_{\text{thermal}} + \lambda_{\text{TV}}\,\mathcal{L}_{\text{TV}}$$

### 7.1 Dice Loss (Foreground Geometry)

The Sørensen–Dice coefficient between rendered mask $\hat{M}_v$ and GT mask
$M_v$, averaged over all $V$ views:

$$\mathcal{L}_{\text{dice}} = \frac{1}{V} \sum_{v=1}^{V} \left(1 - \frac{2\,\langle\hat{M}_v,\, M_v\rangle + \epsilon}{\|\hat{M}_v\|^2 + \|M_v\|^2 + \epsilon}\right)$$

This is numerically more stable than cross-entropy for unbalanced binary
images where background pixels dominate (as is the case here — the breast
region is a fraction of the full image).

### 7.2 Background Ray Loss (Background Geometry Constraint)

The Dice Loss provides zero gradient outside the Ground Truth (GT) silhouette, allowing the network to hallucinate opacity in the background. We apply an explicit $L_2$ penalty to any accumulated opacity on rays where the GT mask is exactly 0:

$$\mathcal{L}_{\text{bg}} = \frac{1}{V} \sum_{v=1}^{V} \frac{1}{|\Omega_{\text{bg},v}|} \sum_{(u,w) \in \Omega_{\text{bg},v}} \left(\hat{M}_v(u,w)\right)^2$$

where $\Omega_{\text{bg},v} = \{(u,w) : M_v(u,w) < 0.5\}$.

*(Decision Note: This mathematically forces the network to sweep the background completely clean, eliminating white background residuals and perfectly matching the visual hull constraint without hard-masking.)*

### 7.3 Thermal MSE Loss (Temperature with Epoch Gating)

Mean squared error between rendered temperature $\hat{T}_v$ and normalised
TIFF $\tilde{I}_v$, computed **only over foreground pixels** defined by $M_v$:

$$\mathcal{L}_{\text{thermal}} = \frac{1}{V} \sum_{v=1}^{V} \frac{1}{|\Omega_{\text{fg},v}|} \sum_{(u,w) \in \Omega_{\text{fg},v}} \left(\hat{T}_v(u,w) - \tilde{I}_v(u,w)\right)^2$$

where $\Omega_{\text{fg},v} = \{(u,w) : M_v(u,w) \geq 0.5\}$ is the foreground pixel set for view $v$.

*(Decision Note: **Epoch Gating** is strictly enforced. $\lambda_{\text{thermal}}$ is set to $0.0$ for the first 50 epochs. At epoch 0, geometry is random "fog", and thermal loss produces poisoned gradients by forcing the network to paint temperatures onto empty space. Gating allows the Dice loss to build a solid 3D surface first. At Epoch 51, $\lambda_{\text{thermal}}$ is ramped to $0.01$, painting temperatures onto physically accurate geometry.)*

### 7.4 Total Variation Regularisation (Smoothness)

Applied to the 3D density grid $\sigma$ evaluated on the marching cubes grid
to suppress high-frequency voxel noise:

$$\mathcal{L}_{\text{TV}} = \frac{1}{|\mathcal{G}|}\sum_{\mathbf{x} \in \mathcal{G}} \left(|\sigma(\mathbf{x} + \hat{e}_x) - \sigma(\mathbf{x})| + |\sigma(\mathbf{x} + \hat{e}_y) - \sigma(\mathbf{x})| + |\sigma(\mathbf{x} + \hat{e}_z) - \sigma(\mathbf{x})|\right)$$

This is computed over a discrete grid $\mathcal{G}$ and penalises
the starfish artifacts that arise when the NeRF overfits to a small dataset.

---

## 8. Training Objective Summary

$$\min_{\phi,\,\psi} \;\; \frac{1}{V}\sum_{v=1}^{V} \Bigg[
\underbrace{\lambda_{\text{dice}}\,\mathcal{L}_{\text{dice}}(\hat{M}_v, M_v)}_{\text{foreground shape}} +
\underbrace{\lambda_{\text{bg}}\,\mathcal{L}_{\text{bg}}(\hat{M}_v, M_v)}_{\text{background shape}} +
\underbrace{\lambda_{\text{thermal}}\,\mathcal{L}_{\text{thermal}}(\hat{T}_v, \tilde{I}_v)}_{\text{thermal fidelity}} +
\underbrace{\lambda_{\text{TV}}\,\mathcal{L}_{\text{TV}}(\sigma)}_{\text{smoothness}}
\Bigg]$$

where $\phi$ are the Siamese encoder weights and $\psi$ are the NeRF MLP weights.

The key difference from the 3D-BreastNet paper is that the thermal loss provides
a **continuous-valued supervision signal** at every training step, whereas the
paper's Dice loss on binary masks provides only a binary occupancy signal.
The NeRF transmittance replaces the paper's visual hull renderer, explicitly
modelling depth and preventing z-axis inflation.

---

## 9. 3D Volume Extraction

At inference, the continuous NeRF is evaluated on a uniform $R^3$ grid
$\mathcal{G} = \{-1, -1 + \frac{2}{R}, \ldots, 1\}^3$:

$$\sigma_{\text{grid}}, T_{\text{grid}} = f_\psi\!\left(\gamma_L(\mathbf{x}),\, \mathbf{g}(\mathbf{x})\right), \quad \forall\,\mathbf{x} \in \mathcal{G}$$

A Gaussian smoothing kernel ($\sigma_{\text{kernel}} = 0.8$ voxels) is applied
to $\sigma_{\text{grid}}$ before surface extraction to remove discretisation
staircasing:

$$\tilde{\sigma}_{\text{grid}} = G_\sigma * \sigma_{\text{grid}}$$

The surface mesh is then extracted via the **Marching Cubes** algorithm at
isosurface level $\tau = 0.3$:

$$\mathcal{S} = \text{MarchingCubes}(\tilde{\sigma}_{\text{grid}},\, \tau)$$

Each mesh vertex $\mathbf{v} \in \mathcal{S}$ is coloured by its predicted
temperature, restored to absolute °C:

$$T_{\text{vertex}}(\mathbf{v}) = T_{\text{grid}}(\mathbf{v}) \cdot (T_{\max} - T_{\min}) + T_{\min}$$

---

## 10. Connection to the Pennes Bioheat Equation (Future PINN Phase)

The NeRF density field $\sigma(\mathbf{x})$ and temperature field $T(\mathbf{x})$
together provide the inputs needed for the inverse bioheat problem.

The steady-state **Pennes Bioheat Equation** is:

$$k\,\nabla^2 T(\mathbf{x}) + \dot{Q}_{\text{met}}(\mathbf{x}) + \omega_b c_b \bigl(T_a - T(\mathbf{x})\bigr) = 0$$

| Symbol | Meaning |
|---|---|
| $k$ | Tissue thermal conductivity (W/m·K) |
| $\dot{Q}_{\text{met}}$ | Metabolic heat generation (W/m³) |
| $\omega_b$ | Blood perfusion rate (m³/s·m³) |
| $c_b$ | Specific heat of blood (J/kg·K) |
| $T_a$ | Arterial blood temperature (°C) |

The PINN extends the NeRF by adding the PDE as a physics loss:

$$\mathcal{L}_{\text{PDE}} = \left\| k\,\nabla^2 T(\mathbf{x}) + \dot{Q}_{\text{met}}(\mathbf{x}) + \omega_b c_b(T_a - T(\mathbf{x})) \right\|^2$$

where $\nabla^2 T$ is computed via **automatic differentiation through the MLP**
(no finite differences needed — the MLP is infinitely differentiable).

Because the ThermalNeRF already represents $T(\mathbf{x})$ as a continuous
function, adding the PINN loss is a direct extension: the same MLP, the same
autodiff graph, just an additional loss term.

The baseline voxel overlay cannot support this because a discrete voxel grid
is not differentiable with respect to spatial coordinates.

---

## 11. v2.0–v2.2 Development Progress

This section documents the iterative engineering decisions made during the development of TherMAM-NeRF from v2.0 through v2.2, including the bugs discovered, the mathematical reasoning behind each fix, and the resulting improvements.

### 11.1 Ray Opacity Entropy Loss (v2.1)

**Problem:** After the Dice loss successfully carved the breast geometry, faint semi-transparent "ghost streaks" persisted in the background. These streaks had rendered mask values in the range $(0.05, 0.5)$ — too faint for the Dice loss to penalise, and below the $L_2$ background penalty threshold.

**Solution:** A quadratic opacity penalty that penalises any ray accumulation that is neither fully transparent (0) nor fully opaque (1):

$$\mathcal{L}_{\text{entropy}} = \frac{1}{V} \sum_{v=1}^{V} \frac{1}{HW} \sum_{u,w} \hat{M}_v(u,w) \cdot \left(1 - \hat{M}_v(u,w)\right)$$

This function has the following key properties:
- $\hat{M} = 0 \Rightarrow \mathcal{L}_{\text{entropy}} = 0$ (empty space is not penalised)
- $\hat{M} = 1 \Rightarrow \mathcal{L}_{\text{entropy}} = 0$ (solid tissue is not penalised)
- $\hat{M} = 0.5 \Rightarrow \mathcal{L}_{\text{entropy}} = 0.25$ (maximum penalty for ambiguous opacity)

**Design Decision:** The quadratic form $p(1-p)$ was chosen over the information-theoretic form $-p\log(p)$ because $\log(p)$ has an infinite derivative as $p \to 0$. Since the network initialises with near-zero densities, the $\log$-based entropy produced catastrophic gradients at Epoch 1, instantly collapsing the entire volume to zero (Dice $= 0.000$). The quadratic form has bounded derivatives everywhere, providing a gentle, stable push.

*(Decision Note: $\lambda_{\text{entropy}} = 0.05$ in the final configuration. Higher values (e.g., 2.0) caused pixelation of smooth boundaries.)*

### 11.2 Updated Total Loss (v2.2)

The total loss now has **five** terms:

$$\mathcal{L} = \lambda_{\text{dice}}\,\mathcal{L}_{\text{dice}} + \lambda_{\text{bg}}\,\mathcal{L}_{\text{bg}} + \lambda_{\text{thermal}}\,\mathcal{L}_{\text{thermal}} + \lambda_{\text{TV}}\,\mathcal{L}_{\text{TV}} + \lambda_{\text{entropy}}\,\mathcal{L}_{\text{entropy}}$$

### 11.3 Softplus Collapse and $\lambda_{\text{bg}}$ Ramp-Up Schedule (v2.2)

**Problem:** With $\lambda_{\text{bg}} = 5.0$ applied at full strength from Epoch 1, the network experienced "Softplus Collapse." The 3D volume is initialised with random noise, which fills the entire bounding box with non-zero density. The background penalty sees this massive block and sends a catastrophic gradient that pushes all density weights to $-\infty$ in a single step. Once the Softplus activation saturates at 0, the gradients vanish permanently and the Dice score is locked at $0.000$.

**Solution:** A linear ramp-up schedule over the first 10 epochs:

$$\lambda_{\text{bg}}^{(e)} = \lambda_{\text{bg}} \cdot \min\!\left(1,\; \frac{e}{10}\right)$$

At Epoch 1, $\lambda_{\text{bg}}^{(1)} = 0.3$ (10% of full strength), giving the Dice loss time to sculpt the breast geometry. By Epoch 10, the geometry is established and $\lambda_{\text{bg}}$ reaches full strength to sweep the background clean.

*(Decision Note: A 50-epoch ramp was initially used but was too slow — the "Solid Block Trap" occurred where density remained uniformly high across the entire volume because the background penalty took too long to activate. 10 epochs was the sweet spot.)*

### 11.4 Density Scale Tuning (v2.2)

**Problem:** Setting `density_scale` $k = 50.0$ caused the "Solid Block Trap." With the volume initialised to random noise, a very high $k$ made the initial opacity so dense that the entire 3D bounding box rendered as a solid white block. Combined with the ramped $\lambda_{\text{bg}}$, the optimiser could not carve through the block fast enough, and the Dice score plateaued at $\approx 0.30$ (the overlap ratio of a solid cube with the breast silhouette).

**Solution:** Reverted to $k = 10.0$, which produces soft initial opacity that the Dice loss can easily carve.

### 11.5 TV Loss Decay Schedule (v2.2)

The total variation regularisation is decayed with an exponential schedule, bottoming out at 50% of the initial value:

$$\lambda_{\text{TV}}^{(e)} = \lambda_{\text{TV}} \cdot \max\!\left(0.5,\; e^{-0.015 \cdot e}\right)$$

This provides strong smoothing in early epochs (suppressing noise during the random initialisation phase) and relaxes later to allow sharp boundary detail.

### 11.6 Entropy Loss Ramp-Up (v2.2)

The entropy penalty is linearly ramped over the first 100 epochs:

$$\lambda_{\text{entropy}}^{(e)} = \lambda_{\text{entropy}} \cdot \min\!\left(1,\; \frac{e}{100}\right)$$

This prevents the entropy loss from interfering with geometry formation. It only reaches full strength after the breast shape is well established.

### 11.7 Near-Plane Bug Fix (v2.2)

**Problem:** The rendered breast shape was violently sliced along straight lines in every view. Some patients had portions of their anatomy completely cut off.

**Root Cause:** The ray marching parameter `near` was set to $0.0$ instead of $-1.0$. Because the 3D bounding box spans $[-1, 1]^3$ and camera ray origins are placed at the centre plane, setting $t_{\text{near}} = 0.0$ meant rays could only travel **forward** from the origin, making the camera completely blind to the half-space behind it.

For the Frontal view ($\theta = 0°$, camera looking along $+z$), this meant the entire $z < 0$ region was invisible. For the LL view ($\theta = 90°$), the entire $x > 0$ region was invisible. The network was forced to reconstruct the breast using only half of each camera's field of view, resulting in clean geometric cuts wherever the breast crossed the camera's centre plane.

**Fix:** Changed `near` from $0.0$ to $-1.0$, allowing rays to traverse the full $[-1, 1]$ depth range.

Additionally, the TV loss grid was incorrectly evaluated over $[t_{\text{near}}, t_{\text{far}}]$ instead of the full $[-1, 1]^3$ volume, causing smoothing to be applied unevenly. This was corrected to always span $[-1, 1]$.

---

## 12. v2.2 Final Hyperparameter Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `img_size` | 128 | Balance of resolution and VRAM |
| `feat_channels` | 32 | Encoder output dimensionality |
| `pos_enc_L` | 8 | Frequency bands ($6L = 48$ dim) |
| `mlp_hidden` | 256 | Increased from 192 for greater capacity |
| `mlp_layers` | 4 | With skip connection at layer 2 |
| `n_samples` | 128 | Ray march samples per ray |
| `near` | $-1.0$ | Full bounding box traversal |
| `far` | $1.0$ | Full bounding box traversal |
| `density_scale` ($k$) | 10.0 | Soft enough to avoid solid block trap |
| `freq_warmup_epochs` | 50 | Coarse-to-fine frequency schedule |
| `n_epochs` | 300 | Total training epochs |
| `lr` | $5 \times 10^{-4}$ | Adam learning rate |
| `lambda_dice` | 1.0 | Primary shape supervision |
| `lambda_bg` | 3.0 | Background cleanup (ramped over 10 epochs) |
| `lambda_thermal` | 0.0 | Currently disabled (geometry-only phase) |
| `lambda_tv` | 0.01 | Smoothness (exponential decay, floor at 50%) |
| `lambda_entropy` | 0.05 | Opacity snap (ramped over 100 epochs) |
| `n_rays` | 4096 | Rays sampled per view per batch |
| `mc_threshold` | 0.3 | Marching cubes isosurface level |
| `mc_resolution` | 128 | Volume extraction grid resolution |

---

## 13. Evaluation Metrics

### 13.1 Projection Audit Pipeline

After training, the best checkpoint is loaded and every patient in the dataset is evaluated. For each patient, all 5 views are re-rendered from the learned 3D volume and compared against the ground truth masks.

### 13.2 Dice Score (per-view)

$$\text{DSC}_v = \frac{2 \cdot |\hat{B}_v \cap B_v|}{|\hat{B}_v| + |B_v| + \epsilon}$$

where $\hat{B}_v = \{(u,w) : \hat{M}_v(u,w) > 0.5\}$ is the binarised rendered mask and $B_v = \{(u,w) : M_v(u,w) > 0.5\}$ is the ground truth mask.

### 13.3 Intersection over Union (per-view)

$$\text{IoU}_v = \frac{|\hat{B}_v \cap B_v|}{|\hat{B}_v \cup B_v| + \epsilon}$$

The relationship between the two metrics is:

$$\text{IoU} = \frac{\text{DSC}}{2 - \text{DSC}}$$

IoU is a stricter metric — it penalises both false positives and false negatives more heavily than Dice. A Dice of 0.965 corresponds to an IoU of 0.933.

### 13.4 Cohort-Level Reporting

For each patient $p$, the mean Dice and mean IoU across all 5 views are computed:

$$\overline{\text{DSC}}_p = \frac{1}{V}\sum_{v=1}^{V} \text{DSC}_{v,p}, \qquad \overline{\text{IoU}}_p = \frac{1}{V}\sum_{v=1}^{V} \text{IoU}_{v,p}$$

The overall cohort performance is the grand mean across all patients:

$$\overline{\text{DSC}} = \frac{1}{P}\sum_{p=1}^{P} \overline{\text{DSC}}_p, \qquad \overline{\text{IoU}} = \frac{1}{P}\sum_{p=1}^{P} \overline{\text{IoU}}_p$$

A bar chart (`cohort_performance_plot.png`) is generated showing the per-patient Dice and IoU scores side by side, enabling rapid identification of outlier patients.

---

## 14. Distributed Data Parallel (DDP) Training

The training loop supports multi-GPU training via PyTorch's `DistributedDataParallel`. When launched with `torchrun --nproc_per_node=N`, each GPU processes a disjoint subset of patients via a `DistributedSampler`. Gradients are synchronised across GPUs via NCCL all-reduce after each backward pass. The training loss is averaged across all ranks before logging.

Key implementation details:
- The encoder and MLP are both wrapped in DDP with per-device output.
- Only the main rank (`RANK == 0`) performs validation, checkpointing, and visualisation.
- `train_sampler.set_epoch(epoch)` ensures proper shuffling across epochs.
- Mixed precision (`torch.cuda.amp`) and gradient checkpointing are enabled for VRAM efficiency.

---

## 15. Preliminary Results (v2.2)

| Patient | Mean Dice | Mean IoU |
|---|---|---|
| Patient_121 | 0.965 | 0.933 |

*(Full cohort results pending completion of the near-plane fix run.)*
