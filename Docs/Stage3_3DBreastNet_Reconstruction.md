# Stage 3 — 3DBreastNet: Learned Multi-View Voxel Reconstruction

**Source notebook:** `UNET_Segmentation/3DBreastnet/breastnet3d_v5.ipynb`

---

## 1. Objective

This stage learns a **direct mapping from five 2D thermal views to a 128³ voxel occupancy grid** representing the 3D breast surface. Rather than relying on classical hand-crafted geometric algorithms (e.g., visual hull intersections or NURBS fitting), it employs a deep encoder–decoder architecture. This network implicitly learns the Bayesian shape prior $P(\mathcal{V} | S_0, \dots, S_4)$ mapping the observed 2D silhouettes to a 3D volumetric field.

The reconstructed 3D volume $\mathbf{V} \in [0,1]^{D \times H \times W}$ serves as:

1. The **anatomical domain** over which the PINN bioheat solver models the Pennes Bioheat Transfer Equation (Stage 4).
2. The geometric substrate for **thermal texture mapping** — projecting calibrated temperature data back onto the reconstructed surface.

---

## 2. Pipeline Overview and Voxel Representation

The network predicts a discretized scalar field $\mathbf{V}$. Each voxel value $v_{x,y,z} \in [0,1]$ represents the probability of that spatial location being occupied by breast tissue.

```
  ┌──────────────┐
  │ 5 Thermal     │
  │ Views (TIFF)  │
  └──────┬───────┘
         │ U-Net (frozen, Stage 2 weights)
         ▼
  ┌──────────────┐
  │ 5 Binary      │
  │ Masks (256²)  │
  └──────┬───────┘
         │ Resize to 128×128, stack → (B, 5, 128, 128)
         ▼
  ┌──────────────┐
  │ Encoder 2D    │  5×128×128 → 1000-d latent
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ Decoder 3D    │  1000-d → 1×128×128×128
  └──────┬───────┘
         │ Differentiable projection (render_projection)
         ▼
  ┌──────────────┐
  │ 5 Projected   │  Silhouettes compared against GT masks
  │ Silhouettes   │  via multi-view Dice loss
  └──────────────┘
```

---

## 3. Model Architecture

The model is a deterministic function $f_\theta: \mathbb{R}^{5 \times H' \times W'} \to [0,1]^{D \times H \times W}$.

### 3.1 Encoder 2D

The encoder acts as an information bottleneck, compressing the spatial relationships of the 5 views into a 1000-dimensional latent vector $\mathbf{z} \in \mathbb{R}^{1000}$. This forces the network to learn a compact, low-dimensional manifold of breast shapes.

### 3.2 Decoder 3D

The decoder expands the latent vector $\mathbf{z}$ back into a full voxel grid using 3D transposed convolutions. 

To encourage the network to carve away empty space rather than building from a solid block, the final convolution's bias $b_f$ is initialised to $-4.0$. Given the sigmoid activation $\sigma(x) = \frac{1}{1 + e^{-x}}$, the initial voxel probabilities prior to learning are $v \approx \sigma(-4.0) \approx 0.018$, representing a near-empty volume.

---

## 4. Differentiable Projection — The Volumetric Rendering Equation

Because we lack ground-truth 3D MRI/CT scans for these patients, the network is trained via **unsupervised 3D discovery**. Supervision is derived entirely from the 2D silhouettes $S_k$ via a differentiable renderer $\mathcal{R}$.

### 4.1 Rigid Affine Transformation

Given a viewing angle $\theta_k$, the predicted volume $\mathbf{V}$ is rotated into the camera's frame of reference. The rotation matrix about the vertical $Y$-axis is:

$$
\mathbf{R}_{\theta_k} = \begin{pmatrix} \cos\theta_k & 0 & \sin\theta_k & 0 \\ 0 & 1 & 0 & 0 \\ -\sin\theta_k & 0 & \cos\theta_k & 0 \end{pmatrix}
$$

The rotated volume $\mathbf{V}^{(k)}$ is interpolated using a continuous grid sampling operator:
$$ \mathbf{V}^{(k)}(\mathbf{p}) = \mathbf{V}(\mathbf{R}_{\theta_k}^{-1} \mathbf{p}) $$

### 4.2 Emission–Absorption Ray Integration

To project the 3D density field back to a 2D silhouette, we approximate the Beer-Lambert law of optical absorption. A ray $r_{i,j}$ originating from pixel $(i,j)$ traverses the volume along the depth axis $z$. The probability $O_{i,j}$ that the ray hits tissue (i.e., the pixel belongs to the silhouette) is computed as the complement of the probability that the ray passes completely through empty space:

$$
O_{i,j}(\theta_k) = 1 - \exp\left(-\sum_{z=1}^{D} \mathbf{V}^{(k)}_{i,j,z} \Delta z\right)
$$

Because exponential and summation functions are trivially differentiable, gradients $\frac{\partial \mathcal{L}}{\partial \mathbf{V}}$ can backpropagate from the 2D loss space into the 3D voxel space.

---

## 5. Loss Function Topology

### 5.1 Multi-View Dice Loss

The loss forces the projected silhouettes $O(\theta_k)$ to match the ground-truth U-Net masks $M_k$:

$$
\mathcal{L}(\theta) = \frac{1}{5} \sum_{k=0}^{4} \mathcal{L}_{\text{Dice\_Squared}}\!\left(O(\theta_k),\; M_k\right)
$$

### 5.2 Squared Dice Formulation

To enforce sharper decision boundaries in the 3D volume, the standard Dice loss is modified to use squared terms:

$$
\mathcal{L}_{\text{Dice\_Squared}}(P, T) = 1 - \frac{2 \sum_{i} P_i^2 \cdot T_i^2 + \varepsilon}{\sum_i P_i^2 + \sum_i T_i^2 + \varepsilon}
$$

Squaring the predictions $P_i \in [0,1]$ heavily penalises uncertain, "fuzzy" voxels (e.g., $0.5^2 = 0.25$), forcing the network to commit to high-confidence occupancy predictions near $0$ or $1$.

---

## 6. Training Protocol

| Parameter | Value |
|---|---|
| Optimiser | AdamW ($\eta_0 = 10^{-4}$, weight decay $= 10^{-4}$) |
| Batch size | 2 (Limited by 3D tensor memory) |
| Mixed precision | Enabled (AMP) except for $\exp(\cdot)$ |
| Gradient checkpointing | Decoder stages 4–6 |

Best validation Dice achieved: **0.8427** at epoch 58.

---

## 7. Isosurface Extraction and Lambertian Thermal Texturing

After predicting the final occupancy grid $\mathbf{V}$, the 3D continuous surface boundary $\partial \Omega$ is extracted at the isovalue $\tau = 0.5$ using the **Marching Cubes** algorithm. 

To map the raw thermal intensities from the 2D images back onto the 3D mesh vertices, we implement a **Lambertian-weighted multi-view blending** scheme. 

Let $\mathbf{v} \in \mathbb{R}^3$ be a vertex with unit normal $\hat{\mathbf{n}}_\mathbf{v}$. Let $\hat{\mathbf{d}}_k$ be the unit direction vector to camera $k$. The assigned temperature $T(\mathbf{v})$ is a weighted sum of the sampled 2D temperatures $I_k(\pi_k(\mathbf{v}))$:

$$
T(\mathbf{v}) = \frac{\sum_{k=0}^4 w_k(\mathbf{v}) \cdot I_k\!\left(\pi_k(\mathbf{v})\right)}{\sum_{k=0}^4 w_k(\mathbf{v})}
$$

where $\pi_k$ is the camera projection function. The weight $w_k(\mathbf{v})$ applies Lambert's cosine law, prioritising cameras that view the surface head-on (orthogonal to the normal) while ignoring occluded surfaces:

$$
w_k(\mathbf{v}) = \max(0,\; \hat{\mathbf{n}}_\mathbf{v} \cdot \hat{\mathbf{d}}_k)
$$

This physically-grounded blending mathematically guarantees a smooth, continuous thermal scalar field across the 3D manifold, free of projection seams.
