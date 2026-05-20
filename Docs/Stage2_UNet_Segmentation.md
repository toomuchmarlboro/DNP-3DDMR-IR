# Stage 2 — U-Net Breast Region Segmentation

**Source notebook:** `UNET_Segmentation/Masking and Segmentation/UNET_Segmentation_newest.ipynb`

---

## 1. Objective

This stage trains a **semantic segmentation** model to automatically delineate the breast region from single-channel thermal images. The output binary masks serve two critical downstream purposes:

1. They define the **silhouette contours** $S_k \subset \Omega$ consumed by the 3D reconstruction pipeline (Stage 3).
2. They isolate the **region of interest (ROI)** for subsequent thermal analysis and PINN-based bioheat inversion (Stage 4), eliminating background thermal noise.

---

## 2. Data Loading and Normalisation (Phase 1)

### 2.1 Thermal Image Preprocessing

Each patient acquisition is stored as a 16-bit TIFF file containing calibrated pixel-level temperature values $P_{i,j}$. Before entering the network, every image is:

1. **Resized** to a canonical spatial resolution of $256 \times 256$ pixels using area interpolation to maintain thermal energy.
2. **Min–max normalised** to the $[0, 1]$ range:

$$
m_{i,j} = \frac{P_{i,j} - \min(P)}{\max(P) - \min(P)}
$$

where $P \in \mathbb{R}^{H \times W}$ denotes the full raw pixel array. This normalisation is theoretically crucial because it ensures that the input feature space lies within a standardized manifold, preventing exploding gradients and allowing the convolutional kernels to learn morphological features invariant to the absolute patient baseline temperature. 

### 2.2 Ground-Truth Mask Annotation Workflow

To train the U-Net via supervised learning, ground-truth binary masks are required to construct the target empirical distribution $y_i \in \{0, 1\}$. These were generated manually via an interactive polygon annotation loop (`MANUAL MASKING.ipynb`).

For each normalised image, a human annotator delineates a closed polygon $\mathcal{P}$ bounding the breast tissue. The annotation protocol enforces strict anatomical priors:
1. **Superior boundary:** Exclusion of the neck and upper torso.
2. **Lateral boundary:** Truncation at the lateral fold to prevent axillary (armpit) leakage.
3. **Inferior boundary:** Consistent tracing of the inframammary fold (IMF).

Once the polygon vertices $\mathcal{P} = \{(x_k, y_k)\}_{k=1}^N$ are defined, the continuous polygon is rasterized into a discrete binary mask using the scan-line fill algorithm (implemented via `cv2.fillPoly`). The resulting binary masks ($256 \times 256$ pixels) represent the ground-truth morphological silhouette of the breast and are saved as PNG files, perfectly mirroring the file hierarchy of the source TIFFs.

### 2.3 Dataset Splitting

Following the experimental protocol described in the reference paper (§3.3), the dataset is partitioned as:

| Split | Fraction | Samples |
|:---:|:---:|:---:|
| Training | 78 % | 126 |
| Test | 22 % | 36 |
| **Total** | 100 % | **162** |

---

## 3. Data Augmentation Theory

To mitigate overfitting and expand the empirical risk minimization (ERM) space on the limited medical dataset, lightweight affine and photometric augmentations are applied to the training distribution $p_{\text{data}}(x, y)$.

Let $\mathcal{T}$ be a set of stochastic transformation functions. We augment the dataset by sampling $T \sim \mathcal{T}$ such that the model minimizes the expected loss:
$$ \min_\theta \mathbb{E}_{(x,y)\sim p_{\text{data}}, T \sim \mathcal{T}} [\mathcal{L}(f_\theta(T(x)), T(y))] $$

The specific transformations applied are:
1. **Horizontal flip:** A geometric transformation $T_{\text{flip}}(x_{i,j}) = x_{W-i, j}$. Applied with $p=0.5$ identically to both the image $x$ and mask $y$.
2. **Brightness jitter:** A photometric transformation $T_{\text{bright}}(x) = \gamma x$ where $\gamma \sim \text{Uniform}[0.8,\, 1.2]$. Applied only to the image $x$ to simulate varying ambient thermal conditions.

---

## 4. Network Architecture (Phase 2)

### 4.1 U-Net Encoder–Decoder Theory

The model is a **4-level encoder–decoder U-Net**. The U-Net architecture is characterized by its contracting path (to capture high-level semantic context) and an expanding path (to enable precise localization).

Each stage of the contracting path can be represented as a composite function $\mathcal{F}_{\text{down}}^{(l)}$ mapping features from level $l-1$ to level $l$:
$$ \mathbf{h}^{(l)} = \text{MaxPool}\left( \text{ReLU}\left( \text{BN}\left( \mathbf{W}_2^{(l)} * \text{ReLU}\left( \text{BN}\left( \mathbf{W}_1^{(l)} * \mathbf{h}^{(l-1)} \right)\right) \right)\right) \right) $$

where $*$ denotes the convolution operator.

The expanding path concatenates high-resolution features $\mathbf{h}^{(l)}$ from the contracting path with upsampled features $\tilde{\mathbf{h}}^{(l)}$ via a skip connection, enabling the recovery of fine spatial details lost during pooling:
$$ \tilde{\mathbf{h}}^{(l-1)} = \mathcal{F}_{\text{up}}^{(l)}\left( \tilde{\mathbf{h}}^{(l)} \oplus \mathbf{h}^{(l-1)} \right) $$

where $\oplus$ represents feature-map concatenation along the channel dimension.

### 4.2 Weight Initialisation

To ensure stable signal propagation and prevent the vanishing/exploding variance problem in deep ReLU networks, all convolutional layers use **Kaiming Normal** (He) initialisation. If $n_l$ is the fan-in (number of input units) of layer $l$, the weights are initialized as:
$$ \mathbf{W}^{(l)} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_l}\right) $$

---

## 5. Loss Function Topology

The loss is a **weighted composite** of Binary Cross-Entropy with Logits (BCE) and a soft Dice loss. This hybrid approach leverages BCE for smooth pixel-level gradient propagation while using Dice loss to explicitly maximize the intersection-over-union metric, which is robust to class imbalance (e.g., small breast regions in lateral views).

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{BCE}} + \beta \cdot \mathcal{L}_{\text{Dice}}
$$

with $\alpha = 0.6$ and $\beta = 0.4$.

### 5.1 BCE with Logits

BCE measures the Kullback-Leibler divergence between the empirical distribution (ground truth $y_i \in \{0,1\}$) and the predicted Bernoulli distribution parameterised by $p_i = \sigma(z_i)$, where $z_i$ are the raw network logits:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i)\log(1 - p_i) \right]
$$

### 5.2 Soft Dice Loss

The Dice loss is a differentiable relaxation of the Sørensen–Dice index.

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i \cdot y_i + \varepsilon}{\sum_i p_i + \sum_i y_i + \varepsilon}
$$

where $\varepsilon = 10^{-6}$ is a Laplacian smoothing term preventing division by zero and stabilizing gradients when both prediction and ground truth are empty.

---

## 6. Evaluation Metric — Sørensen–Dice Coefficient

The primary evaluation metric measures volumetric overlap:

$$
\text{Dice}(A, B) = \frac{2 |A \cap B|}{|A| + |B|} = \frac{2 \text{TP}}{2\text{TP} + \text{FP} + \text{FN}}
$$

where $A$ is the binarised prediction (threshold $t = 0.5$) and $B$ is the ground-truth mask.

---

## 7. Training Protocol and Convergence

Training was executed using the AdamW optimiser, which decouples weight decay from the adaptive gradient updates, improving generalization.

| Parameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate $\eta_0$ | $3 \times 10^{-4}$ |
| Weight decay | $1 \times 10^{-4}$ |
| Scheduler | `ReduceLROnPlateau` |

Early stopping triggered at epoch 40, achieving a **Test Dice of $\mathbf{0.8935 \pm 0.0542}$**.

---

## 8. Inference Pipeline — Automated Segmentation

For inference, the network models a maximum a posteriori (MAP) estimation problem. The raw logits $z_i$ are passed through a sigmoid function to obtain posterior probabilities $P(y_i=1 | x)$. We estimate the optimal mask by thresholding:
$$ \hat{y}_i = \begin{cases} 1 & \text{if } \sigma(z_i) \geq 0.5 \\ 0 & \text{otherwise} \end{cases} $$

To enforce topological priors (a breast should be a single continuous anatomical mass), a morphological post-processing step computes connected components and retains only the component $C_{\max}$ with the largest area:
$$ C_{\max} = \arg\max_{C \in \mathcal{C}} \text{Area}(C) $$

Finally, Canny edge detection isolates the morphological boundary $\partial C_{\max}$, which defines the geometric silhouette for 3D visual hull reconstruction in Stage 3.
