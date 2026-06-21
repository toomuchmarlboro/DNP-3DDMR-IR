# Stage 2 — Attention U-Net Breast Region Segmentation

**Source notebook:** `finalized/2_unetsegmentation_fixed.ipynb`

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

where $P \in \mathbb{R}^{H \times W}$ denotes the full raw pixel array.

### 2.2 Ground-Truth Mask Annotation Workflow

Ground-truth binary masks were generated manually via an interactive polygon annotation loop (`MANUAL MASKING.ipynb`). The annotation protocol enforces strict anatomical priors:
1. **Superior boundary:** Exclusion of the neck and upper torso.
2. **Lateral boundary:** Truncation at the lateral fold to prevent axillary (armpit) leakage.
3. **Inferior boundary:** Consistent tracing of the inframammary fold (IMF).

Once the polygon vertices $\mathcal{P} = \{(x_k, y_k)\}_{k=1}^N$ are defined, the polygon is rasterized into a discrete binary mask using `cv2.fillPoly`. The resulting $256 \times 256$ binary masks are saved as PNG files.

### 2.3 Dataset Splitting

The dataset is partitioned into a **held-out test set** (fixed before any training) and a **training pool** over which 5-fold cross-validation is performed.

| Split | Fraction | Samples |
|:---:|:---:|:---:|
| Training pool | 78 % | 126 |
| Held-out test | 22 % | 36 |
| **Total** | 100 % | **162** |

The held-out test set is never used during model selection or hyperparameter tuning. Cross-validation folds are constructed exclusively within the 126-sample training pool.

---

## 3. Data Augmentation

To mitigate overfitting on the limited medical dataset, the following augmentations are applied during each training batch via **Albumentations**:

| Transform | Parameters | Applied to |
|---|---|---|
| Horizontal flip | $p = 0.5$ | image + mask |
| ShiftScaleRotate | shift 5%, scale 5%, rotate ±15° | image + mask |
| ElasticTransform | $\alpha=1$, $\sigma=50$ | image + mask |
| RandomBrightnessContrast | $p = 0.5$ | image only |
| CoarseDropout | 8 holes, 16×16 px | image only |

Geometric transforms are applied identically to image and mask to preserve spatial correspondence. Photometric transforms are applied to the image only.

---

## 4. Network Architecture (Phase 2)

### 4.1 Attention U-Net

The model is a **4-level Attention U-Net**. Attention gates are placed at every decoder level, suppressing irrelevant background activations and focusing the skip connections on the breast region.

Each encoder block applies:
$$ \mathbf{h}^{(l)} = \text{MaxPool}\left( \text{ReLU}\left( \text{BN}\left( \mathbf{W}_2^{(l)} * \text{ReLU}\left( \text{BN}\left( \mathbf{W}_1^{(l)} * \mathbf{h}^{(l-1)} \right)\right) \right)\right) \right) $$

The attention gate at each decoder level computes:
$$ \psi = \sigma\!\left( \mathbf{W}_\psi \cdot \text{ReLU}\!\left( \mathbf{W}_g g + \mathbf{W}_x x \right) \right) $$

where $g$ is the upsampled decoder feature and $x$ is the encoder skip-connection feature. The output is $x \cdot \psi$, gating skip features by spatial relevance.

### 4.2 Channel Flow

| Stage | Channels |
|---|---|
| Encoder 1 | 1 → 64 |
| Encoder 2 | 64 → 128 |
| Encoder 3 | 128 → 256 |
| Encoder 4 | 256 → 512 |
| Bottleneck | 512 → 1024 |
| Decoder 4 | 1024 → 512 (+ attention) |
| Decoder 3 | 512 → 256 (+ attention) |
| Decoder 2 | 256 → 128 (+ attention) |
| Decoder 1 | 128 → 64 (+ attention) |
| Output | 64 → 1 (logits) |

Total trainable parameters: **31,388,013**.

### 4.3 Weight Initialisation

All convolutional layers use **Kaiming Normal** (He) initialisation:
$$ \mathbf{W}^{(l)} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_l}\right) $$

---

## 5. Loss Function

The loss is a **Focal-Dice composite**:

$$
\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{Focal}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}
$$

### 5.1 Focal Loss

$$
\mathcal{L}_{\text{Focal}} = -\frac{\alpha}{N}\sum_{i=1}^N (1 - p_i)^\gamma \log p_i
$$

with $\alpha = 0.25$, $\gamma = 2.0$. The $(1-p_i)^\gamma$ modulating factor down-weights easy background pixels and concentrates the gradient on hard, mis-classified foreground pixels — particularly useful for class-imbalanced lateral views.

### 5.2 Soft Dice Loss

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i \cdot y_i + \varepsilon}{\sum_i p_i + \sum_i y_i + \varepsilon}
$$

with $\varepsilon = 10^{-6}$.

---

## 6. Evaluation Metric — Sørensen–Dice Coefficient

$$
\text{Dice}(A, B) = \frac{2 |A \cap B|}{|A| + |B|} = \frac{2 \text{TP}}{2\text{TP} + \text{FP} + \text{FN}}
$$

where $A$ is the binarised prediction (threshold $t = 0.5$) and $B$ is the ground-truth mask.

---

## 7. Training Protocol — 5-Fold Cross-Validation

Training uses 5-fold CV exclusively on the 126-sample training pool. The 36-sample test set is not touched until final evaluation.

| Parameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate $\eta_0$ | $3 \times 10^{-4}$ |
| Weight decay | $1 \times 10^{-4}$ |
| Scheduler | `ReduceLROnPlateau` (factor 0.5, patience 3) |
| Batch size | 2 |
| Max epochs | 120 |
| Early-stopping patience | 12 epochs |

Each fold saves the best-val-Dice checkpoint (`unet_fold_N.pth`).

**5-Fold CV Results (within training pool):**

| Fold | Best Val Dice | Early-stop epoch |
|:---:|:---:|:---:|
| 1 | 0.8772 | 56 |
| 2 | 0.9197 | 94 |
| 3 | 0.9206 | 86 |
| 4 | 0.9051 | 46 |
| 5 | 0.9087 | 41 |
| **Mean ± SD** | **0.9063 ± 0.0157** | **64.6 ± 22.3** |

---

## 8. Inference — 5-Model Ensemble

For inference, all 5 fold checkpoints are loaded and their sigmoid outputs are averaged before thresholding:

$$
\hat{p}_i = \frac{1}{5} \sum_{k=1}^{5} \sigma\!\left(z_i^{(k)}\right), \qquad \hat{y}_i = \mathbf{1}\!\left[\hat{p}_i \geq 0.5\right]
$$

A morphological post-processing step retains only the largest connected component $C_{\max}$:
$$ C_{\max} = \arg\max_{C \in \mathcal{C}} \text{Area}(C) $$

Finally, Canny edge detection isolates the morphological boundary $\partial C_{\max}$, which feeds into Stage 3 as the geometric silhouette for 3D visual hull reconstruction.

---

## 9. Held-Out Test Set Evaluation

After ensemble inference, the 36-sample held-out test set (never seen during cross-validation) is evaluated to obtain an unbiased performance estimate.

| Metric | Value |
|---|---|
| Ensemble Test Dice | TBD after rerun |
| Ensemble Test IoU | TBD after rerun |

> **Note:** The per-view Dice breakdown (Table 4.4) is computed exclusively over the 36 held-out test masks to ensure no training samples inflate the reported numbers.