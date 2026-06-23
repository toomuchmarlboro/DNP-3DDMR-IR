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

1. **Resized** to a canonical spatial resolution of $256 \times 256$ pixels using area interpolation (`cv2.INTER_AREA`) to maintain thermal energy density.
2. **Min–max normalised** to the $[0, 1]$ range:

$$
m_{i,j} = \frac{P_{i,j} - \min(P)}{\max(P) - \min(P) + \varepsilon}
$$

where $P \in \mathbb{R}^{H \times W}$ denotes the full raw pixel array and $\varepsilon$ prevents division by zero for constant-valued images.

### 2.2 Ground-Truth Mask Annotation Workflow

Ground-truth binary masks were generated manually via an interactive polygon annotation loop (`MANUAL MASKING.ipynb`). The annotation protocol enforces strict anatomical priors:

1. **Superior boundary:** Exclusion of the neck and upper torso.
2. **Lateral boundary:** Truncation at the lateral fold to prevent axillary leakage.
3. **Inferior boundary:** Consistent tracing of the inframammary fold (IMF).

Once the polygon vertices $\mathcal{P} = \{(x_k, y_k)\}_{k=1}^N$ are defined, the polygon is rasterized into a discrete binary mask using `cv2.fillPoly`. The resulting $256 \times 256$ binary masks are saved as PNG files, mirroring the file hierarchy of the source TIFFs.

### 2.3 Dataset Splitting

The full dataset (162 labelled images) is partitioned into a **fixed held-out test set** and a **training pool**. The split is performed once before any training and seeded for reproducibility (`SEED = 42`):

```python
n_train = int(0.78 * n_total)   # 126
n_test  = n_total - n_train     # 36
train_set, test_set = random_split(full_dataset, [n_train, n_test],
                                   generator=torch.Generator().manual_seed(SEED))
```

| Split | Fraction | Samples | Role |
|:---:|:---:|:---:|:---|
| Training pool | 78 % | 126 | 5-fold CV — training and validation |
| Held-out test | 22 % | 36 | Final unbiased evaluation only |
| **Total** | 100 % | **162** | |

The held-out test set is never used during model selection, hyperparameter tuning, or early stopping.

---

## 3. Data Augmentation (Phase 1.2)

To mitigate overfitting on the limited medical dataset, the following stochastic transformations are applied per training batch via **Albumentations**:

| Transform | Parameters | Applied to |
|---|---|---|
| `HorizontalFlip` | $p = 0.5$ | image + mask |
| `ShiftScaleRotate` | shift 5 %, scale 5 %, rotate ±15° | image + mask |
| `ElasticTransform` | $\alpha=1$, $\sigma=50$ | image + mask |
| `RandomBrightnessContrast` | $p = 0.5$ | image only |
| `CoarseDropout` | 8 holes, $16 \times 16$ px | image only |

Geometric transforms are applied identically to image and mask to preserve spatial correspondence. Photometric transforms are applied to the image only, as masks are binary.

---

## 4. Network Architecture — Attention U-Net (Phase 2)

### 4.1 Architecture Overview

The model is a **4-level Attention U-Net** with skip-connection attention gates at every decoder level. Attention gates learn to suppress irrelevant background activations before the skip-connection features are concatenated, focusing capacity on the breast silhouette.

The encoder contracting path applies at each level $l$:

$$
\mathbf{h}^{(l)} = \text{MaxPool}\!\left(\text{ReLU}\!\left(\text{BN}\!\left(\mathbf{W}_2^{(l)} * \text{ReLU}\!\left(\text{BN}\!\left(\mathbf{W}_1^{(l)} * \mathbf{h}^{(l-1)}\right)\right)\right)\right)\right)
$$

Each attention gate computes a soft spatial mask $\psi \in [0, 1]$:

$$
\psi = \sigma\!\left(\mathbf{W}_\psi \cdot \text{ReLU}\!\left(\mathbf{W}_g\, g + \mathbf{W}_x\, x\right)\right)
$$

where $g$ is the upsampled decoder feature and $x$ is the encoder skip-connection feature. The gated output $x \cdot \psi$ replaces the raw skip connection, suppressing background regions before concatenation.

### 4.2 Channel Flow

| Stage | In → Out channels | Notes |
|---|---|---|
| Encoder 1 | 1 → 64 | No dropout |
| Encoder 2 | 64 → 128 | No dropout |
| Encoder 3 | 128 → 256 | Dropout 0.1 |
| Encoder 4 | 256 → 512 | Dropout 0.1 |
| Bottleneck | 512 → 1024 | Dropout 0.2 |
| Decoder 4 | 1024 → 512 | Attention + dropout 0.1 |
| Decoder 3 | 512 → 256 | Attention + dropout 0.1 |
| Decoder 2 | 256 → 128 | Attention |
| Decoder 1 | 128 → 64 | Attention |
| Output | 64 → 1 | Logits (no sigmoid in head) |

Total trainable parameters: **31,388,013**.

### 4.3 Weight Initialisation

All convolutional layers use **Kaiming Normal** (He) initialisation. For a layer with fan-in $n_l$:

$$
\mathbf{W}^{(l)} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_l}\right)
$$

BatchNorm weights are initialised to 1, biases to 0.

---

## 5. Loss Function

The loss is a **Focal-Dice composite**:

$$
\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{Focal}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}
$$

### 5.1 Focal Loss

$$
\mathcal{L}_{\text{Focal}} = -\frac{\alpha}{N}\sum_{i=1}^N (1 - p_i)^\gamma \left[ y_i \log p_i + (1-y_i)\log(1-p_i) \right]
$$

with $\alpha = 0.25$, $\gamma = 2.0$. The $(1-p_i)^\gamma$ modulating factor down-weights easy, correctly-classified background pixels and concentrates the gradient on hard foreground boundary pixels — particularly beneficial for class-imbalanced lateral views where the breast occupies a small fraction of the image.

### 5.2 Soft Dice Loss

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i \cdot y_i + \varepsilon}{\sum_i p_i + \sum_i y_i + \varepsilon}
$$

with $\varepsilon = 10^{-6}$. Dice loss directly maximises volumetric overlap and is robust to the foreground/background pixel imbalance inherent in breast silhouette segmentation.

---

## 6. Evaluation Metric — Sørensen–Dice Coefficient

$$
\text{Dice}(A, B) = \frac{2 |A \cap B|}{|A| + |B|} = \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}
$$

where $A$ is the binarised prediction at threshold $t = 0.5$ and $B$ is the ground-truth mask.

---

## 7. Training Protocol — 5-Fold Cross-Validation (Phase 2.2)

5-fold CV is performed **exclusively within the 126-sample training pool**. The 36-sample test set is sealed until Section 9.

```python
kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(train_set)))):
    train_sub = torch.utils.data.Subset(train_set, train_ids)   # ~101 samples
    val_sub   = torch.utils.data.Subset(train_set, val_ids)     # ~25 samples
```

Each fold trains a fresh model and saves the best-val-Dice checkpoint (`unet_fold_N.pth`).

| Hyperparameter | Value |
|---|---|
| Device | `cuda:1` |
| Optimiser | AdamW |
| Learning rate $\eta_0$ | $3 \times 10^{-4}$ |
| Weight decay | $1 \times 10^{-4}$ |
| LR scheduler | `ReduceLROnPlateau` (factor 0.5, patience 3, min $10^{-6}$) |
| Batch size | 2 |
| Max epochs | 120 |
| Early-stopping patience | 12 epochs (no val-Dice improvement) |
| AMP | Enabled (CUDA) |
| Gradient clipping | max norm 1.0 |

**5-Fold CV Results (within training pool, 126 samples):**

| Fold | CV Train samples | CV Val samples | Best Val Dice | Early-stop epoch |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 101 | 25 | 0.8772 | 56 |
| 2 | 101 | 25 | 0.9197 | 94 |
| 3 | 101 | 25 | 0.9206 | 86 |
| 4 | 101 | 25 | 0.9051 | 46 |
| 5 | 101 | 25 | 0.9087 | 41 |
| **Mean ± SD** | | | **0.9063 ± 0.0157** | **64.6 ± 22.3** |

> **Note:** These CV results are from the original notebook run where K-fold used `full_dataset`. After rerunning `finalized/2_unetsegmentation_fixed.ipynb` (which correctly restricts K-fold to `train_set`), these values will change slightly.

---

## 8. Inference Pipeline — 5-Model Ensemble (Phase 3)

All 5 fold checkpoints are loaded and their sigmoid outputs are averaged before thresholding:

$$
\hat{p}_i = \frac{1}{5} \sum_{k=1}^{5} \sigma\!\left(z_i^{(k)}\right), \qquad \hat{y}_i = \mathbf{1}\!\left[\hat{p}_i \geq 0.5\right]
$$

Ensemble averaging reduces variance from any single fold's idiosyncrasies and typically outperforms individual fold models.

To enforce the topological prior that a breast is a single continuous anatomical mass, a morphological post-processing step retains only the largest connected component $C_{\max}$:

$$
C_{\max} = \arg\max_{C \in \mathcal{C}} \text{Area}(C)
$$

The final mask is saved at $128 \times 128$ resolution (downsampled with `cv2.INTER_NEAREST`) to match the NeRF pipeline input dimensions. Canny edge detection on $C_{\max}$ produces the geometric silhouette $\partial C_{\max}$ consumed by Stage 3.

---

## 9. Held-Out Test Set Evaluation

After ensemble generation, the sealed 36-sample test set is evaluated once to obtain an unbiased performance estimate:

```python
# ensemble forward pass over test_set DataLoader
ensemble_probs = mean([sigmoid(model_k(img)) for model_k in model_ensemble])
pred_bin = (ensemble_probs > 0.5).float()
```

After rerunning the fixed notebook (K-fold restricted to `train_set`), the
5-fold validation Dice settled at **0.895 ± 0.021** (from `unet_fold_*.pth`
checkpoints), and the sealed 36-sample test set gave:

| Metric | Value |
|---|---|
| Ensemble Test Dice (weighted over 36 images) | **0.906** |

Per-view breakdown (Tabel 4.4):

| View | n | Dice (mean ± std) |
|---|---|---|
| Anterior (Front) | 7 | 0.923 ± 0.037 |
| Right Oblique (45°) | 8 | 0.939 ± 0.022 |
| Left Oblique (45°) | 7 | 0.921 ± 0.022 |
| Right Lateral (90°) | 8 | 0.868 ± 0.083 |
| Left Lateral (90°) | 6 | 0.878 ± 0.084 |
| **Overall** | **36** | **0.906** |

> Source: `finalized/Stage 2/Tabel_4_4_Dice_Score_UNet_Set_Uji.csv`,
> `finalized/Stage 2/unet_training_history/fold_*_history.json`. A
> thesis-ready Indonesian write-up of method + results is in
> `Docs/Stage2_Hasil_Metode_Segmentasi.md`.

The per-view breakdown (Table 4.4) is computed exclusively over the 36 held-out test masks (`test_mask_paths = {full_dataset.mask_paths[i] for i in test_set.indices}`), ensuring no training samples inflate the reported per-view numbers.