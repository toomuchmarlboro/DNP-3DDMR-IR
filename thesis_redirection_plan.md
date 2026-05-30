# Thesis Redirection: Dual-Input 3D-BreastNet

## The Problem We Discovered

We spent extensive time trying to fix the 2D U-Net segmentation, believing the shattered 3D models were caused by bad masks. We tried:
- Attention U-Net
- Focal-Dice Loss
- Contour/Perimeter Loss (Total Variation)
- CoordConv (Coordinate Convolutions)
- Edge-only Gaussian smoothing

**None of it worked.** The real problem was never the U-Net. It was a fundamental mismatch between two papers.

---

## Root Cause: Incompatible Paradigms

We were mixing two papers that require completely different types of input masks:

| Paper | What it does | What mask it needs |
|---|---|---|
| **Paper 3** — Costa et al. (978-3-031-44511-8_3) | 2D segmentation → curve extraction → NURBS surface | **Breast-only** masks (small, localized) |
| **Paper 2** — Saha et al. (978-3-031-44511-8_2) | 2D silhouettes → 3D-BreastNet → voxel volume | **Full body** silhouettes (entire torso) |

Our pipeline fed **breast-only masks** (from Paper 3's U-Net) into **3D-BreastNet** (from Paper 2). This broke the 3D math because:

1. **Front breast mask** = wide area across chest → 3D-BreastNet thinks torso exists here
2. **Side breast mask** = tiny floating blob → 3D-BreastNet thinks only a small piece of body exists
3. **3D intersection** of wide front + tiny side = shattered floating fragments

Paper 2 originally uses **Otsu thresholding** to extract the **entire body** as the silhouette. The full body is visible from all angles, giving the 3D algorithm strong geometric constraints.

---

## The Solution: Dual-Input 3D-BreastNet (Plan B)

Instead of choosing one paper over the other, we combine both.

### Architecture Change

Modify 3D-BreastNet to accept **10 input channels** instead of 5:

| Channels | Source | Role |
|---|---|---|
| 1–5 | **Otsu thresholding** (full body silhouette) | Spatial context — tells the network WHERE the body is |
| 6–10 | **U-Net** (breast-only segmentation) | Reconstruction target — tells the network WHAT to output |

### How It Works

```
┌─────────────────────┐     ┌─────────────────────┐
│  Otsu Full Body     │     │  U-Net Breast Only   │
│  (5 views × 128²)   │     │  (5 views × 128²)    │
└────────┬────────────┘     └────────┬────────────┘
         │                           │
         └───────────┬───────────────┘
                     │
              Stack: 10 channels
                     │
                     ▼
          ┌─────────────────────┐
          │   3D-BreastNet      │
          │   Encoder-Decoder   │
          │   (10ch → 128³)     │
          └────────┬────────────┘
                   │
                   ▼
          128 × 128 × 128 volume
          (breast-only voxels)
                   │
                   ▼
          ┌─────────────────────┐
          │   Self-Supervised    │
          │   Loss Function      │
          │                      │
          │  Project 3D → 2D     │
          │  Compare with        │
          │  BREAST-ONLY masks   │
          └─────────────────────┘
```

### Why This Works

- The **loss function** compares 2D projections against **breast-only masks** → network learns to only output breast voxels
- The **full body channels** give the encoder spatial reference → no more floating shards because the network knows where the breast sits relative to the body
- **No cropping needed** — the output is directly breast-only

### Why This Is Better Than v5

| Problem | v5 (current) | Dual-Input (proposed) |
|---|---|---|
| Floating shards | Severe | Eliminated (body anchors the shape) |
| Lateral view Dice | 0.79 (broken) | Should improve significantly |
| Spatial context | None (breast masks only) | Full body provides reference frame |
| Output | Full bust or shattered fragments | Breast-only volume directly |

---

## Honest Limitations

> [!WARNING]
> This approach will NOT produce a medically perfect anatomical model. It will produce a reasonable 3D approximation suitable for thermal analysis.

**What will be good:**
- Front and side surfaces of the breast
- Overall volume and curvature
- Sufficient for temperature overlay and bioheat simulation

**What will be imperfect:**
- Back of breast (chest wall attachment) — no camera sees this
- Cleavage area — visual hull cannot reconstruct concavities
- Fine skin details — 128³ resolution ≈ 3-5mm per voxel

**Defensible thesis claim:**
> *"We extended 3D-BreastNet with dual-channel context-guided reconstruction, where full-body thermal silhouettes provide spatial context while breast-specific segmentation masks define the reconstruction target, enabling direct breast-only 3D volume generation without post-processing."*

---

## Implementation Steps

### Phase 1: Full Body Silhouettes (Otsu)
> **Effort: ~1 hour | No training needed**

- [ ] **Step 1.1:** Write Otsu silhouette extraction function
  - Input: thermal TIFF image
  - Output: binary mask (body=white, background=black)
  - Include morphological cleanup + largest component filtering
  - Already stress-tested: works perfectly at 128×128

- [ ] **Step 1.2:** Generate Otsu masks for all 137 patients × 5 views = 685 images
  - Save to `data/organized_by_patient_otsu/` alongside existing data
  - Verify a random sample visually

### Phase 2: Breast-Only U-Net Cleanup
> **Effort: ~30 min | Use existing model**

- [ ] **Step 2.1:** Revert `UNET_SegmentationV2.ipynb` — remove CoordConv and Contour Loss
  - Go back to the original Focal-Dice loss (no contour penalty)
  - Remove the `use_coordconv` flag from AttentionUNet
  - The original Attention U-Net with Focal-Dice was actually producing reasonable breast masks

- [ ] **Step 2.2:** Add post-processing to U-Net inference
  - After the U-Net predicts a mask, keep only the largest connected component
  - This removes floating blobs (the "refinement" step from Paper 3)
  - This is 3 lines of OpenCV code

### Phase 3: Dual-Input DataLoader
> **Effort: ~2 hours**

- [ ] **Step 3.1:** Create new `PatientDataset` class in a new notebook (`breastnet3d_v7.ipynb`)
  - For each patient, load 5 thermal images
  - Generate 5 Otsu masks (full body) on-the-fly
  - Generate 5 U-Net masks (breast-only) using frozen pretrained model
  - Stack into 10-channel tensor: shape `(10, 128, 128)`
  - Target for loss: the 5 U-Net breast masks (channels 6–10)

- [ ] **Step 3.2:** Verify the dataloader
  - Visualize a few patients: show all 10 channels side by side
  - Confirm Otsu and U-Net masks are aligned and properly sized

### Phase 4: Modified 3D-BreastNet Architecture
> **Effort: ~1 hour**

- [ ] **Step 4.1:** Copy encoder-decoder from v5, modify input channels
  - Change first encoder layer: `DoubleConv3D(5, 64)` → `DoubleConv3D(10, 64)`
  - Change output volume size: 64³ → 128³ (adjust decoder transpose convolutions)
  - Everything else stays the same

- [ ] **Step 4.2:** Modify the loss function
  - Keep the self-supervised Dice loss from Paper 2
  - But compare projections against breast-only masks (channels 6–10), not full body
  - Add optional auxiliary loss: also compare against full body at lower weight (0.1×)
    to ensure the network uses the body context channels

### Phase 5: Training
> **Effort: ~4-8 hours of GPU time**

- [ ] **Step 5.1:** Train on GPU (RTX 2080 Ti)
  - Epochs: 400 (Paper 2 used 400)
  - Monitor: Dice loss convergence on validation set
  - Save best checkpoint

- [ ] **Step 5.2:** Evaluate
  - Project the 3D volume to all 5 standard angles
  - Compare projections with input breast masks (Dice, Hausdorff)
  - Visually inspect the 3D volume for floating artifacts
  - Apply `cc3d.connected_components` to clean up any minor blobs

### Phase 6: Temperature Overlay & Visualization
> **Effort: ~2 hours**

- [ ] **Step 6.1:** Implement temperature overlay (same as Paper 2, Section 2.2.5)
  - Rotate 3D volume to estimated view angle
  - Map 2D temperatures onto visible 3D surface voxels

- [ ] **Step 6.2:** Export as mesh (STL/OBJ) for visualization
  - Use marching cubes to convert voxels → mesh
  - Apply thermal colormap to mesh vertices

---

## Order of Work

```
START HERE
    │
    ▼
Phase 2.1: Revert U-Net (remove contour loss + coordconv)
    │
    ▼
Phase 1.1: Write Otsu function
    │
    ▼
Phase 1.2: Generate all Otsu masks
    │
    ▼
Phase 3.1: Build dual-input DataLoader
    │
    ▼
Phase 3.2: Visually verify DataLoader output
    │
    ▼
Phase 4.1: Modify 3D-BreastNet architecture (10ch, 128³)
    │
    ▼
Phase 4.2: Update loss function
    │
    ▼
Phase 5.1: Train (400 epochs)
    │
    ▼
Phase 5.2: Evaluate + clean up artifacts
    │
    ▼
Phase 6: Temperature overlay + mesh export
    │
    ▼
DONE — Thesis-ready 3D breast model
```

> [!IMPORTANT]
> Start with Phase 2.1 (reverting the U-Net) because the dual-input approach needs the breast-only U-Net to work reasonably well. The original Attention U-Net before our contour/coordconv changes was the best performing version.

---

## Files That Will Be Created/Modified

| File | Action |
|---|---|
| `UNET_SegmentationV2.ipynb` | Revert to original Focal-Dice (remove contour loss + coordconv) |
| `breastnet3d_v7.ipynb` | NEW — Dual-input 3D-BreastNet with 128³ output |
| `data/organized_by_patient_otsu/` | NEW — Generated Otsu full-body masks |
| `breastnet3d_v5.ipynb` | Keep as reference (do not modify) |
| `breastnet3d_v6.ipynb` | Keep as reference (do not modify) |
