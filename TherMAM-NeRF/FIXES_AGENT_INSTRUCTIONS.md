# Agent Instructions: ThermalNeRF Fix Pass
# Target file: thermalnerf_v1.ipynb
# Apply all changes in order. Do not reformat unrelated code.

---

## Context

Current observed state (lambda_thermal=0.0, Mean Dice=0.950):
- Dark cuts and holes inside rendered masks — background ray aliasing
- White residuals outside GT silhouette boundary — unsupervised background rays
- Vertical stripe banding in thermal row — cause uncertain, see Diagnostic below
- Edge artifacts on RL/LL lateral views — border padding on grid_sample

Four confirmed bugs are fixed below. A mandatory diagnostic cell must be
added BEFORE applying Fix 5 (re-enabling thermal loss) to determine the
cause of the vertical stripe pattern. Do not skip the diagnostic.

Primary culprit for mask residuals: **background rays have no supervision**.
Secondary culprit for mask residuals: **border padding on grid_sample**.
Confirmed fix for thermal denominator: **opacity gate on rendered_temp**.
Separate correctness issue: **global vs per-view thermal normalisation**.
Stripe cause: **unknown — requires diagnostic before fixing**.

---

## Fix 1 — Background Ray Loss (PRIMARY FIX)
### Confidence: HIGH — direct mathematical cause of outside-mask residuals

### Location
Cell 8 (`compute_loss` and `tv_loss_3d` definitions cell).

### What to add
Add the following function **immediately before** `compute_loss`:

```python
def background_ray_loss(rendered_masks: list, gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Explicitly supervise background rays to have zero accumulated density.
    For every pixel where GT mask = 0, the rendered opacity must also → 0.
    This is the primary fix for density accumulating outside the breast region.

    Without this, the Dice loss only pulls the foreground shape toward the GT —
    it provides zero gradient on background rays, leaving the network free to
    place density anywhere in the volume boundary.

    rendered_masks : list of (B, H, W) tensors, one per view
    gt_masks       : (B, V, H, W)
    returns        : scalar loss
    """
    loss = torch.tensor(0.0, device=gt_masks.device)
    V = gt_masks.shape[1]
    for v in range(V):
        bg_pixels = (gt_masks[:, v] < 0.5)          # True where GT is background
        if bg_pixels.any():
            # L2 penalty on any opacity accumulating in background
            # Using .pow(2) rather than .abs() gives stronger gradient near zero
            loss += rendered_masks[v][bg_pixels].pow(2).mean()
    return loss / V
```

### What to change in `compute_loss`

The function signature must accept `cfg` as a parameter.
Add the background loss call and include it in `total`.

Replace the `total` computation at the end of `compute_loss` with:

```python
    # ── Background ray loss ──────────────────────────────────────────────────
    loss_bg = background_ray_loss(rendered_masks, gt_masks)

    total = (cfg['lambda_dice']    * loss_dice    +
             cfg['lambda_thermal'] * loss_thermal +
             cfg['lambda_tv']      * loss_tv      +
             cfg['lambda_bg']      * loss_bg)

    return {'total': total, 'dice': loss_dice,
            'thermal': loss_thermal, 'tv': loss_tv, 'bg': loss_bg}
```

### What to add to CFG (Cell 1)

Inside the `CFG` dict, add:
```python
'lambda_bg': 1.0,    # same weight as Dice — this is a hard geometric constraint
```

### Verification
After this fix, rendered masks should contain no white regions outside the GT
silhouette boundary. Check the projection audit — all residuals outside the
GT mask should be gone or negligible.

---

## Fix 2 — Grid Sample Padding Mode (SECONDARY FIX)
### Confidence: HIGH — known PyTorch grid_sample behavior

### Location
Cell 4 (`project_and_sample` function), the `F.grid_sample` call.

### What to change

```python
# FROM:
sampled = F.grid_sample(fmap, grid, mode='bilinear',
                        padding_mode='border', align_corners=True)

# TO:
sampled = F.grid_sample(fmap, grid, mode='bilinear',
                        padding_mode='zeros', align_corners=True)
```

### Why
`padding_mode='border'` clamps out-of-bounds projected coordinates to the
nearest edge pixel. When a 3D point near the volume boundary projects outside
the image plane, it samples the edge feature value — which is nonzero because
the breast mask touches image borders. This gives the MLP a spurious signal
to place density at volume boundaries even when the GT mask is empty there.

`padding_mode='zeros'` returns a zero feature vector for out-of-bounds
projections, giving the MLP no signal to place density outside the image
footprint.

### Verification
Out-of-bounds projection artifacts (typically appearing at left/right edges
of lateral views) should disappear. RL and LL Dice scores should improve
slightly as the encoder no longer confuses boundary wrapping with real anatomy.

---

## Fix 3 — Rendered Temperature Denominator Instability
### Confidence: HIGH for thermal row noise — moderate for whether it affects mask row

### Location
Cell 7 (`volume_render` function), the `rendered_temp` computation.

### What to change

```python
# FROM:
rendered_temp = rendered_temp / (rendered_mask.detach() + 1e-6)

# TO:
# Zero out temperature on rays where accumulated opacity is negligible.
# Without this, near-empty background rays produce T = tiny_numerator / tiny_denominator
# which is numerically undefined and produces the purple grid pattern seen in
# the thermal row of the projection audit.
opacity_mask  = (rendered_mask.detach() > 0.05).float()
rendered_temp = (rendered_temp / (rendered_mask.detach() + 1e-6)) * opacity_mask
```

### Why
For background rays, both the temperature numerator and the opacity denominator
are near zero. The ratio is numerically unstable — small floating point
differences between the two produce large temperature values that appear as
the purple grid artifact in the rendered thermal row. Gating with `opacity_mask`
forces the output to exactly zero where the ray is effectively empty.

### Verification
The purple/dark grid pattern in the bottom row of the projection audit should
disappear. Thermal rendering should show clean zeros in the background with
smooth temperature values only inside the breast region.

---

## Fix 4 — Per-View Thermal Normalisation
### Confidence: MEDIUM — correct fix but not the primary cause of mask residuals
### This fixes gradient consistency when lambda_thermal > 0

### Location
Cell 3 (`BreastThermDataset.__getitem__`).

### What to change

The normalisation must happen **per view**, not globally across all five views.

```python
# FROM (global normalisation across all views):
def __getitem__(self, idx):
    p = self.patients[idx]
    tiffs_norm, tiffs_abs, masks = [], [], []
    tmins, tmaxs = [], []

    for v in self.view_names:
        raw   = load_tiff_celsius(str(p['tiffs'][v]), self.S)
        normd, tmin, tmax = normalize_thermal(raw)   # BUG: this is per-view already
        mask  = load_mask(str(p['masks'][v]), self.S)
        ...

# Verify that normalize_thermal is called once per view independently.
# If tmin/tmax are computed globally (across all views concatenated), fix as below:

# CORRECT — ensure normalize_thermal is called independently per view:
for v in self.view_names:
    raw  = load_tiff_celsius(str(p['tiffs'][v]), self.S)
    # normalize_thermal must use ONLY this view's raw array, not a concatenation
    normd, tmin_v, tmax_v = normalize_thermal(raw)
    mask = load_mask(str(p['masks'][v]), self.S)

    tiffs_norm.append(normd)
    tiffs_abs.append(raw)
    masks.append(mask)
    tmins.append(tmin_v)   # per-view scalar
    tmaxs.append(tmax_v)   # per-view scalar
```

### Why this matters
The frontal view captures a wider temperature range than lateral views.
Global normalisation maps these to different positions in [0,1] depending on
which view has the global min/max. When the thermal MSE compares rendered
temperature against normalised TIFFs, it penalises the network with
contradictory targets — a predicted temperature of 0.8 at a 3D point
is compared against 0.8 (frontal) and 0.7 (lateral) at the same physical
location, producing opposing gradients. Per-view normalisation makes each
view's [0,1] scale internally consistent, removing the contradiction.

### When this matters
Only active when `lambda_thermal > 0`. If thermal loss is currently disabled
this fix has no training effect, but it should be applied for correctness
before re-enabling thermal loss.

### Verification
With `lambda_thermal = 0.01` and this fix applied, thermal loss should
decrease smoothly without oscillation. The rendered thermal row should show
spatially coherent temperature patterns that agree across views.

---

## Diagnostic — Vertical Stripe Pattern in Thermal Row
### Run BEFORE Fix 5. Do not re-enable thermal loss until cause is identified.

### What was observed
With `lambda_thermal=0.0`, the rendered thermal row shows consistent vertical
banding across all 5 views. The mask row does not show this pattern.
Three possible causes — they require different fixes:

**Hypothesis A — Unsupervised temperature head**
With `lambda_thermal=0.0` the temperature MLP head has no loss signal.
It outputs whatever pattern emerges from the network's internal structure,
which for a positional-encoding MLP often produces periodic sine/cosine
patterns aligned to the encoding frequencies.
Signature: high std, values spread across full [0,1] range, no spatial
coherence with breast anatomy.

**Hypothesis B — Ray sampling aliasing**
The orthographic camera casts parallel rays along a fixed direction per view.
With stratified sampling, rays in the same pixel column share similar
sampling patterns. If density is uneven, this creates column-aligned
rendering artifacts that appear in both density and temperature outputs.
Signature: stripes correlate with ray direction, present even in mask row
on close inspection, not perfectly vertical (slight angle matching view).

**Hypothesis C — MLP skip connection periodic activations**
The skip connection at the midpoint layer re-injects positional encoding
into mid-network activations. If the MLP overfits to a specific frequency
band, its outputs can show periodic spatial patterns along one axis.
Signature: stripes have a specific spatial frequency matching 2^k * pi
for some integer k from the positional encoding.

### What to add
Add a new cell immediately after Cell 11 (the projection audit cell).
Label it clearly: `## Diagnostic — Thermal Stripe Analysis`.

```python
# ── Diagnostic: Thermal Stripe Analysis ─────────────────────────────────────
# Run this cell after the projection audit to identify the cause of
# vertical banding in the rendered thermal row.
# DO NOT proceed to Fix 5 until you have read the output of this cell.

import scipy.signal

print('=' * 60)
print('THERMAL STRIPE DIAGNOSTIC')
print('=' * 60)

diag_results = {}

for v, vname in enumerate(CFG['view_names']):
    rm, rt = render_view(encoder, mlp, tiffs_norm, masks, v, CFG,
                         DEVICE, alpha=float(CFG['pos_enc_L']))

    rt_np  = rt.cpu().numpy()
    rm_np  = rm.cpu().numpy()

    # Only analyse inside the breast region
    fg_mask = (rm_np > 0.3)
    fg_vals = rt_np[fg_mask]

    t_min, t_max = rt_np.min(), rt_np.max()
    t_std        = rt_np.std()
    fg_std       = fg_vals.std() if fg_mask.any() else 0.0

    # Column-wise mean to detect vertical periodicity
    col_means    = rt_np.mean(axis=0)          # (W,)
    col_std      = col_means.std()             # high = vertical stripes

    # Row-wise mean to detect horizontal periodicity
    row_means    = rt_np.mean(axis=1)          # (H,)
    row_std      = row_means.std()             # high = horizontal stripes

    # FFT along columns to find dominant stripe frequency
    fft_col      = np.abs(np.fft.rfft(col_means - col_means.mean()))
    dominant_freq_idx = np.argmax(fft_col[1:]) + 1  # skip DC
    dominant_freq     = dominant_freq_idx / len(col_means)

    diag_results[vname] = {
        'range'         : (t_min, t_max),
        'full_std'      : t_std,
        'fg_std'        : fg_std,
        'col_std'       : col_std,
        'row_std'       : row_std,
        'dominant_freq' : dominant_freq,
    }

    print(f'\n{vname} ({CFG["view_angles_deg"][v]}°):')
    print(f'  Temp range   : [{t_min:.4f}, {t_max:.4f}]')
    print(f'  Full std     : {t_std:.4f}   (>0.3 → spread across [0,1] → Hypothesis A)')
    print(f'  FG-only std  : {fg_std:.4f}  (high inside mask → not just background noise)')
    print(f'  Col variance : {col_std:.4f}  (>0.05 → strong vertical stripes)')
    print(f'  Row variance : {row_std:.4f}  (compare with col — directional?)')
    print(f'  Dominant col freq: {dominant_freq:.4f} cycles/px '
          f'(~{1/dominant_freq:.1f}px period)')

print('\n' + '=' * 60)
print('INTERPRETATION GUIDE')
print('=' * 60)
print("""
If full_std > 0.3 AND fg_std > 0.2:
    → Hypothesis A (unsupervised head)
    → Fix: re-enable lambda_thermal=0.01 with epoch gate (Fix 5)
    → The stripes will disappear once the head has supervision.

If col_std >> row_std AND full_std < 0.3:
    → Hypothesis B (ray sampling aliasing)
    → Fix: increase n_samples from 64 to 128 in CFG
    → Alternatively: add small random per-ray depth offset during inference
      (currently jitter is training-only).

If dominant_freq matches a power-of-2 pattern (0.031, 0.062, 0.125, 0.25):
    → Hypothesis C (positional encoding frequency artefact)
    → Fix: reduce pos_enc_L from 8 to 6, or add weight decay 1e-4 to MLP.

Multiple hypotheses can be true simultaneously.
""")

# Visual: plot column variance profile for frontal view
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: raw thermal render
rm_f, rt_f = render_view(encoder, mlp, tiffs_norm, masks, 2, CFG,
                          DEVICE, alpha=float(CFG['pos_enc_L']))
axes[0].imshow(rt_f.cpu().numpy(), cmap='inferno', vmin=0, vmax=1)
axes[0].set_title('Frontal Thermal (raw)')
axes[0].axis('off')

# Panel 2: column mean profile — stripe signature
col_m = rt_f.cpu().numpy().mean(axis=0)
axes[1].plot(col_m, color='tomato')
axes[1].set_title('Column Mean Profile\n(flat=no stripes, oscillating=stripes)')
axes[1].set_xlabel('Pixel column')
axes[1].set_ylabel('Mean temperature')
axes[1].grid(alpha=0.3)

# Panel 3: FFT of column means — frequency fingerprint
fft_m = np.abs(np.fft.rfft(col_m - col_m.mean()))
freqs = np.fft.rfftfreq(len(col_m))
axes[2].plot(freqs[1:], fft_m[1:], color='steelblue')
axes[2].set_title('FFT of Column Means\n(peaks = stripe frequencies)')
axes[2].set_xlabel('Frequency (cycles/pixel)')
axes[2].set_ylabel('Amplitude')
axes[2].grid(alpha=0.3)

plt.suptitle('Thermal Stripe Diagnostic — Frontal View', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'stripe_diagnostic.png'), dpi=120)
plt.show()

print('\nSave diagnostic output and read interpretation before proceeding to Fix 5.')
```

### What to look for in the output
Read the printed interpretation guide and match your numbers to one of the
three hypotheses. Save `stripe_diagnostic.png` and include it when reporting
results. The agent applying Fix 5 must reference which hypothesis was
confirmed before proceeding.

---

## Fix 5 — Re-enable Thermal Loss with Safe Hyperparameters
### Apply only after Fixes 1–4 are verified AND diagnostic confirms Hypothesis A

### Location
Cell 1 (CFG dict) and Cell 9 (training loop).

### Changes to CFG

```python
# FROM:
'lambda_thermal': 0.0,    # disabled due to noise

# TO:
'lambda_thermal': 0.01,   # reduced from 0.1 — thermal is secondary to geometry
```

### Change to training loop (Cell 9)
Add epoch gating so thermal loss only activates after geometry has stabilised:

```python
# Inside the training loop, before run_one_batch:
cfg_step = dict(CFG)
if epoch <= 50:
    cfg_step['lambda_thermal'] = 0.0   # geometry-only phase
# pass cfg_step instead of CFG to run_one_batch
loss_dict, _, _ = run_one_batch(batch, encoder, mlp, cfg_step, alpha, DEVICE)
```

### Why
Thermal loss on unstable geometry produces poisoned gradients — the rendered
temperature denominator is unstable when σ is near-random at epoch 1.
Waiting 50 epochs gives the Dice loss time to establish a coherent breast
shape before thermal supervision is added.

---

## Implementation Order

Apply in this exact order. Do not skip steps.

1. Fix 2 (padding_mode) — one line, zero risk, apply first
2. Fix 3 (denominator gate) — two lines, zero risk
3. Fix 1 (background ray loss) — new function + CFG key, retrain, verify audit plot
4. Fix 4 (per-view normalisation) — verify `normalize_thermal` call structure
5. **Run Diagnostic** — add diagnostic cell, run it, read output, identify hypothesis
6. Fix 5 (re-enable thermal) — only after diagnostic confirms cause and Fixes 1–4 verified

If diagnostic identifies Hypothesis B (ray aliasing) instead of A:
- Increase `n_samples` from 64 to 128 in CFG before re-enabling thermal
- Also enable jitter during inference by removing the training-only gate

If diagnostic identifies Hypothesis C (positional encoding):
- Reduce `pos_enc_L` from 8 to 6 in CFG
- Add `weight_decay=1e-4` to the Adam optimiser in Cell 9

---

## Expected Outcome After All Fixes

| Metric | Before (0.931) | Current (0.950) | Expected After |
|---|---|---|---|
| Mask residuals outside GT | Present | Present | Gone |
| Dark cuts inside mask | Present | Present | Gone |
| Vertical thermal stripes | Present | Present | Gone (cause-dependent) |
| Purple thermal background | Present | Gone (thermal disabled) | Gone |
| Mean Dice | 0.931 | 0.950 | 0.950–0.965 |
| Thermal MSE convergence | Diverges | N/A (disabled) | Smooth decrease |
| lambda_thermal usable | No | No | Yes at 0.01 |

The Dice score should not regress from 0.950 after applying Fixes 1–4.
Fix 1 adds a constraint geometrically consistent with the Dice loss —
it cannot hurt Dice if implemented correctly. If Dice regresses after
Fix 1, reduce `lambda_bg` from 1.0 to 0.5 before retraining.
