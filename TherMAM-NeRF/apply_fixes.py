import json
import re

in_path = r"c:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\DNP-3DDMR-IR\TherMAM-NeRF\thermamnerf_v1.2.ipynb"
out_path = r"c:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\DNP-3DDMR-IR\TherMAM-NeRF\thermamnerf_v1.3.ipynb"

with open(in_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    
    # Fix CFG
    if "'lambda_dice'     : 1.0," in source:
        if "'lambda_bg'" not in source:
            source = source.replace("'lambda_dice'     : 1.0,", "'lambda_dice'     : 1.0,\n    'lambda_bg'       : 1.0,")
    
    # Fix 4: Global Normalization in __getitem__
    if "def __getitem__(self, idx):" in source and "tiffs_norm, tiffs_abs, masks = [], [], []" in source:
        old_getitem = """    def __getitem__(self, idx):
        p = self.patients[idx]
        tiffs_norm, tiffs_abs, masks = [], [], []
        tmins, tmaxs = [], []

        for v in self.view_names:
            raw   = load_tiff_celsius(str(p['tiffs'][v]), self.S)
            normd, tmin, tmax = normalize_thermal(raw)
            mask  = load_mask(str(p['masks'][v]), self.S)

            tiffs_norm.append(normd)
            tiffs_abs.append(raw)
            masks.append(mask)
            tmins.append(tmin)
            tmaxs.append(tmax)"""
        
        new_getitem = """    def __getitem__(self, idx):
        p = self.patients[idx]
        tiffs_norm, tiffs_abs, masks = [], [], []
        tmins, tmaxs = [], []

        # -- GLOBAL NORMALIZATION FIX --
        tiffs_abs_temp = []
        for v in self.view_names:
            raw = load_tiff_celsius(str(p['tiffs'][v]), self.S)
            tiffs_abs_temp.append(raw)
        
        tiffs_abs_stack = np.stack(tiffs_abs_temp)
        global_tmin = tiffs_abs_stack.min()
        global_tmax = tiffs_abs_stack.max()

        for v in range(len(self.view_names)):
            raw = tiffs_abs_temp[v]
            normd = (raw - global_tmin) / (global_tmax - global_tmin + 1e-6)
            mask  = load_mask(str(p['masks'][self.view_names[v]]), self.S)

            tiffs_norm.append(normd)
            tiffs_abs.append(raw)
            masks.append(mask)
            tmins.append(global_tmin)
            tmaxs.append(global_tmax)"""
        source = source.replace(old_getitem, new_getitem)

    # Fix 2: Grid Sample 'zeros'
    if "padding_mode='border'" in source:
        source = source.replace("padding_mode='border'", "padding_mode='zeros'")

    # Fix 3: Rendered Temp Denominator
    if "rendered_temp = rendered_temp / (rendered_mask.detach() + 1e-6)" in source:
        source = source.replace(
            "rendered_temp = rendered_temp / (rendered_mask.detach() + 1e-6)",
            "opacity_mask  = (rendered_mask.detach() > 0.05).float()\n        rendered_temp = (rendered_temp / (rendered_mask.detach() + 1e-6)) * opacity_mask"
        )
        
    # Fix 1: Background Ray Loss
    if "def run_one_batch(" in source and "loss_bg = background_ray_loss" not in source:
        bg_func = """def background_ray_loss(rendered_masks: list, gt_masks: torch.Tensor) -> torch.Tensor:
    loss = torch.tensor(0.0, device=gt_masks.device)
    V = gt_masks.shape[1]  # or length of rendered_masks
    for v in range(V):
        bg_pixels = (gt_masks[:, v] < 0.5)
        if bg_pixels.any():
            loss += rendered_masks[v][bg_pixels].pow(2).mean()
    return loss / V

"""
        source = source.replace("def run_one_batch(", bg_func + "def run_one_batch(")
        
        old_total = """    total = (cfg['lambda_dice']    * loss_dice +
             cfg['lambda_thermal'] * loss_thermal +
             cfg.get('lambda_tv', 0.0) * loss_tv)

    return {'dice': loss_dice, 'thermal': loss_thermal, 'tv': loss_tv, 'total': total}"""
        
        new_total = """    loss_bg = background_ray_loss(rendered_masks, gt_masks_sampled)

    total = (cfg['lambda_dice']    * loss_dice +
             cfg['lambda_thermal'] * loss_thermal +
             cfg.get('lambda_tv', 0.0) * loss_tv +
             cfg.get('lambda_bg', 0.0) * loss_bg)

    return {'dice': loss_dice, 'thermal': loss_thermal, 'tv': loss_tv, 'bg': loss_bg, 'total': total}"""
        source = source.replace(old_total, new_total)

    # Fix 5: Epoch Gating in training loop
    if "ld, _, _ = run_one_batch(batch, encoder, mlp, CFG, alpha, DEVICE)" in source:
        old_loop = "ld, _, _ = run_one_batch(batch, encoder, mlp, CFG, alpha, DEVICE)"
        new_loop = """cfg_step = dict(CFG)
        if epoch <= 50:
            cfg_step['lambda_thermal'] = 0.0
        else:
            cfg_step['lambda_thermal'] = 0.01
        ld, _, _ = run_one_batch(batch, encoder, mlp, cfg_step, alpha, DEVICE)"""
        source = source.replace(old_loop, new_loop)

    lines = source.split('\n')
    cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

# Add diagnostic cell after Projection Audit
audit_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '## 11. Projection Audit' in "".join(cell['source']):
        audit_idx = i + 1 # Code cell follows
        break

if audit_idx != -1:
    diag_code = """# ── Diagnostic: Thermal Stripe Analysis ─────────────────────────────────────
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

    print(f'\\n{vname} ({CFG["view_angles_deg"][v]}°):')
    print(f'  Temp range   : [{t_min:.4f}, {t_max:.4f}]')
    print(f'  Full std     : {t_std:.4f}')
    print(f'  FG-only std  : {fg_std:.4f}')
    print(f'  Col variance : {col_std:.4f}')
    print(f'  Row variance : {row_std:.4f}')
    print(f'  Dominant col freq: {dominant_freq:.4f} cycles/px')

# Visual: plot column variance profile for frontal view
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

rm_f, rt_f = render_view(encoder, mlp, tiffs_norm, masks, 2, CFG,
                          DEVICE, alpha=float(CFG['pos_enc_L']))
axes[0].imshow(rt_f.cpu().numpy(), cmap='inferno', vmin=0, vmax=1)
axes[0].set_title('Frontal Thermal (raw)')
axes[0].axis('off')

col_m = rt_f.cpu().numpy().mean(axis=0)
axes[1].plot(col_m, color='tomato')
axes[1].set_title('Column Mean Profile')
axes[1].set_xlabel('Pixel column')
axes[1].set_ylabel('Mean temperature')
axes[1].grid(alpha=0.3)

fft_m = np.abs(np.fft.rfft(col_m - col_m.mean()))
freqs = np.fft.rfftfreq(len(col_m))
axes[2].plot(freqs[1:], fft_m[1:], color='steelblue')
axes[2].set_title('FFT of Column Means')
axes[2].set_xlabel('Frequency (cycles/pixel)')
axes[2].set_ylabel('Amplitude')
axes[2].grid(alpha=0.3)

plt.suptitle('Thermal Stripe Diagnostic — Frontal View', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'stripe_diagnostic.png'), dpi=120)
plt.show()
"""
    
    diag_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in diag_code.split('\n')]
    }
    nb['cells'].insert(audit_idx + 1, diag_cell)
    
    diag_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 11.5 Diagnostic — Thermal Stripe Analysis\n"]
    }
    nb['cells'].insert(audit_idx + 1, diag_markdown)


with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Created thermamnerf_v1.3.ipynb with all fixes applied.")
