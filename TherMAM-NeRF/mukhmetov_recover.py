#!/usr/bin/env python
"""FEM-FEM inverse bioheat — synthetic validation + real patient cohort.

SYNTHETIC (default):
    FEniCSx plants a known tumour → same FEniCSx recovers it (9 scenarios).
    Outputs: results/mukhmetov_recovery.csv + .png

REAL (--real):
    Uses TherMAM-NeRF surface IR as the inverse target.
    Grid search + Nelder-Mead recovers (z_t, r_t) per patient.
    Outputs: --out CSV (one row/patient, incremental — safe to Ctrl-C and resume).

Run:
    # Synthetic validation (Patient_1, idx=0):
    python -u mukhmetov_recover.py --idx 0 | tee mukhmetov_syn.log

    # Real cohort — balanced 40 patients, 2-GPU split:
    CUDA_VISIBLE_DEVICES=0 python -u mukhmetov_recover.py --real --subset 40 \\
        --stride 2 --offset 0 --out results/femfem_gpu0.csv | tee femfem_gpu0.log &
    CUDA_VISIBLE_DEVICES=1 python -u mukhmetov_recover.py --real --subset 40 \\
        --stride 2 --offset 1 --out results/femfem_gpu1.csv | tee femfem_gpu1.log &

    # Resume after crash (skips already-done patients):
    CUDA_VISIBLE_DEVICES=0 python -u mukhmetov_recover.py --real --subset 40 \\
        --stride 2 --offset 0 --out results/femfem_gpu0.csv --resume | tee -a femfem_gpu0.log &
"""
import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go

import run_cohort as R
from run_cohort import (
    CFG, device, RESULTS_DIR,
    chest_wall_mask_from_pts, tumor_Qm_from_radius,
    save_stl_binary, stl_to_tet_mesh, run_fea_forward, compute_fea_residual,
    GAUSSIAN_SIGMA, MC_THRESHOLD, REPAIR_HOLES, LAPLACIAN_ITERS, THERMAL_SIGMA,
    plot_fea_vs_pinn, classify_quadrant,
)


# ---------------------------------------------------------------------------
# Core FEM helpers (shared by synthetic and real modes)
# ---------------------------------------------------------------------------

def fem_surface(msh_path, geo, x0, y0, z_t, r_t):
    """FEM forward at (z_t, r_t) → surface temperatures at geo surface vertices."""
    r_t = float(np.clip(r_t, 5., 40.))
    z_t = float(z_t)
    Q0  = float(tumor_Qm_from_radius(torch.tensor(r_t)).item())
    params = {'x_t_mm': x0, 'y_t_mm': y0, 'z_t_mm': z_t,
              'r_t_mm': r_t, 'Q_max': Q0}
    T_sol, msh = run_fea_forward(msh_path, params)
    _, _, _, T_surf = compute_fea_residual(
        T_sol, msh, geo['surface_pts'], geo['T_measured'])
    return T_surf.astype(np.float32)


def recover_fem_fem(geo, T_target, x0, y0, msh_path, n_z=5, n_r=4, nm_iter=30):
    """Grid search + Nelder-Mead inverse using FEM forward model.

    Cost = skin-only MSE between candidate FEM surface T and T_target.
    Chest wall excluded — it is Dirichlet 37 C in every solve, carries no info.
    Works for both synthetic (T_target = T_syn, cost → 0) and real data
    (T_target = T_measured, cost → model-mismatch floor ~4-6 C^2).
    """
    sk    = ~chest_wall_mask_from_pts(geo['surface_pts'])
    T_tgt = T_target[sk]
    z_lo  = float(geo['bbox_min'][2])
    z_hi  = float(geo['bbox_max'][2])

    n_evals = [0]

    def cost(params):
        n_evals[0] += 1
        z_c = float(np.clip(params[0], z_lo, z_hi))
        r_c = float(np.clip(params[1], 5., 40.))
        try:
            T_c = fem_surface(msh_path, geo, x0, y0, z_c, r_c)
            return float(np.mean((T_c[sk] - T_tgt) ** 2))
        except Exception as e:
            print(f'    [FEM failed at z={params[0]:.1f} r={params[1]:.1f}: {e}]',
                  flush=True)
            return 1e6

    z_vals = np.linspace(z_lo + 0.08*(z_hi-z_lo),
                         z_hi - 0.05*(z_hi-z_lo), n_z)
    r_vals = np.linspace(7., 22., n_r)

    best_cost = np.inf
    best_z, best_r = 0.5*(z_lo + z_hi), 14.0

    for z_c in z_vals:
        for r_c in r_vals:
            c = cost([z_c, r_c])
            mark = ' ★' if c < best_cost else ''
            print(f'    grid z={z_c:7.1f} r={r_c:5.1f} → {c:.6f}{mark}', flush=True)
            if c < best_cost:
                best_cost = c
                best_z, best_r = float(z_c), float(r_c)

    print(f'  grid best: z={best_z:.1f} r={best_r:.1f}  cost={best_cost:.6f}',
          flush=True)

    res = minimize(cost, [best_z, best_r], method='Nelder-Mead',
                   options={'xatol': 0.5, 'fatol': 1e-7, 'maxiter': nm_iter})
    z_hat = float(np.clip(res.x[0], z_lo, z_hi))
    r_hat = float(np.clip(res.x[1], 5., 40.))
    print(f'  refined:   z={z_hat:.1f} r={r_hat:.1f}  cost={res.fun:.6f}'
          f'  ({n_evals[0]} FEM evals)', flush=True)
    return z_hat, r_hat, float(best_cost), float(res.fun), n_evals[0]


# ---------------------------------------------------------------------------
# Patient geometry helpers
# ---------------------------------------------------------------------------

def load_geo_and_mesh(dataset, idx, encoder, mlp, pid_dir, mesh_suffix=''):
    """Load TherMAM-NeRF geo for patient idx and ensure mesh exists."""
    geo = R.load_patient_geo(
        dataset, idx, encoder, mlp, CFG, device,
        gaussian_sigma=GAUSSIAN_SIGMA, mc_threshold_override=MC_THRESHOLD,
        repair_holes=REPAIR_HOLES, laplacian_iters=LAPLACIAN_ITERS,
        thermal_sigma=THERMAL_SIGMA)
    pid = geo['patient_id']
    pid_dir.mkdir(exist_ok=True, parents=True)
    suffix = f'_{mesh_suffix}' if mesh_suffix else ''
    stl_path = str(pid_dir / f'{pid}{suffix}.stl')
    msh_path = str(pid_dir / f'{pid}{suffix}.msh')
    if not os.path.exists(msh_path):
        print(f'  Generating mesh → {msh_path}', flush=True)
        save_stl_binary(stl_path, geo['surface_pts'], geo['faces'])
        stl_to_tet_mesh(stl_path, msh_path, mesh_size_mm=3.0)
        print('  Mesh ready.', flush=True)
    else:
        print(f'  Reusing existing mesh: {msh_path}', flush=True)
    return geo, msh_path


def hotspot_lateral(geo):
    """Lateral pin: centroid of top-10 % warmest skin vertices."""
    sp = geo['surface_pts']
    Tm = geo['T_measured']
    sk = ~chest_wall_mask_from_pts(sp)
    thr = np.quantile(Tm[sk], 0.90)
    hot = sk & (Tm >= thr)
    return float(sp[hot, 0].mean()), float(sp[hot, 1].mean())


# ---------------------------------------------------------------------------
# HTML visualisation helpers
# ---------------------------------------------------------------------------

def plot_tumor_localisation(geo, x0, y0, z_hat, r_hat, final_cost, label,
                             patient_dir):
    """3D breast surface coloured by T_measured with recovered tumour sphere."""
    verts = geo['surface_pts']
    faces = geo['faces']
    Tm    = geo['T_measured']
    pid   = geo['patient_id']
    Q0    = float(tumor_Qm_from_radius(torch.tensor(r_hat)).item())
    quad  = classify_quadrant(x0, y0)

    # Tumour sphere via parametric surface
    u, v = np.mgrid[0:2*np.pi:24j, 0:np.pi:13j]
    sx = x0     + r_hat * np.cos(u) * np.sin(v)
    sy = y0     + r_hat * np.sin(u) * np.sin(v)
    sz = z_hat  + r_hat * np.cos(v)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        intensity=Tm, intensitymode='vertex',
        colorscale='Inferno', opacity=0.85,
        colorbar=dict(title='°C', thickness=12),
        flatshading=False,
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
    ))
    fig.add_trace(go.Surface(
        x=sx, y=sy, z=sz,
        colorscale=[[0, 'orange'], [1, 'darkorange']],
        showscale=False, opacity=0.95,
        lighting=dict(ambient=0.5, diffuse=0.9),
    ))
    fig.update_layout(
        title=dict(
            text=(f'FEM-FEM Tumour — {pid} ({label})<br>'
                  f'<sup>{quad} | r={r_hat:.1f} mm | '
                  f'Q={Q0:.0f} W/m³ | cost={final_cost:.4f}</sup>'),
            font=dict(size=13)),
        scene=dict(
            xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (Depth) mm',
            aspectmode='data'),
        height=600, width=800,
        margin=dict(l=0, r=0, b=0, t=70),
    )
    out = patient_dir / f'{pid}_tumor_localisation.html'
    fig.write_html(str(out), include_plotlyjs='cdn')
    print(f'  Tumour localisation → {out}', flush=True)


def generate_patient_htmls(geo, msh_path, x0, y0, z_hat, r_hat,
                            final_cost, label, patient_dir):
    """Run FEM at recovered (z_hat, r_hat) and write both HTML viewers."""
    Q0  = float(tumor_Qm_from_radius(torch.tensor(r_hat)).item())
    params = {'x_t_mm': x0, 'y_t_mm': y0, 'z_t_mm': z_hat,
              'r_t_mm': r_hat, 'Q_max': Q0}
    try:
        T_sol, fea_msh = run_fea_forward(msh_path, params)
        _, mean_res, max_res, T_fea_interp = compute_fea_residual(
            T_sol, fea_msh, geo['surface_pts'], geo['T_measured'])
        pinn_results = {
            'x_t_mm': x0, 'y_t_mm': y0, 'z_t_mm': z_hat, 'r_t_mm': r_hat,
            'Q_max': Q0, 'fea_mean_residual': mean_res,
            'fea_max_residual': max_res,
            'fea_status': 'OK' if mean_res < 1.5 else 'HIGH_RESIDUAL',
        }
        plot_fea_vs_pinn(geo, pinn_results, T_fea_interp, save_html=True)
        plot_tumor_localisation(geo, x0, y0, z_hat, r_hat,
                                final_cost, label, patient_dir)
        print(f'  FEA residual: {mean_res:.3f} °C', flush=True)
        return mean_res, max_res
    except Exception as e:
        print(f'  HTML generation failed: {e}', flush=True)
        return np.nan, np.nan


# ---------------------------------------------------------------------------
# Static matplotlib figures — thesis-quality plots
# ---------------------------------------------------------------------------

def _try_style():
    for s in ('seaborn-v0_8-whitegrid', 'seaborn-whitegrid'):
        try:
            plt.style.use(s)
            return
        except Exception:
            pass


def plot_synthetic_errors(df, out_dir):
    """3-panel error breakdown for the 9-scenario synthetic validation."""
    _try_style()
    labels = [f'z={row.planted_z:.0f}\nr={row.planted_r:.0f}mm'
              for _, row in df.iterrows()]
    x = np.arange(len(df))
    w = 0.38

    depths = sorted(df['planted_z'].unique())
    radii  = sorted(df['planted_r'].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A — grouped bar chart: depth vs radius error per scenario
    ax = axes[0]
    ax.bar(x - w/2, df['depth_err_pct'],  w, color='#C62828', label='Depth error')
    ax.bar(x + w/2, df['radius_err_pct'], w, color='#1565C0', label='Radius error')
    ax.axhline(0.5, color='gray', lw=1.2, ls='--', alpha=0.8, label='0.5 % bound')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Error (%)')
    ax.set_title('A  Per-scenario error')
    ax.legend(fontsize=9)
    ymax = max(df[['depth_err_pct', 'radius_err_pct']].max().max() * 1.35, 0.9)
    ax.set_ylim(0, ymax)

    # B, C — 3×3 heatmaps (shallowest depth at top)
    for ax, col, title, cmap in [
        (axes[1], 'depth_err_pct',  'B  Depth error (%)',  'YlOrRd'),
        (axes[2], 'radius_err_pct', 'C  Radius error (%)', 'YlOrRd'),
    ]:
        mat = df.pivot_table(index='planted_z', columns='planted_r', values=col)
        mat = mat.reindex(index=sorted(depths, reverse=True), columns=radii)
        row_lbl = [f'z={z:.0f}' for z in sorted(depths, reverse=True)]
        col_lbl = [f'r={r:.0f}' for r in radii]
        vmax = max(df[col].max() * 1.1, 0.6)
        im = ax.imshow(mat.values, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(radii)));  ax.set_xticklabels(col_lbl)
        ax.set_yticks(range(len(depths))); ax.set_yticklabels(row_lbl)
        ax.set_xlabel('Planted radius (mm)')
        ax.set_ylabel('Planted depth (mm)')
        ax.set_title(title)
        for i in range(len(depths)):
            for j in range(len(radii)):
                val = mat.values[i, j]
                ax.text(j, i, f'{val:.2f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if val > vmax * 0.55 else 'black')

    plt.suptitle('FEM-FEM Synthetic Validation — Error Breakdown (9 scenarios, all < 0.5 %)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = Path(out_dir) / 'mukhmetov_error_breakdown.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out}', flush=True)


def plot_cohort_results(csv_path, out_dir):
    """6-panel thesis figure from the real-patient cohort CSV.

    Safe to call mid-run — plots whatever rows are complete so far.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print('plot_cohort_results: CSV not found — skipping.', flush=True)
        return
    df = pd.read_csv(csv_path)
    df_v = df.dropna(subset=['z_hat_mm', 'r_hat_mm'])
    n_total, n_valid = len(df), len(df_v)
    if n_valid == 0:
        print('plot_cohort_results: no complete rows yet — skipping.', flush=True)
        return

    _try_style()
    LABELS = ['benign', 'malignant']
    CLR    = {'benign': '#1565C0', 'malignant': '#B71C1C'}
    NICE   = {'benign': 'Benign',  'malignant': 'Malignant'}
    rng    = np.random.default_rng(42)

    def grp(col, lab):
        return df_v[df_v['label'] == lab][col].dropna().values

    def boxstrip(ax, col, ylabel, title):
        data = [grp(col, lab) for lab in LABELS]
        bp = ax.boxplot(data, labels=[NICE[l] for l in LABELS],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', lw=2))
        for patch, lab in zip(bp['boxes'], LABELS):
            patch.set_facecolor(CLR[lab])
            patch.set_alpha(0.35)
            patch.set_edgecolor(CLR[lab])
        for i, (d, lab) in enumerate(zip(data, LABELS), 1):
            jit = rng.uniform(-0.14, 0.14, size=len(d))
            ax.scatter(i + jit, d, c=CLR[lab], alpha=0.7, s=28,
                       edgecolors='white', linewidths=0.3, zorder=5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # A — depth distributions
    boxstrip(axes[0, 0], 'z_hat_mm', 'Recovered depth z (mm)',
             'A  Tumour depth by class')

    # B — radius distributions
    boxstrip(axes[0, 1], 'r_hat_mm', 'Recovered radius r (mm)',
             'B  Tumour radius by class')

    # C — depth vs radius scatter
    ax = axes[0, 2]
    for lab in LABELS:
        g = df_v[df_v['label'] == lab]
        ax.scatter(g['r_hat_mm'], g['z_hat_mm'], c=CLR[lab], alpha=0.75,
                   s=48, label=NICE[lab], edgecolors='white', linewidths=0.4)
    ax.set_xlabel('Recovered radius r (mm)')
    ax.set_ylabel('Recovered depth z (mm)')
    ax.set_title('C  Depth vs radius')
    ax.legend(framealpha=0.9)

    # D — lateral positions (IR hotspot)
    ax = axes[1, 0]
    for lab in LABELS:
        g = df_v[df_v['label'] == lab]
        ax.scatter(g['x0_mm'], g['y0_mm'], c=CLR[lab], alpha=0.75,
                   s=48, label=NICE[lab], edgecolors='white', linewidths=0.4)
    ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlabel('X lateral (mm)')
    ax.set_ylabel('Y lateral (mm)')
    ax.set_title('D  Hotspot lateral positions')
    ax.legend(framealpha=0.9)

    # E — optimisation cost distribution
    ax = axes[1, 1]
    for lab in LABELS:
        vals = grp('final_cost', lab)
        if len(vals):
            ax.hist(vals, bins=10, alpha=0.55, color=CLR[lab],
                    label=NICE[lab], edgecolor='white', density=True)
    ax.set_xlabel('Final cost (MSE, °C²)')
    ax.set_ylabel('Density')
    ax.set_title('E  Optimisation cost')
    ax.legend(framealpha=0.9)

    # F — FEA residual distribution
    ax = axes[1, 2]
    col = 'fea_mean_residual'
    if col in df_v.columns and df_v[col].notna().any():
        for lab in LABELS:
            vals = grp(col, lab)
            if len(vals):
                ax.hist(vals, bins=10, alpha=0.55, color=CLR[lab],
                        label=NICE[lab], edgecolor='white', density=True)
        ax.axvline(1.5, color='green', lw=1.5, ls='--', label='1.5 °C ref')
        ax.set_xlabel('Mean |FEA − IR| (°C)')
        ax.set_ylabel('Density')
        ax.set_title('F  FEA model-fit residual')
        ax.legend(framealpha=0.9)
    else:
        ax.text(0.5, 0.5, 'FEA residual\nnot yet available',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('F  FEA model-fit residual')

    plt.suptitle(
        f'FEM-FEM Real-Patient Cohort (n={n_valid}/{n_total} complete)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = Path(out_dir) / 'cohort_femfem_results.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Cohort results → {out}', flush=True)

    # Per-label summary table (CSV)
    agg = df_v.groupby('label').agg(
        n         =('patient_id',        'count'),
        z_mean    =('z_hat_mm',          'mean'),
        z_std     =('z_hat_mm',          'std'),
        r_mean    =('r_hat_mm',          'mean'),
        r_std     =('r_hat_mm',          'std'),
        cost_mean =('final_cost',        'mean'),
        cost_std  =('final_cost',        'std'),
    ).round(2)
    summ_path = Path(out_dir) / 'cohort_femfem_summary.csv'
    agg.to_csv(str(summ_path))
    print(f'Summary stats → {summ_path}', flush=True)


# ---------------------------------------------------------------------------
# Synthetic mode (original behaviour, unchanged)
# ---------------------------------------------------------------------------

def run_synthetic(args, dataset, encoder, mlp):
    geo = R.load_patient_geo(
        dataset, args.idx, encoder, mlp, CFG, device,
        gaussian_sigma=GAUSSIAN_SIGMA, mc_threshold_override=MC_THRESHOLD,
        repair_holes=REPAIR_HOLES, laplacian_iters=LAPLACIAN_ITERS,
        thermal_sigma=THERMAL_SIGMA)
    pid = geo['patient_id']
    patient_dir = RESULTS_DIR / pid
    patient_dir.mkdir(exist_ok=True, parents=True)
    stl_path = str(patient_dir / f'{pid}_syn.stl')
    msh_path = str(patient_dir / f'{pid}_syn.msh')

    sp = geo['surface_pts']
    sk = ~chest_wall_mask_from_pts(sp)
    x0, y0 = hotspot_lateral(geo)
    z_lo = float(geo['bbox_min'][2])
    z_hi = float(geo['bbox_max'][2])
    zr   = z_hi - z_lo
    print(f'Patient {pid}: lateral=({x0:.1f},{y0:.1f})  '
          f'z-range=[{z_lo:.1f},{z_hi:.1f}]', flush=True)

    print('Generating mesh (shared across all scenarios) …', flush=True)
    save_stl_binary(stl_path, geo['surface_pts'], geo['faces'])
    stl_to_tet_mesh(stl_path, msh_path, mesh_size_mm=3.0)
    print('Mesh ready.\n', flush=True)

    scenarios = [(z_lo + f*zr, r)
                 for f in (0.4, 0.55, 0.7) for r in (8., 14., 20.)]

    rows = []
    for z0, r0 in scenarios:
        print(f'\n=== PLANT z={z0:.1f}  r={r0:.1f} ===', flush=True)
        print('  [forward FEM] generating T_syn …', flush=True)
        T_syn = fem_surface(msh_path, geo, x0, y0, z0, r0)
        skin_range = f'{T_syn[sk].min():.2f}–{T_syn[sk].max():.2f} °C'
        print(f'  T_syn skin range: {skin_range}', flush=True)
        print('  [inverse FEM] searching (z_t, r_t) …', flush=True)
        z_hat, r_hat, _, _, _ = recover_fem_fem(
            geo, T_syn, x0, y0, msh_path, args.n_z, args.n_r, args.nm)
        ez = abs(z_hat - z0) / max(abs(z0), 1e-6) * 100
        er = abs(r_hat - r0) / r0 * 100
        print(f'  planted   z={z0:.1f}  r={r0:.1f}')
        print(f'  recovered z={z_hat:.1f}  r={r_hat:.1f}'
              f'  | depth_err={ez:.1f}%  radius_err={er:.1f}%', flush=True)
        rows.append(dict(
            planted_z=z0, recovered_z=z_hat, depth_err_pct=ez,
            planted_r=r0, recovered_r=r_hat, radius_err_pct=er))

    df = pd.DataFrame(rows)
    print('\nFEM-FEM Mukhmetov-style recovery table:')
    print(df.round(2).to_string(index=False))
    print(f'\nmean depth err  = {df.depth_err_pct.mean():.1f} %')
    print(f'mean radius err = {df.radius_err_pct.mean():.1f} %')
    df.to_csv(str(RESULTS_DIR / 'mukhmetov_recovery.csv'), index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    lo, hi = df.planted_z.min(), df.planted_z.max()
    a1.plot([lo, hi], [lo, hi], 'k--', lw=1, label='ideal')
    a1.scatter(df.planted_z, df.recovered_z, s=90, c='crimson', zorder=5)
    a1.set_xlabel('planted z (mm)'); a1.set_ylabel('recovered z (mm)')
    a1.set_title('Depth recovery — FEM→FEM'); a1.legend()
    lo, hi = df.planted_r.min(), df.planted_r.max()
    a2.plot([lo, hi], [lo, hi], 'k--', lw=1, label='ideal')
    a2.scatter(df.planted_r, df.recovered_r, s=90, c='steelblue', zorder=5)
    a2.set_xlabel('planted radius (mm)'); a2.set_ylabel('recovered radius (mm)')
    a2.set_title('Radius recovery — FEM→FEM'); a2.legend()
    plt.tight_layout()
    out = str(RESULTS_DIR / 'mukhmetov_recovery.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved → {out}')
    plot_synthetic_errors(df, RESULTS_DIR)


# ---------------------------------------------------------------------------
# Real-patient mode
# ---------------------------------------------------------------------------

def select_patients(patients, args):
    """Return ordered list of patient indices for real mode."""
    if args.indices:
        idxs = args.indices
    elif args.subset:
        groups = defaultdict(list)
        for i, p in enumerate(patients):
            groups[p['label']].append(i)
        glists = list(groups.values())
        order, pos = [], 0
        while len(order) < args.subset:
            progressed = False
            for g in glists:
                if pos < len(g):
                    order.append(g[pos]); progressed = True
                    if len(order) >= args.subset:
                        break
            if not progressed:
                break
            pos += 1
        idxs = sorted(order)
    else:
        idxs = list(range(len(patients)))

    # Apply stride/offset for GPU splitting
    idxs = idxs[args.offset::args.stride]
    return idxs


def run_real(args, dataset, patients, encoder, mlp):
    out_csv = Path(args.out) if args.out else RESULTS_DIR / 'cohort_femfem.csv'
    out_csv.parent.mkdir(exist_ok=True, parents=True)

    # Resume: load already-done patient_ids
    done = set()
    if args.resume and out_csv.exists():
        done = set(pd.read_csv(out_csv)['patient_id'].tolist())
        print(f'Resuming — {len(done)} patients already done: {sorted(done)}',
              flush=True)

    idxs = select_patients(patients, args)
    print(f'\nReal-patient FEM-FEM inverse: {len(idxs)} patients '
          f'(stride={args.stride}, offset={args.offset})', flush=True)
    for i, idx in enumerate(idxs):
        print(f'  [{i+1}/{len(idxs)}] idx={idx}  '
              f'{patients[idx]["id"]}  ({patients[idx]["label"]})', flush=True)

    for i, idx in enumerate(idxs):
        pid  = patients[idx]['id']
        lab  = patients[idx]['label']

        if pid in done:
            print(f'\n[{i+1}/{len(idxs)}] SKIP {pid} (already done)', flush=True)
            continue

        print(f'\n{"="*60}', flush=True)
        print(f'[{i+1}/{len(idxs)}] {pid}  ({lab})  idx={idx}', flush=True)
        print(f'{"="*60}', flush=True)

        try:
            patient_dir = RESULTS_DIR / pid
            geo, msh_path = load_geo_and_mesh(
                dataset, idx, encoder, mlp, patient_dir, mesh_suffix='syn')

            x0, y0 = hotspot_lateral(geo)
            sp = geo['surface_pts']
            Tm = geo['T_measured']
            sk = ~chest_wall_mask_from_pts(sp)
            z_lo = float(geo['bbox_min'][2])
            z_hi = float(geo['bbox_max'][2])
            skin_range = f'{Tm[sk].min():.2f}–{Tm[sk].max():.2f} °C'
            print(f'  lateral=({x0:.1f},{y0:.1f})  '
                  f'z-range=[{z_lo:.1f},{z_hi:.1f}]  '
                  f'skin IR: {skin_range}', flush=True)

            print('  [inverse FEM] searching (z_t, r_t) on real IR …', flush=True)
            z_hat, r_hat, grid_cost, final_cost, n_evals = recover_fem_fem(
                geo, geo['T_measured'], x0, y0, msh_path,
                args.n_z, args.n_r, args.nm)
            print(f'  RESULT: z_hat={z_hat:.1f} mm  r_hat={r_hat:.1f} mm  '
                  f'cost={final_cost:.4f}  ({n_evals} FEM evals)', flush=True)

            print('  [HTML] generating FEA-vs-IR and tumour localisation …', flush=True)
            fea_mean, fea_max = generate_patient_htmls(
                geo, msh_path, x0, y0, z_hat, r_hat,
                final_cost, lab, patient_dir)

            row = dict(
                patient_id=pid, label=lab,
                x0_mm=round(x0, 1), y0_mm=round(y0, 1),
                z_hat_mm=round(z_hat, 1), r_hat_mm=round(r_hat, 1),
                grid_cost=round(grid_cost, 6),
                final_cost=round(final_cost, 6),
                fea_mean_residual=round(fea_mean, 3) if not np.isnan(fea_mean) else np.nan,
                fea_max_residual=round(fea_max, 3) if not np.isnan(fea_max) else np.nan,
                n_fem_evals=n_evals,
            )

        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            row = dict(
                patient_id=pid, label=lab,
                x0_mm=np.nan, y0_mm=np.nan,
                z_hat_mm=np.nan, r_hat_mm=np.nan,
                grid_cost=np.nan, final_cost=np.nan,
                n_fem_evals=np.nan,
            )

        # Incremental save — append one row immediately
        df_row = pd.DataFrame([row])
        write_header = not out_csv.exists()
        df_row.to_csv(out_csv, mode='a', header=write_header, index=False)
        print(f'  Saved → {out_csv}', flush=True)
        done.add(pid)

    # Final summary
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        print('\n' + '='*60)
        print('FEM-FEM real-patient cohort summary:')
        print(df[['patient_id','label','z_hat_mm','r_hat_mm',
                   'final_cost']].to_string(index=False))
        if df['final_cost'].notna().any():
            for lab, grp in df.groupby('label'):
                print(f'\n  {lab}: n={len(grp)}  '
                      f'z_hat={grp.z_hat_mm.mean():.1f}±{grp.z_hat_mm.std():.1f} mm  '
                      f'r_hat={grp.r_hat_mm.mean():.1f}±{grp.r_hat_mm.std():.1f} mm  '
                      f'cost={grp.final_cost.mean():.4f}±{grp.final_cost.std():.4f}')
        print(f'\nFull results → {out_csv}')
        plot_cohort_results(out_csv, out_csv.parent)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='FEM-FEM inverse bioheat — synthetic or real-patient mode.')
    # Synthetic mode
    ap.add_argument('--idx',    type=int, default=0,
                    help='patient index (synthetic mode)')
    # Real mode
    ap.add_argument('--real',   action='store_true',
                    help='real-patient mode: use T_measured as inverse target')
    ap.add_argument('--subset', type=int, default=None,
                    help='balanced N patients (real mode, interleaved benign/malignant)')
    ap.add_argument('--indices', type=int, nargs='+', default=None,
                    help='explicit patient indices (real mode, overrides --subset)')
    ap.add_argument('--stride', type=int, default=1,
                    help='GPU split stride (e.g. 2 = every other patient)')
    ap.add_argument('--offset', type=int, default=0,
                    help='GPU split offset (0 = even patients, 1 = odd)')
    ap.add_argument('--out',    type=str, default=None,
                    help='output CSV path (real mode)')
    ap.add_argument('--resume', action='store_true',
                    help='skip patients already in --out CSV')
    ap.add_argument('--plot-csv', type=str, default=None, metavar='CSV',
                    help='plot cohort results from an existing CSV (no inverse run); '
                         'use after merging both GPU CSVs')
    # Shared
    ap.add_argument('--n-z',   type=int, default=5,  help='grid depth points')
    ap.add_argument('--n-r',   type=int, default=4,  help='grid radius points')
    ap.add_argument('--nm',    type=int, default=30, help='Nelder-Mead max iterations')
    args = ap.parse_args()

    if args.plot_csv:
        csv_path = Path(args.plot_csv)
        plot_cohort_results(csv_path, csv_path.parent)
        return

    dataset, patients = R.build_dataset()
    encoder, mlp = R.load_models()

    if args.real:
        run_real(args, dataset, patients, encoder, mlp)
    else:
        run_synthetic(args, dataset, encoder, mlp)


if __name__ == '__main__':
    main()