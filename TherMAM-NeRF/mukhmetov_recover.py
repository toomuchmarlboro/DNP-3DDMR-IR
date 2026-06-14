#!/usr/bin/env python
"""FEM-FEM synthetic recovery (Mukhmetov-style, self-consistent).

FEniCSx generates the surface temperature for a KNOWN tumour, then a
grid search + Nelder-Mead with FEniCSx as the forward model recovers the
planted (z_t, r_t).  FEM-vs-FEM comparison has ~0.001 degC numerical noise
vs a 0.67 degC tumour signal (SNR ~670:1).  The PINN failed because it could
only fit the BC to 7 degC RMSE (SNR ~0.09:1) — an 11x signal deficit.

Run (env `bioheat`):
    python -u mukhmetov_recover.py --idx 0 | tee mukhmetov_syn.log
Timing: ~12-42 FEM evals per scenario, ~1-3 h total for 9 scenarios.
Outputs: results/mukhmetov_recovery.csv + results/mukhmetov_recovery.png
"""
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import run_cohort as R
from run_cohort import (
    CFG, device, RESULTS_DIR,
    chest_wall_mask_from_pts, tumor_Qm_from_radius,
    save_stl_binary, stl_to_tet_mesh, run_fea_forward, compute_fea_residual,
    GAUSSIAN_SIGMA, MC_THRESHOLD, REPAIR_HOLES, LAPLACIAN_ITERS, THERMAL_SIGMA,
)


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

    Cost = skin-only MSE between candidate FEM surface T and T_target (the
    planted FEM surface T).  Chest-wall is Dirichlet 37 C in every solve so
    it carries no information; masking it out keeps the cost function clean.
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

    # Coarse grid: span 85 % of the depth range, 3 physiological radii
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
    return z_hat, r_hat


def main():
    ap = argparse.ArgumentParser(
        description='FEM-FEM synthetic recovery (Mukhmetov 2025 style).')
    ap.add_argument('--idx',    type=int, default=0,  help='patient index')
    ap.add_argument('--n-z',   type=int, default=5,  help='grid depth points')
    ap.add_argument('--n-r',   type=int, default=4,  help='grid radius points')
    ap.add_argument('--nm',    type=int, default=30, help='Nelder-Mead iterations')
    args = ap.parse_args()

    dataset, _ = R.build_dataset()
    encoder, mlp = R.load_models()
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

    sp = geo['surface_pts']; Tm = geo['T_measured']
    sk = ~chest_wall_mask_from_pts(sp)
    thr = np.quantile(Tm[sk], 0.90)
    hot = sk & (Tm >= thr)
    x0 = float(sp[hot, 0].mean())
    y0 = float(sp[hot, 1].mean())
    z_lo = float(geo['bbox_min'][2])
    z_hi = float(geo['bbox_max'][2])
    zr   = z_hi - z_lo
    print(f'Patient {pid}: lateral=({x0:.1f},{y0:.1f})  '
          f'z-range=[{z_lo:.1f},{z_hi:.1f}]', flush=True)

    # Build mesh ONCE — geometry is the same for every scenario
    print('Generating mesh (shared across all scenarios) …', flush=True)
    save_stl_binary(stl_path, geo['surface_pts'], geo['faces'])
    stl_to_tet_mesh(stl_path, msh_path, mesh_size_mm=3.0)
    print('Mesh ready.\n', flush=True)

    scenarios = [(z_lo + f*zr, r)
                 for f in (0.4, 0.55, 0.7) for r in (8., 14., 20.)]

    rows = []
    for z0, r0 in scenarios:
        print(f'\n=== PLANT z={z0:.1f}  r={r0:.1f} ===', flush=True)

        # Forward: FEM generates synthetic surface T for planted params
        print('  [forward FEM] generating T_syn …', flush=True)
        T_syn = fem_surface(msh_path, geo, x0, y0, z0, r0)
        skin_range = f'{T_syn[sk].min():.2f}–{T_syn[sk].max():.2f} °C'
        print(f'  T_syn skin range: {skin_range}', flush=True)

        # Inverse: FEM grid search + Nelder-Mead
        print('  [inverse FEM] searching (z_t, r_t) …', flush=True)
        z_hat, r_hat = recover_fem_fem(
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
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
