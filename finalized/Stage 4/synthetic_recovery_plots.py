#!/usr/bin/env python
"""
Synthetic inverse-recovery figures for the thesis.
Shows planted vs recovered tumour centres in 3D + orthographic front/side/top
views, plus per-quadrant recovery-error metrics.

Input : Stage 4/lm_synthetic_20260623_010523.csv
Output: Stage 4/figs/syn_recovery_multiview.png
        Stage 4/figs/syn_recovery_metrics.png
Run   : conda activate bioheat && python "Stage 4/synthetic_recovery_plots.py"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV  = HERE / 'lm_synthetic_20260623_010523.csv'
OUT  = HERE / 'figs'; OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

s = pd.read_csv(CSV)
SUCC = 5.0   # mm — position-error success threshold
s['ok'] = s.pos_err_mm < SUCC
QC = {'UI': '#2c7fb8', 'UO': '#d7301f', 'C': '#238b45', 'LI': '#984ea3', 'LO': '#ff7f00'}


def _lims(a, pad=20):
    """Axis limits from planted coords (physical breast region) + margin."""
    v = np.r_[s[f'planted_{a}'].values]
    return v.min() - pad, v.max() + pad


def _ortho(ax, a, b, title, xl, yl):
    """Scatter planted (o) and recovered (x) with error vector, 2 axes a,b.
    Axes clipped to the planted (physical) region — divergent recoveries
    leave the frame, which is honest about the UO ill-posedness."""
    for _, r in s.iterrows():
        c = QC.get(r.quad, 'gray')
        px, py = r[f'planted_{a}'], r[f'planted_{b}']
        rx, ry = r[f'recovered_{a}'], r[f'recovered_{b}']
        ax.plot([px, rx], [py, ry], '-', color=c, alpha=0.35, lw=0.8, zorder=1)
        ax.scatter(px, py, s=45, facecolor=c, edgecolor='k', lw=0.5, zorder=3)
        ax.scatter(rx, ry, s=45, marker='x', color=c, lw=1.4, zorder=3)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=11)
    ax.set_xlim(*_lims(a)); ax.set_ylim(*_lims(b))
    ax.set_aspect('equal', adjustable='box')


def fig_multiview():
    fig = plt.figure(figsize=(13.5, 4.6))
    # 3D
    ax0 = fig.add_subplot(1, 4, 1, projection='3d')
    for _, r in s.iterrows():
        c = QC.get(r.quad, 'gray')
        ax0.plot([r.planted_x, r.recovered_x], [r.planted_y, r.recovered_y],
                 [r.planted_z, r.recovered_z], '-', color=c, alpha=0.35, lw=0.8)
        ax0.scatter(r.planted_x, r.planted_y, r.planted_z, s=35,
                    facecolor=c, edgecolor='k', lw=0.4)
        ax0.scatter(r.recovered_x, r.recovered_y, r.recovered_z, s=35,
                    marker='x', color=c, lw=1.2)
    ax0.set_xlabel('x (mm)'); ax0.set_ylabel('y (mm)'); ax0.set_zlabel('z (mm)')
    ax0.set_title('(a) 3D', fontsize=11)
    ax0.set_xlim(*_lims('x')); ax0.set_ylim(*_lims('y')); ax0.set_zlim(*_lims('z'))
    ax0.view_init(elev=18, azim=-60)

    _ortho(fig.add_subplot(1, 4, 2), 'x', 'y',
           '(b) Front view (x–y)', 'x lateral (mm)', 'y supero-inferior (mm)')
    _ortho(fig.add_subplot(1, 4, 3), 'z', 'y',
           '(c) Side view (z–y)', 'z depth (mm)', 'y supero-inferior (mm)')
    _ortho(fig.add_subplot(1, 4, 4), 'x', 'z',
           '(d) Top view (x–z)', 'x lateral (mm)', 'z depth (mm)')

    # shared legend
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markeredgecolor='k', label='Planted (ground truth)', markersize=8),
           Line2D([0], [0], marker='x', color='gray', label='Recovered (LM)',
                  markersize=8, lw=0)]
    leg += [Line2D([0], [0], marker='o', color='w', markerfacecolor=QC[q],
                   markeredgecolor='k', label=q, markersize=8)
            for q in ['UI', 'UO', 'C']]
    fig.legend(handles=leg, loc='lower center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle('Synthetic tumour recovery — planted vs LM-recovered centres '
                 f'(n = {len(s)} scenarios, 2 patients)', y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / 'syn_recovery_multiview.png')
    plt.close(fig)
    print('  saved figs/syn_recovery_multiview.png')


def fig_metrics():
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))

    # (a) position error per scenario, coloured by quadrant
    sv = s.sort_values('pos_err_mm').reset_index(drop=True)
    ax[0].bar(range(len(sv)), sv.pos_err_mm,
              color=[QC.get(q, 'gray') for q in sv.quad], edgecolor='k', lw=0.3)
    ax[0].axhline(SUCC, ls='--', c='k', lw=1, label=f'{SUCC:g} mm success')
    ax[0].set_yscale('symlog', linthresh=1)
    ax[0].set_ylabel('Position error (mm, symlog)')
    ax[0].set_xlabel('scenario (sorted)')
    ax[0].set_title('(a) Position-recovery error')
    ax[0].legend(frameon=False, fontsize=9)

    # (b) planted vs recovered diameter
    ax[1].scatter(s.planted_d, s.recovered_d, s=55,
                  color=[QC.get(q, 'gray') for q in s.quad], edgecolor='k', lw=0.5)
    lim = [0, max(s.planted_d.max(), 50)]
    ax[1].plot(lim, lim, ls='--', c='gray', lw=1)
    ax[1].set_xlim(0, 25); ax[1].set_ylim(0, 50)
    ax[1].set_xlabel('Planted diameter (mm)')
    ax[1].set_ylabel('Recovered diameter (mm)')
    ax[1].set_title('(b) Diameter recovery')

    # (c) per-quadrant summary (well-posed vs UO)
    grp = s.groupby('quad').agg(pos=('pos_err_mm', 'median'),
                                d=('d_err_mm', 'median'),
                                n=('pos_err_mm', 'size')).reindex(['UI', 'C', 'UO'])
    xx = np.arange(len(grp)); w = 0.38
    ax[2].bar(xx - w/2, grp.pos, w, color='#2c7fb8', edgecolor='k', lw=0.4,
              label='median pos err')
    ax[2].bar(xx + w/2, grp.d, w, color='#d7301f', edgecolor='k', lw=0.4,
              label='median d err')
    ax[2].set_xticks(xx)
    ax[2].set_xticklabels([f'{q}\n(n={int(n)})' for q, n in zip(grp.index, grp.n)])
    ax[2].set_ylabel('Median error (mm)')
    ax[2].set_title('(c) Error by quadrant')
    ax[2].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / 'syn_recovery_metrics.png')
    plt.close(fig)
    print('  saved figs/syn_recovery_metrics.png')


if __name__ == '__main__':
    print(f'Reading {CSV.name}')
    fig_multiview(); fig_metrics()
    # headline numbers
    wp = s[s.quad.isin(['UI', 'C', 'LI'])]
    print(f'\nMedian pos_err (all)     : {s.pos_err_mm.median():.2f} mm')
    print(f'Median d_err   (all)     : {s.d_err_mm.median():.2f} mm')
    print(f'UI quadrant pos_err mean : {s[s.quad=="UI"].pos_err_mm.mean():.2f} mm')
    print(f'Success (<5mm)           : {int(s.ok.sum())}/{len(s)}')
    print('Done.')