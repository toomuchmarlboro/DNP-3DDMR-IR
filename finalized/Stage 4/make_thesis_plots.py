#!/usr/bin/env python
"""
Thesis figure set for the LM FEM-inverse real-data subset.
Input : finalized/lm_results/lm_real_subset.csv
Output: finalized/lm_results/figs/*.png  (300 dpi, publication-ready)
Run   : conda activate bioheat && python finalized/lm_results/make_thesis_plots.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy import stats

HERE   = Path(__file__).resolve().parent
CSV    = HERE / 'lm_real_subset.csv'
OLDCSV = HERE.parent.parent / 'TherMAM-NeRF' / 'results' / 'cohort_BCfix_ALL.csv'
OUT    = HERE / 'figs'
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_BEN, C_MAL = '#2c7fb8', '#d7301f'   # benign blue, malignant red

df  = pd.read_csv(CSV)
ben = df[df.label == 'benign']
mal = df[df.label == 'malignant']
y   = (df.label == 'malignant').astype(int).values


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f'  saved {p.relative_to(HERE.parent.parent)}')


# ── Fig 1 — ROC curves (final_rms & recovered_d as malignancy scores) ─────────
def fig_roc():
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for col, lbl, c in [('final_rms', 'Final RMS', C_MAL),
                        ('recovered_d', 'Recovered diameter', C_BEN)]:
        s = df[col].values
        auc = roc_auc_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=2, color=c, label=f'{lbl}  (AUC = {auc:.2f})')
    ax.plot([0, 1], [0, 1], ls=':', c='gray', lw=1.2, label='Chance (AUC = 0.50)')
    ax.set_xlabel('False-positive rate (1 − specificity)')
    ax.set_ylabel('True-positive rate (sensitivity)')
    ax.set_title('ROC — malignancy discrimination (n = 30)')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.set_aspect('equal')
    save(fig, 'fig1_roc.png')


# ── Fig 2 — distribution of final_rms & recovered_d by class ──────────────────
def _box_strip(ax, col, title, ylabel, unit):
    data = [ben[col].values, mal[col].values]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', lw=1.5))
    for patch, c in zip(bp['boxes'], [C_BEN, C_MAL]):
        patch.set_facecolor(c); patch.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for i, (d, c) in enumerate(zip(data, [C_BEN, C_MAL])):
        ax.scatter(np.full(len(d), i) + rng.uniform(-0.12, 0.12, len(d)),
                   d, s=28, color=c, edgecolor='k', linewidth=0.4, zorder=3)
    t, p = stats.ttest_ind(ben[col], mal[col], equal_var=False)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Benign\n(n=15)', 'Malignant\n(n=15)'])
    ax.set_ylabel(f'{ylabel} ({unit})')
    ax.set_title(f'{title}\nWelch t-test  p = {p:.3f}', fontsize=10)


def fig_dist():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.6))
    _box_strip(axes[0], 'final_rms', 'Inverse-fit residual', 'Final RMS', '°C')
    _box_strip(axes[1], 'recovered_d', 'Recovered tumour size', 'Diameter d', 'mm')
    axes[1].axhline(35, ls='--', c='gray', lw=1, alpha=0.7)
    axes[1].text(1.45, 35, 'clamp 35mm', va='center', ha='right',
                 fontsize=8, color='gray')
    fig.tight_layout()
    save(fig, 'fig2_distributions.png')


# ── Fig 3 — feature scatter (d vs rms) coloured by class ──────────────────────
def fig_scatter():
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(ben.final_rms, ben.recovered_d, s=70, color=C_BEN,
               edgecolor='k', linewidth=0.5, label='Benign', zorder=3)
    ax.scatter(mal.final_rms, mal.recovered_d, s=70, color=C_MAL, marker='^',
               edgecolor='k', linewidth=0.5, label='Malignant', zorder=3)
    ax.axhline(35, ls='--', c='gray', lw=1, alpha=0.6)
    ax.text(df.final_rms.min(), 34.4, 'd clamp 35mm', fontsize=8, color='gray')
    ax.set_xlabel('Final RMS (°C)')
    ax.set_ylabel('Recovered diameter d (mm)')
    ax.set_title('Feature space — overlap of classes (n = 30)')
    ax.legend(frameon=False)
    save(fig, 'fig3_feature_scatter.png')


# ── Fig 4 — confusion matrix of the as-run verdict ────────────────────────────
def fig_confusion():
    pred = (df.verdict == 'MALIGNANT').astype(int).values
    cm = confusion_matrix(y, pred, labels=[1, 0])  # rows truth M,B ; cols pred M,B
    tp, fn = cm[0]; fp, tn = cm[1]
    sens = tp / (tp + fn); spec = tn / (tn + fp); acc = (tp + tn) / cm.sum()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred\nMalignant', 'Pred\nBenign'])
    ax.set_yticklabels(['True\nMalignant', 'True\nBenign'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=20,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_title(f'Confusion (as-run rule)\n'
                 f'sens={sens:.2f}  spec={spec:.2f}  acc={acc:.2f}', fontsize=10)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    save(fig, 'fig4_confusion.png')


# ── Fig 5 — per-patient overview (rms + n_solves) ─────────────────────────────
def fig_overview():
    d = df.sort_values(['label', 'final_rms']).reset_index(drop=True)
    colors = [C_MAL if l == 'malignant' else C_BEN for l in d.label]
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)
    ax[0].bar(range(len(d)), d.final_rms, color=colors, edgecolor='k', linewidth=0.3)
    ax[0].set_ylabel('Final RMS (°C)')
    ax[0].set_title('Per-patient inverse fit (sorted by class, then RMS)')
    ax[1].bar(range(len(d)), d.n_solves, color=colors, edgecolor='k', linewidth=0.3)
    ax[1].set_ylabel('FEM solves')
    ax[1].set_xticks(range(len(d)))
    ax[1].set_xticklabels(d.patient_id, rotation=90, fontsize=7)
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=C_BEN, label='Benign'),
                          Patch(color=C_MAL, label='Malignant')],
                 frameon=False, loc='upper left')
    fig.tight_layout()
    save(fig, 'fig5_per_patient.png')


# ── Fig 6 — OLD vs NEW method comparison (complete, 6 panels) ─────────────────
C_OLD, C_NEW = '#7b3294', '#008837'   # old purple, new green


def fig_method_compare():
    old = pd.read_csv(OLDCSV)
    m = df.merge(old, on='patient_id', suffixes=('_new', ''))
    ym = (m.label_new == 'malignant').astype(int).values
    benm = ym == 0
    rng = np.random.default_rng(1)

    fig = plt.figure(figsize=(13.5, 8.6))
    gs = GridSpec(2, 3, hspace=0.42, wspace=0.34)

    # (a) ROC head-to-head — best score of each method
    axA = fig.add_subplot(gs[0, 0])
    auc_old = roc_auc_score(ym, m.r_hat_mm)
    auc_new = roc_auc_score(ym, m.final_rms)
    fo, to, _ = roc_curve(ym, m.r_hat_mm)
    fn, tn, _ = roc_curve(ym, m.final_rms)
    axA.plot(fo, to, lw=2.2, color=C_OLD, label=f'OLD r_hat  (AUC {auc_old:.2f})')
    axA.plot(fn, tn, lw=2.2, color=C_NEW, label=f'NEW RMS  (AUC {auc_new:.2f})')
    axA.plot([0, 1], [0, 1], ls=':', c='gray', lw=1.2, label='Chance 0.50')
    axA.set_xlabel('False-positive rate'); axA.set_ylabel('True-positive rate')
    axA.set_title(f'(a) ROC head-to-head  (n = {len(m)})')
    axA.set_xlim(-0.02, 1.02); axA.set_ylim(-0.02, 1.02); axA.set_aspect('equal')
    axA.legend(loc='lower right', frameon=False, fontsize=8.5)

    # (b) AUC bar — every discriminative feature, both methods
    axB = fig.add_subplot(gs[0, 1])
    feats = [('OLD r_hat', m.r_hat_mm, C_OLD), ('OLD cost', m.final_cost, C_OLD),
             ('OLD resid', m.fea_mean_residual, C_OLD),
             ('NEW d', m.recovered_d, C_NEW), ('NEW RMS', m.final_rms, C_NEW)]
    names = [f[0] for f in feats]
    aucs = [roc_auc_score(ym, f[1]) for f in feats]
    cols = [f[2] for f in feats]
    axB.bar(names, aucs, color=cols, alpha=0.75, edgecolor='k', linewidth=0.4)
    axB.axhline(0.5, ls=':', c='gray', lw=1.2)
    axB.set_ylabel('AUC'); axB.set_ylim(0, 0.8)
    axB.set_title('(b) Discrimination by feature')
    axB.set_xticklabels(names, rotation=30, ha='right', fontsize=8.5)
    for i, a in enumerate(aucs):
        axB.text(i, a + 0.01, f'{a:.2f}', ha='center', fontsize=8)

    # (c) Residual reduction — paired box+strip
    axC = fig.add_subplot(gs[0, 2])
    data = [m.fea_mean_residual.values, m.final_rms.values]
    bp = axC.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True,
                     showfliers=False, medianprops=dict(color='black', lw=1.5))
    for patch, c in zip(bp['boxes'], [C_OLD, C_NEW]):
        patch.set_facecolor(c); patch.set_alpha(0.30)
    x0 = rng.uniform(-0.08, 0.08, len(m)); x1 = 1 + rng.uniform(-0.08, 0.08, len(m))
    for i in range(len(m)):
        axC.plot([x0[i], x1[i]], [data[0][i], data[1][i]],
                 color='gray', alpha=0.25, lw=0.7, zorder=1)
    axC.scatter(x0, data[0], s=24, color=C_OLD, edgecolor='k', lw=0.4, zorder=3)
    axC.scatter(x1, data[1], s=24, color=C_NEW, edgecolor='k', lw=0.4, zorder=3)
    t, p = stats.ttest_rel(data[0], data[1])
    axC.set_xticks([0, 1]); axC.set_xticklabels(['OLD\n(no mean-sub)', 'NEW\n(mean-sub)'])
    axC.set_ylabel('Residual (°C)'); axC.set_ylim(bottom=0)
    axC.set_title(f'(c) Residual  {data[0].mean():.2f}→{data[1].mean():.2f}°C\n'
                  f'paired p = {p:.1e}', fontsize=9.5)

    # (d) Rail behaviour — fraction at the size clamp
    axD = fig.add_subplot(gs[1, 0])
    old_rail = [ (m[benm].r_hat_mm >= 39.9).mean(), (m[~benm].r_hat_mm >= 39.9).mean() ]
    new_rail = [ (m[benm].recovered_d >= 34.9).mean(), (m[~benm].recovered_d >= 34.9).mean() ]
    xx = np.arange(2); w = 0.35
    axD.bar(xx - w/2, old_rail, w, color=C_OLD, alpha=0.75, edgecolor='k',
            linewidth=0.4, label='OLD r_hat @40mm')
    axD.bar(xx + w/2, new_rail, w, color=C_NEW, alpha=0.75, edgecolor='k',
            linewidth=0.4, label='NEW d @35mm')
    axD.set_xticks(xx); axD.set_xticklabels(['Benign', 'Malignant'])
    axD.set_ylabel('Fraction railed at clamp'); axD.set_ylim(0, 1.05)
    axD.set_title('(d) Size-parameter saturation')
    axD.legend(frameon=False, fontsize=8.5)

    # (e) Size-estimate agreement between methods
    axE = fig.add_subplot(gs[1, 1])
    od = m.r_hat_mm * 2
    axE.scatter(od[benm], m.recovered_d[benm], s=55, color=C_BEN,
                edgecolor='k', lw=0.5, label='Benign', zorder=3)
    axE.scatter(od[~benm], m.recovered_d[~benm], s=55, color=C_MAL, marker='^',
                edgecolor='k', lw=0.5, label='Malignant', zorder=3)
    lim = [0, 82]; axE.plot(lim, lim, ls='--', c='gray', lw=1, label='y = x')
    r = np.corrcoef(od, m.recovered_d)[0, 1]
    axE.set_xlim(lim); axE.set_ylim(0, 40)
    axE.set_xlabel('OLD diameter 2·r_hat (mm)')
    axE.set_ylabel('NEW diameter d (mm)')
    axE.set_title(f'(e) Size agreement  r = {r:.2f}')
    axE.legend(frameon=False, fontsize=8.5, loc='lower right')

    # (f) Per-class residual, both methods
    axF = fig.add_subplot(gs[1, 2])
    groups = [('OLD ben', m[benm].fea_mean_residual, C_OLD),
              ('OLD mal', m[~benm].fea_mean_residual, C_OLD),
              ('NEW ben', m[benm].final_rms, C_NEW),
              ('NEW mal', m[~benm].final_rms, C_NEW)]
    bp = axF.boxplot([g[1].values for g in groups], positions=range(4),
                     widths=0.55, patch_artist=True, showfliers=False,
                     medianprops=dict(color='black', lw=1.4))
    for patch, g in zip(bp['boxes'], groups):
        patch.set_facecolor(g[2]); patch.set_alpha(0.30)
    for i, g in enumerate(groups):
        xj = i + rng.uniform(-0.1, 0.1, len(g[1]))
        axF.scatter(xj, g[1].values, s=18, color=g[2], edgecolor='k', lw=0.3, zorder=3)
    axF.set_xticks(range(4)); axF.set_xticklabels([g[0] for g in groups],
                                                  rotation=20, fontsize=8.5)
    axF.set_ylabel('Residual (°C)'); axF.set_ylim(bottom=0)
    axF.set_title('(f) Residual by method × class')

    fig.suptitle('OLD (FEM grid+opt, no mean-sub) vs NEW (LM 4-DoF, mean-sub) — '
                 f'shared {len(m)} patients', fontsize=13, y=0.98)
    save(fig, 'fig6_method_compare.png')
    print(f'  [compare] OLD AUC(r_hat)={auc_old:.3f}  NEW AUC(rms)={auc_new:.3f}  '
          f'resid {data[0].mean():.2f}->{data[1].mean():.2f}C  paired p={p:.2e}  '
          f'size r={r:.2f}')


# ── summary numbers to stdout ────────────────────────────────────────────────
def summary():
    print('\nSummary (n=30, 15 benign / 15 malignant):')
    for col in ['final_rms', 'recovered_d']:
        print(f'  {col:13s}  benign {ben[col].mean():.2f}±{ben[col].std():.2f}   '
              f'malignant {mal[col].mean():.2f}±{mal[col].std():.2f}   '
              f'AUC {roc_auc_score(y, df[col]):.3f}')


if __name__ == '__main__':
    print(f'Reading {CSV.name}  →  {OUT}/')
    fig_roc(); fig_dist(); fig_scatter(); fig_confusion(); fig_overview()
    fig_method_compare()
    summary()
    print('Done.')