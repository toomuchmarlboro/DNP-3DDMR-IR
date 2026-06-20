#!/usr/bin/env python3
"""
β-VAE Discriminability Experiment  (strong version)
=====================================================
Trains a β-VAE on DMR-IR thermal images (135 patients: 106 benign, 29 malignant)
fully unsupervised, then asks: do the latent codes separate benign from malignant?

Improvements over v1:
  - ResNet encoder/decoder with GroupNorm + SiLU (no dead-ReLU, better gradients)
  - Self-attention at bottleneck (captures global breast-pattern relationships)
  - Input = masked thermal + mask (2 ch per view) → background noise zeroed
  - Cyclical KL annealing → fixes posterior collapse (dead latent dims)
  - Data augmentation: random horizontal flip + Gaussian noise
  - All 5 views × 2 ch = 10 ch input option (--views all)

Run (frontal only — fast):
    python vae_discriminability.py --epochs 300

Run (all 5 views — recommended for thesis):
    python vae_discriminability.py --epochs 300 --views all

Outputs → VAE_Discriminability/results/
    fig0_training_loss.png
    fig1_reconstruction.png
    fig2_umap.png
    fig3_pca.png
    fig4_classifier.png   ← LOO-CV ROC + confusion matrix
    fig5_latent_dims.png  ← per-dimension discriminability
    latent_codes.npz      ← μ, labels, patient_ids
"""
import argparse, math, pathlib, warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO     = pathlib.Path(__file__).resolve().parent.parent
TIFF_DIR = REPO / 'data' / 'organized_by_patient'
MASK_DIR = REPO / 'data' / 'organized_by_patient_unet'
OUT_DIR  = pathlib.Path(__file__).resolve().parent / 'results'
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 64

VIEW_KEYWORDS = {
    'RL': 'right later',
    'RO': 'right obli',
    'F':  'anterior',       # matches "anterior (front)"
    'LO': 'left obliq',
    'LL': 'left later',
}

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--epochs',     type=int,   default=300)
ap.add_argument('--latent-dim', type=int,   default=16)   # PCA showed ~2 effective dims; 16 gives headroom without KL dominance
ap.add_argument('--beta',       type=float, default=0.5)  # lighter KL penalty so recon loss can compete
ap.add_argument('--free-bits',  type=float, default=0.1,  # 0.1×16=1.6 nats floor, comparable to recon ~0.065
                help='min KL per latent dim (nats); prevents posterior collapse)')
ap.add_argument('--lr',         type=float, default=5e-4)
ap.add_argument('--batch-size', type=int,   default=16)
ap.add_argument('--views',      default='frontal',
                choices=['frontal', 'all'],
                help='frontal = 1 thermal view; all = 5 views')
ap.add_argument('--no-aug', action='store_true', help='disable data augmentation')
args = ap.parse_args()

USE_ALL   = (args.views == 'all')
VIEW_LIST = ['RL', 'RO', 'F', 'LO', 'LL'] if USE_ALL else ['F']
IN_CH     = len(VIEW_LIST)       # one thermal channel per view, no masks

print(f'β-VAE strong  latent={args.latent_dim}  β={args.beta}  '
      f'epochs={args.epochs}  in_ch={IN_CH}  aug={not args.no_aug}')

# ── Data discovery ─────────────────────────────────────────────────────────────
def get_view_key(stem: str) -> str | None:
    s = stem.lower()
    for vk, kw in VIEW_KEYWORDS.items():
        if kw in s:
            return vk
    return None


def discover():
    records = []
    for pid_dir in sorted(TIFF_DIR.iterdir()):
        if not pid_dir.is_dir():
            continue
        for lab_dir in pid_dir.iterdir():
            if not lab_dir.is_dir():
                continue
            label = 1 if 'malig' in lab_dir.name.lower() else 0
            tiffs = {}
            for f in lab_dir.glob('*.tiff'):
                vk = get_view_key(f.stem)
                if vk:
                    tiffs[vk] = f
            # only require thermal TIFFs — no mask dependency
            if all(vk in tiffs for vk in VIEW_LIST):
                records.append({'id': pid_dir.name, 'label': label, 'tiffs': tiffs})
    records.sort(key=lambda r: int(r['id'].split('_')[-1]) if '_' in r['id'] else r['id'])
    n_ben = sum(1 for r in records if r['label'] == 0)
    n_mal = sum(1 for r in records if r['label'] == 1)
    print(f'Patients: {len(records)} total  ({n_ben} benign, {n_mal} malignant)')
    return records


patients = discover()


def load_thermal(path: pathlib.Path) -> np.ndarray:
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    img = Image.fromarray(arr, mode='F').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-6)


def load_mask(path: pathlib.Path) -> np.ndarray:
    img = Image.open(str(path)).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return (np.array(img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)


class ThermalDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        channels = [load_thermal(r['tiffs'][vk]) for vk in VIEW_LIST]
        x = np.stack(channels, axis=0).astype(np.float32)   # (IN_CH, H, W)

        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()   # horizontal flip
            x += np.random.randn(*x.shape).astype(np.float32) * 0.02
            x = np.clip(x, 0.0, 1.0)

        return torch.tensor(x), r['label'], r['id']


train_ds = ThermalDataset(patients, augment=(not args.no_aug))
infer_ds = ThermalDataset(patients, augment=False)

# Natural class distribution — no oversampling/reweighting (purely the dataset)
train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
infer_dl = DataLoader(infer_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

# ── Model components ───────────────────────────────────────────────────────────
def gn(ch):
    g = min(8, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class ResDown(nn.Module):
    """Residual block with optional stride-2 downsampling."""
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            gn(out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            gn(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            gn(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


class ResUp(nn.Module):
    """Residual block with 2× upsample (avoids checkerboard from ConvTranspose2d)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            gn(out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            gn(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            gn(out_ch),
        )
        self.act  = nn.SiLU()

    def forward(self, x):
        x = self.up(x)
        return self.act(self.main(x) + self.skip(x))


class SelfAttn(nn.Module):
    """Lightweight multi-head self-attention over spatial positions."""
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm  = gn(ch)
        self.attn  = nn.MultiheadAttention(ch, heads, batch_first=True, dropout=0.0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).permute(0, 2, 1)   # B, HW, C
        h, _ = self.attn(h, h, h)
        return x + self.gamma * h.permute(0, 2, 1).reshape(B, C, H, W)


class Encoder(nn.Module):
    # 64 → 32 → 16 → 8 [attn] → 4 → μ, logvar
    def __init__(self, in_ch, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            ResDown(in_ch,  64, stride=2),   # 64→32
            ResDown(64,    128, stride=2),   # 32→16
            ResDown(128,   256, stride=2),   # 16→8
            SelfAttn(256, heads=4),
            ResDown(256,   512, stride=2),   # 8→4
        )
        self.fc_mu     = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        nn.init.zeros_(self.fc_logvar.weight)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    # latent → 4 → 8 [attn] → 16 → 32 → 64
    def __init__(self, in_ch, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.net = nn.Sequential(
            ResUp(512, 256),         # 4→8
            SelfAttn(256, heads=4),
            ResUp(256, 128),         # 8→16
            ResUp(128,  64),         # 16→32
            ResUp(64,   in_ch),      # 32→64
        )
        self.out = nn.Sigmoid()

    def forward(self, z):
        h = F.silu(self.fc(z)).view(-1, 512, 4, 4)
        return self.out(self.net(h))


class BetaVAE(nn.Module):
    def __init__(self, in_ch, latent_dim, beta):
        super().__init__()
        self.encoder = Encoder(in_ch, latent_dim)
        self.decoder = Decoder(in_ch, latent_dim)
        self.beta    = beta

    def reparameterise(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z    = self.reparameterise(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss(self, x, recon, mu, logvar, beta_eff, free_bits=0.5):
        # Plain MSE over all pixels — no mask, no background issues
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # Free-bits KL: each dim contributes at least free_bits nats,
        # preventing posterior collapse without needing low β.
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, D)
        kl_per_dim = kl_per_dim.mean(0)                               # (D,)
        kl         = torch.clamp(kl_per_dim, min=free_bits).sum()

        total = recon_loss + beta_eff * kl
        return total, recon_loss, kl_per_dim.mean()


# ── KL annealing (cyclical) ────────────────────────────────────────────────────
def kl_weight(epoch, total, n_cycles=2, ramp_frac=0.5):
    """Linearly ramp from 0→1 over first ramp_frac of each cycle, hold at 1.
    2 cycles over 300 epochs = 150 epoch cycles; ramp takes 75 epochs each time."""
    cycle_len = total / n_cycles
    pos = (epoch % cycle_len) / cycle_len
    return min(1.0, pos / ramp_frac)


# ── Training ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = BetaVAE(IN_CH, args.latent_dim, args.beta).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {n_params:,}')

opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)

hist_loss, hist_recon, hist_kl, hist_beta = [], [], [], []

for epoch in range(1, args.epochs + 1):
    model.train()
    ep_total, ep_recon, ep_kl = [], [], []
    b_eff = args.beta * kl_weight(epoch, args.epochs)

    for imgs, _, _ in train_dl:
        imgs = imgs.to(device)
        recon, mu, logvar = model(imgs)
        loss, rl, kl = model.loss(imgs, recon, mu, logvar, b_eff, args.free_bits)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ep_total.append(loss.item())
        ep_recon.append(rl.item())
        ep_kl.append(kl.item())

    sch.step()
    hist_loss.append(np.mean(ep_total))
    hist_recon.append(np.mean(ep_recon))
    hist_kl.append(np.mean(ep_kl))
    hist_beta.append(b_eff)

    if epoch % 50 == 0 or epoch == 1:
        print(f'Epoch {epoch:4d}/{args.epochs}  '
              f'Loss={np.mean(ep_total):.4f}  '
              f'Recon={np.mean(ep_recon):.4f}  '
              f'KL={np.mean(ep_kl):.3f}  '
              f'β_eff={b_eff:.2f}')

torch.save(model.state_dict(), OUT_DIR / 'vae_weights.pth')
print('Weights saved.')

# ── Encode all patients ────────────────────────────────────────────────────────
model.eval()
all_mu, all_lab, all_ids = [], [], []
with torch.no_grad():
    for imgs, labs, ids in infer_dl:
        mu, logvar = model.encoder(imgs.to(device))
        all_mu.append(mu.cpu().numpy())
        all_lab.extend(labs.numpy())
        all_ids.extend(ids)

Z   = np.concatenate(all_mu, axis=0)
y   = np.array(all_lab)
ids = np.array(all_ids)
np.savez(OUT_DIR / 'latent_codes.npz', Z=Z, y=y, ids=ids)

per_dim_var = Z.var(axis=0)
dead = (per_dim_var < 0.01).sum()
print(f'Z={Z.shape}  dead_dims={dead}/{args.latent_dim}  '
      f'var_mean={per_dim_var.mean():.4f}')

# ── Fig 0: Training curves ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(hist_loss,  color='#4C72B0'); axes[0].set_title('Total Loss');   axes[0].set_xlabel('Epoch'); axes[0].grid(True, alpha=0.3)
axes[1].plot(hist_recon, color='#55A868'); axes[1].set_title('Recon Loss');  axes[1].set_xlabel('Epoch'); axes[1].grid(True, alpha=0.3)
axes[2].plot(hist_kl,    color='#C44E52', label='KL');
ax2b = axes[2].twinx()
ax2b.plot(hist_beta, color='gray', linestyle='--', lw=1, label='β_eff')
ax2b.set_ylabel('β_eff', color='gray')
axes[2].set_title('KL divergence + β schedule'); axes[2].set_xlabel('Epoch'); axes[2].grid(True, alpha=0.3)
axes[2].legend(loc='upper left'); ax2b.legend(loc='upper right')
plt.suptitle(f'β-VAE Training  (latent={args.latent_dim}, β={args.beta})', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig0_training_loss.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig0_training_loss.png')

# ── Fig 1: Reconstructions ─────────────────────────────────────────────────────
model.eval()
n_show = min(8, len(patients))
with torch.no_grad():
    sample = torch.stack([infer_ds[i][0] for i in range(n_show)]).to(device)
    recons, _, _ = model(sample)

fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 5))
# Always show the frontal (F) view regardless of how many channels were used
f_ch = VIEW_LIST.index('F')   # ch 0 for frontal-only, ch 2 for all-views
for i in range(n_show):
    inp = sample[i, f_ch].cpu().numpy()   # raw frontal thermal
    rec = recons[i, f_ch].cpu().numpy()
    lbl = 'MAL' if patients[i]['label'] == 1 else 'BEN'
    axes[0, i].imshow(inp, cmap='inferno', vmin=0, vmax=1)
    axes[0, i].set_title(f'{patients[i]["id"]}\n({lbl})', fontsize=7)
    axes[0, i].axis('off')
    axes[1, i].imshow(rec, cmap='inferno', vmin=0, vmax=1)
    axes[1, i].set_title('Recon (frontal)', fontsize=7)
    axes[1, i].axis('off')
plt.suptitle(f'β-VAE Reconstructions — frontal view  '
             f'({"all 5 views" if USE_ALL else "frontal only"}, ch {f_ch})', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_reconstruction.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig1_reconstruction.png')

# ── Fig 2: UMAP ───────────────────────────────────────────────────────────────
print('Computing UMAP...')
n_nbrs = min(15, len(Z) - 1)
Z2 = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_nbrs).fit_transform(Z)

fig, ax = plt.subplots(figsize=(8, 7))
for lbl, c, mk in [('benign', '#4C72B0', 'o'), ('malignant', '#C44E52', '^')]:
    mask = (y == (1 if lbl == 'malignant' else 0))
    ax.scatter(Z2[mask, 0], Z2[mask, 1], c=c, label=f'{lbl} (n={mask.sum()})',
               marker=mk, alpha=0.8, edgecolors='white', linewidths=0.4, s=70)
ax.set_title(f'UMAP of β-VAE Latent Codes\n(latent={args.latent_dim}, β={args.beta}, '
             f'epochs={args.epochs})', fontsize=13)
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_umap.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig2_umap.png')

# ── Fig 3: PCA ────────────────────────────────────────────────────────────────
pca  = PCA(n_components=2, random_state=42)
Zpca = pca.fit_transform(Z)
vr   = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(8, 7))
for lbl, c, mk in [('benign', '#4C72B0', 'o'), ('malignant', '#C44E52', '^')]:
    mask = (y == (1 if lbl == 'malignant' else 0))
    ax.scatter(Zpca[mask, 0], Zpca[mask, 1], c=c, label=f'{lbl} (n={mask.sum()})',
               marker=mk, alpha=0.8, edgecolors='white', linewidths=0.4, s=70)
ax.set_title(f'PCA of β-VAE Latent Codes\nPC1={vr[0]*100:.1f}%  PC2={vr[1]*100:.1f}%', fontsize=13)
ax.set_xlabel(f'PC1 ({vr[0]*100:.1f}% var)'); ax.set_ylabel(f'PC2 ({vr[1]*100:.1f}% var)')
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig3_pca.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig3_pca.png')

# ── Fig 4: LOO-CV SVM on the real data only (no SMOTE, no oversampling) ───────
# Pure test of dataset discriminability: leave-one-out RBF-SVM on the latent
# codes, evaluated on the natural class distribution. class_weight='balanced'
# only reweights the SVM loss — it never fabricates samples.
print('Running LOO-CV SVM  (real data only)...')
loo = LeaveOneOut()

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', C=1.0, probability=True,
                   class_weight='balanced', random_state=42)),
])

y_prob = cross_val_predict(clf, Z, y, cv=loo, method='predict_proba')[:, 1]

def eval_clf(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return dict(auc=auc, fpr=fpr, tpr=tpr, cm=cm,
                sens=tp/(tp+fn), spec=tn/(tn+fp), tp=tp, fp=fp, tn=tn, fn=fn)

res = eval_clf(y, y_prob)

# MWU test on distance from benign centroid (model-free)
c_ben    = Z[y == 0].mean(0)
d_ben    = np.linalg.norm(Z[y == 0] - c_ben, axis=1)
d_mal    = np.linalg.norm(Z[y == 1] - c_ben, axis=1)
_, p_mwu = mannwhitneyu(d_ben, d_mal, alternative='less')

print(f'\n== LOO-CV Results (real data only) ==')
print(f'  RBF-SVM:  AUC={res["auc"]:.3f}  '
      f'Sens={res["sens"]:.3f}  Spec={res["spec"]:.3f}')
print(f'  MWU p={p_mwu:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC curve
axes[0].plot(res['fpr'], res['tpr'], color='#4C72B0', lw=2,
             label=f'RBF-SVM  AUC={res["auc"]:.3f}')
axes[0].plot([0,1],[0,1], 'k--', lw=1, label='Random')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('LOO-CV ROC — RBF-SVM on β-VAE Latent Codes\n(real data, no oversampling)')
axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)

# Confusion matrix
axes[1].imshow(res['cm'], cmap='Blues', vmin=0)
axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
axes[1].set_xticklabels(['Pred B','Pred M']); axes[1].set_yticklabels(['True B','True M'])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(res['cm'][i,j]), ha='center', va='center', fontsize=18,
                     color='white' if res['cm'][i,j] > res['cm'].max()/2 else 'black')
axes[1].set_title(f'RBF-SVM (class_weight=balanced)\n'
                  f'AUC={res["auc"]:.3f}  Sens={res["sens"]:.2f}  '
                  f'Spec={res["spec"]:.2f}\nMWU p={p_mwu:.4f}')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_classifier.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig4_classifier.png')

# Headline metrics
auc  = res['auc']
sens = res['sens']
spec = res['spec']
cm   = res['cm']

# ── Fig 5: Per-dimension discriminability ─────────────────────────────────────
t_abs, p_vals = [], []
for d in range(args.latent_dim):
    t, p = ttest_ind(Z[y==0, d], Z[y==1, d])
    t_abs.append(abs(t)); p_vals.append(p)
t_abs = np.array(t_abs); p_vals = np.array(p_vals)
order = np.argsort(t_abs)[::-1]
best  = order[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bar_colors = ['#C44E52' if p_vals[order[i]] < 0.05 else '#4C72B0'
              for i in range(args.latent_dim)]
axes[0].bar(range(args.latent_dim), t_abs[order], color=bar_colors)
axes[0].axhline(2.0, color='gray', linestyle='--', lw=1)
axes[0].set_xlabel('Latent dim (sorted by |t|)'); axes[0].set_ylabel('|t|')
axes[0].set_title(f'Per-dimension discriminability\nRed=p<0.05  '
                  f'({(p_vals<0.05).sum()}/{args.latent_dim} significant)')

parts = axes[1].violinplot([Z[y==0, best], Z[y==1, best]], positions=[0,1], showmedians=True)
parts['bodies'][0].set_facecolor('#4C72B0')
parts['bodies'][1].set_facecolor('#C44E52')
axes[1].set_xticks([0,1]); axes[1].set_xticklabels(['Benign','Malignant'])
axes[1].set_ylabel(f'z_{best}')
t_b, p_b = ttest_ind(Z[y==0, best], Z[y==1, best])
axes[1].set_title(f'Most discriminative dim: z_{best}\nt={t_b:.2f}  p={p_b:.4f}')
axes[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig5_latent_dims.png', dpi=150, bbox_inches='tight')
plt.close(); print('Saved fig5_latent_dims.png')

# ── Final summary ──────────────────────────────────────────────────────────────
n_sig = (p_vals < 0.05).sum()
print('\n' + '='*56)
print('SUMMARY')
print('='*56)
print(f'  Patients          : {len(patients)} ({( y==0).sum()} benign, {(y==1).sum()} malignant)')
print(f'  Input             : {args.views} view(s)  {IN_CH} ch  {IMG_SIZE}×{IMG_SIZE}')
print(f'  β-VAE             : latent={args.latent_dim}  β={args.beta}  epochs={args.epochs}')
print(f'  Dead latent dims  : {dead} / {args.latent_dim}')
print(f'  Significant dims  : {n_sig} / {args.latent_dim}  (t-test p<0.05)')
print(f'  LOO-CV RBF-SVM    : AUC={res["auc"]:.3f}  '
      f'Sens={res["sens"]:.3f}  Spec={res["spec"]:.3f}')
print(f'  MWU p-value       : {p_mwu:.4f}')
print('='*56)
print(f'Results → {OUT_DIR}')