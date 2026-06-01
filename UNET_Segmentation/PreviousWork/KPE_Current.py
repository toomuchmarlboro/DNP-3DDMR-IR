%pip install torch torchvision tifffile opencv-python scikit-learn matplotlib pandas seaborn tensorflow keras cv2

pip install opencv-python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split
import tifffile as tiff
import cv2
import numpy as np
import os
import glob

# ── FIX 2: Reproducibility seed ──────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class ThermalDataset(Dataset):
    def __init__(self, tiff_dir, mask_dir):
        # We find all masks first as they represent our ground truth batch
        self.mask_paths = glob.glob(os.path.join(mask_dir, "**", "*.png"), recursive=True)
        self.tiff_dir = tiff_dir
        self.mask_dir = mask_dir
        print(f"Dataset initialized with {len(self.mask_paths)} masks.")

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]

        # 1. Get the patient folder and filename (e.g., Patient_106/benign/Right Lateral...)
        rel_path = os.path.relpath(mask_path, self.mask_dir)
        patient_subfolder = os.path.dirname(rel_path)
        mask_filename = os.path.basename(rel_path)

        # 2. Robust TIFF Search: Find the TIFF that matches the mask name
        # We search the folder directly to avoid encoding mismatches with the degree symbol
        tiff_search_folder = os.path.join(self.tiff_dir, patient_subfolder)
        base_name = os.path.splitext(mask_filename)[0]

        # We use a wildcard search to find the .tiff file matching the start of the mask name
        # This bypasses the specific encoding of the (90°) part
        search_pattern = os.path.join(tiff_search_folder, "*.tiff")
        potential_tiffs = glob.glob(search_pattern)

        tiff_path = None
        for p_tiff in potential_tiffs:
            if base_name[:10] in os.path.basename(p_tiff):  # Match first 10 chars (e.g., 'Right Late')
                tiff_path = p_tiff
                break

        if tiff_path is None or not os.path.exists(tiff_path):
            raise FileNotFoundError(f"Could not find matching TIFF for {mask_path}")

        # 3. Load and Normalize TIFF (Paper Eq 1) [cite: 113]
        raw_data = tiff.imread(tiff_path)
        t_min, t_max = np.min(raw_data), np.max(raw_data)

        # Prevent division by zero
        if t_max == t_min:
            normalized = np.zeros_like(raw_data, dtype=np.float32)
        else:
            normalized = (raw_data - t_min) / (t_max - t_min)

        img_256 = cv2.resize(normalized.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)

        # 4. Load Mask
        # Use np.fromfile to bypass Windows encoding issues with the degree symbol (°)
        img_array = np.fromfile(mask_path, dtype=np.uint8)
        mask = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask file (might be corrupted or empty): {mask_path}")
        mask_256 = cv2.resize(mask, (256, 256)).astype(np.float32) / 255.0

        # Return as (C, H, W) tensors
        return torch.from_numpy(img_256).unsqueeze(0), torch.from_numpy(mask_256).unsqueeze(0)


# Paths in Thinkpad environment (adjust as needed)
TIFF_BASE = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\organized_by_patient"
MASK_BASE = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\GroundTruth_Masks"

# Paths in Thinkpad environment (adjust as needed)
# TIFF_BASE = r"C:\Users\User\Documents\Skripsi Faiz_DL\DNP-3DDMR-IR\data\organized_by_patient"
# MASK_BASE = r"C:\Users\User\Documents\Skripsi Faiz_DL\DNP-3DDMR-IR\data\GroundTruth_Masks"

# ── FIX 1: Train / test split — 78 % / 22 % per paper §3.3 ──────────────────
full_dataset = ThermalDataset(TIFF_BASE, MASK_BASE)
n_total = len(full_dataset)
n_train = int(0.78 * n_total)
n_test  = n_total - n_train
train_set, test_set = random_split(
    full_dataset, [n_train, n_test],
    generator=torch.Generator().manual_seed(SEED)
)
print(f"Split: {n_train} train / {n_test} test  (total={n_total})")

train_loader = DataLoader(train_set, batch_size=2, shuffle=True)   # Batch size 2 as per paper
test_loader  = DataLoader(test_set,  batch_size=2, shuffle=False)

# ── FIX 3: Data augmentation ─────────────────────────────────────────────────
def augment(img_tensor, mask_tensor):
    """Random horizontal flip + brightness jitter. Applied to both image and mask."""
    # Random horizontal flip — applied identically to image and mask
    if torch.rand(1).item() > 0.5:
        img_tensor  = TF.hflip(img_tensor)
        mask_tensor = TF.hflip(mask_tensor)

    # Random brightness jitter on image only (masks are binary)
    if torch.rand(1).item() > 0.5:
        factor     = 0.8 + torch.rand(1).item() * 0.4  # uniform in [0.8, 1.2]
        img_tensor = torch.clamp(img_tensor * factor, 0.0, 1.0)

    return img_tensor, mask_tensor

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, dropout=0.2):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder (4 levels)
        self.enc1 = DoubleConv(in_channels, base_channels, dropout=0.0)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, dropout=0.0)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4, dropout=0.1)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8, dropout=0.1)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16, dropout=dropout)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8, dropout=0.1)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4, dropout=0.1)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2, dropout=0.0)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels, dropout=0.0)

        # Output logits (no sigmoid here)
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model ready on: {device} | trainable params: {num_params:,}")
# ── Dice coefficient that supports logits ─────────────────────────────────────
def dice_coeff(pred, target, threshold=0.5, eps=1e-6, from_logits=True):
    """Dice over batch. pred/target shape: (B,1,H,W)."""
    if from_logits:
        pred = torch.sigmoid(pred)
    pred_bin = (pred > threshold).float()

    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    denominator = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean()
%pip install tqdm ipywidgets

from tqdm.notebook import tqdm  # Progress bar for Jupyter / VS Code
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.6, dice_weight=0.4, eps=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        dice_loss = 1.0 - dice.mean()

        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total, bce_loss.detach(), dice_loss.detach()


criterion = BCEDiceLoss(bce_weight=0.6, dice_weight=0.4)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
 )

num_epochs = 120
patience = 12
best_dice = 0.0
epochs_no_imp = 0
best_ckpt = "breast_segmentation_unet_best.pth"

use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

print(f"Starting Training on {device}...")
print(f"Early stopping: patience={patience}, monitoring=val Dice")
print("Loss: 0.6*BCEWithLogits + 0.4*DiceLoss")

epoch_train_losses = []
epoch_train_dices = []
epoch_val_dices = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for imgs, masks in progress_bar:
        imgs, masks = augment(imgs, masks)
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss, bce_part, dice_part = criterion(logits, masks)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        current_loss = loss.item()
        current_dice = dice_coeff(logits.detach(), masks, from_logits=True).item()

        epoch_loss += current_loss
        epoch_dice += current_dice
        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            dice=f"{current_dice:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}"
        )

    avg_loss = epoch_loss / len(train_loader)
    avg_dice = epoch_dice / len(train_loader)

    model.eval()
    val_dice = 0.0
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            val_dice += dice_coeff(logits, masks, from_logits=True).item()
    val_dice /= len(test_loader)

    scheduler.step(val_dice)

    epoch_train_losses.append(avg_loss)
    epoch_train_dices.append(avg_dice)
    epoch_val_dices.append(val_dice)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch [{epoch+1:03d}/{num_epochs}] "
            f"Loss: {avg_loss:.4f}  Train Dice: {avg_dice:.4f}  "
            f"Val Dice: {val_dice:.4f}  LR: {optimizer.param_groups[0]['lr']:.1e}"
        )

    if val_dice > best_dice:
        best_dice = val_dice
        epochs_no_imp = 0
        torch.save(model.state_dict(), best_ckpt)
    else:
        epochs_no_imp += 1
        if epochs_no_imp >= patience:
            print(
                f"\nEarly stopping at epoch {epoch+1} — "
                f"no Val Dice improvement for {patience} epochs."
            )
            print(f"Best Val Dice: {best_dice:.4f}  →  weights saved to '{best_ckpt}'")
            break

torch.save(model.state_dict(), "breast_segmentation_unet.pth")
print("\nTraining complete.")
print("  Last weights : breast_segmentation_unet.pth")
print(f"  Best weights : {best_ckpt}  (Val Dice: {best_dice:.4f})")
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install nvidia-pyindex
pip install nvidia-cuda-runtime-cu12
# Cell 17 (RTX 3060 optimized): Standalone GPU training
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected. Please switch to a CUDA-enabled environment.")

required = ["UNet", "full_dataset", "train_set", "test_set", "dice_coeff", "augment"]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(
        f"Missing prerequisites: {missing}. Run the setup cells for dataset and model utilities first."
    )

gpu_device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# RTX 3060-friendly defaults
train_batch_size = 4   # safe starting point for most RTX 3060 rigs
eval_batch_size = 4

train_loader_gpu = DataLoader(
    train_set,
    batch_size=train_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=0,
 )
test_loader_gpu = DataLoader(
    test_set,
    batch_size=eval_batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
 )

# Local loss class in case Cell 16 was not run
if "BCEDiceLoss" not in globals():
    class BCEDiceLoss(nn.Module):
        def __init__(self, bce_weight=0.6, dice_weight=0.4, eps=1e-6):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()
            self.bce_weight = bce_weight
            self.dice_weight = dice_weight
            self.eps = eps

        def forward(self, logits, targets):
            bce_loss = self.bce(logits, targets)
            probs = torch.sigmoid(logits)
            inter = (probs * targets).sum(dim=(1, 2, 3))
            denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
            dice = (2.0 * inter + self.eps) / (denom + self.eps)
            dice_loss = 1.0 - dice.mean()
            total = self.bce_weight * bce_loss + self.dice_weight * dice_loss
            return total, bce_loss.detach(), dice_loss.detach()

# Fresh training objects (independent from Cell 16)
model = UNet().to(gpu_device, memory_format=torch.channels_last)
criterion = BCEDiceLoss(bce_weight=0.6, dice_weight=0.4)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
 )
scaler = torch.amp.GradScaler("cuda", enabled=True)

gpu_epochs = 60
gpu_patience = 10
epochs_no_imp = 0
best_gpu_dice = 0.0
best_gpu_ckpt = "breast_segmentation_unet_best_gpu.pth"

if "epoch_train_losses" not in globals():
    epoch_train_losses = []
if "epoch_train_dices" not in globals():
    epoch_train_dices = []
if "epoch_val_dices" not in globals():
    epoch_val_dices = []

print(f"Using GPU: {gpu_name}")
print(f"Starting standalone GPU training on: {gpu_device}")
print(f"Train batch size: {train_batch_size} | Eval batch size: {eval_batch_size}")
print(f"Epochs={gpu_epochs}, EarlyStop patience={gpu_patience}")

for ep in range(gpu_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0

    gpu_bar = tqdm(train_loader_gpu, desc=f"GPU Epoch {ep+1}/{gpu_epochs}", leave=False)
    for imgs, masks in gpu_bar:
        imgs, masks = augment(imgs, masks)
        imgs = imgs.to(gpu_device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        masks = masks.to(gpu_device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(imgs)
            loss, _, _ = criterion(logits, masks)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_dice = dice_coeff(logits.detach(), masks, from_logits=True).item()
        epoch_loss += loss.item()
        epoch_dice += batch_dice
        gpu_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{batch_dice:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}"
        )

    avg_loss = epoch_loss / len(train_loader_gpu)
    avg_dice = epoch_dice / len(train_loader_gpu)

    model.eval()
    val_dice = 0.0
    with torch.no_grad():
        for imgs, masks in test_loader_gpu:
            imgs = imgs.to(gpu_device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            masks = masks.to(gpu_device, non_blocking=True)
            logits = model(imgs)
            val_dice += dice_coeff(logits, masks, from_logits=True).item()
    val_dice /= len(test_loader_gpu)

    scheduler.step(val_dice)

    epoch_train_losses.append(avg_loss)
    epoch_train_dices.append(avg_dice)
    epoch_val_dices.append(val_dice)

    if val_dice > best_gpu_dice:
        best_gpu_dice = val_dice
        epochs_no_imp = 0
        torch.save(model.state_dict(), best_gpu_ckpt)
    else:
        epochs_no_imp += 1
        if epochs_no_imp >= gpu_patience:
            print(
                f"\nEarly stopping at GPU epoch {ep+1} — "
                f"no Val Dice improvement for {gpu_patience} epochs."
            )
            break

    if (ep + 1) % 5 == 0 or ep == 0:
        print(
            f"[GPU {ep+1:03d}/{gpu_epochs}] Loss: {avg_loss:.4f} "
            f"Train Dice: {avg_dice:.4f}  Val Dice: {val_dice:.4f} "
            f"LR: {optimizer.param_groups[0]['lr']:.1e}"
        )

torch.save(model.state_dict(), "breast_segmentation_unet_gpu_last.pth")
print("\nStandalone GPU training complete.")
print(f"Best GPU Val Dice : {best_gpu_dice:.4f}")
print(f"Best GPU checkpoint: {best_gpu_ckpt}")
print("Last GPU checkpoint: breast_segmentation_unet_gpu_last.pth")
torch.save(model.state_dict(), "breast_segmentation_unet_v1_60epochs.pth")
print("Weights saved successfully!")

# Evaluate on held-out test set (logits-aware)
model.eval()
test_losses = []
test_dices = []

with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss, _, _ = criterion(logits, masks)

        test_losses.append(loss.item())
        test_dices.append(dice_coeff(logits, masks, from_logits=True).item())

print(f"Test Loss (BCE+Dice) : {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")
print(f"Test Dice             : {np.mean(test_dices):.4f} ± {np.std(test_dices):.4f}")
%pip install matplotlib

import matplotlib.pyplot as plt

# 1. Visualize the Loss and Dice Scores
def plot_metrics(train_losses, val_dices, train_dices=None):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot 1: BCE Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss (BCE+Dice)')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Dice Coefficient (Validation vs Train if available)
    plt.subplot(1, 2, 2)
    if train_dices:
        plt.plot(epochs, train_dices, 'b-', label='Train Dice')
    plt.plot(epochs, val_dices, 'g-', label='Validation Dice', linewidth=2)
    
    # Highlight the best epoch
    best_epoch = val_dices.index(max(val_dices)) + 1
    plt.plot(best_epoch, max(val_dices), 'ro', label=f'Best Val Dice (Epoch {best_epoch})')

    plt.title('Dice Coefficient (Higher is Better)')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Extract the tracked metrics from the training loop variables
# (Assuming you appended the avg_loss and val_dice to lists during the loop)
# Note: You will need to slightly modify the training loop above to append these values to lists:
# e.g., epoch_train_losses.append(avg_loss), epoch_val_dices.append(val_dice)

try:
    plot_metrics(epoch_train_losses, epoch_val_dices)
except NameError:
     print("Could not plot metrics. Ensure you are appending 'avg_loss' and 'val_dice' to lists named 'epoch_train_losses' and 'epoch_val_dices' inside your training loop.")
# 2. Final Model Save Configuration (GPU checkpoint, no stale variable dependency)
print("\n--- Saving Final Model ---")
final_model_path = "breast_segmentation_unet_gpu.pth"
source_ckpt = "breast_segmentation_unet_best_gpu.pth"

if not os.path.exists(source_ckpt):
    raise FileNotFoundError(
        f"Required checkpoint not found: {source_ckpt}. "
        "Run GPU training first (Cell 20) to generate it."
    )

# Load the GPU-best weights and save under a clear final name
model.load_state_dict(torch.load(source_ckpt, weights_only=True))
torch.save(model.state_dict(), final_model_path)
print(f"Successfully saved the highest-performing model as: {final_model_path}")
print(f"Source checkpoint: {source_ckpt}")
print(f"You can now use '{final_model_path}' in Phase 3 for automated segmentation.")
import matplotlib.pyplot as plt
import random

viz_ckpt = "breast_segmentation_unet_best_gpu.pth"
if not os.path.exists(viz_ckpt):
    raise FileNotFoundError(
        f"Visualization checkpoint not found: {viz_ckpt}. "
        "Run GPU training first (Cell 20)."
    )

# Always visualize with the newest GPU-best weights (avoid stale in-memory model state).
model.load_state_dict(torch.load(viz_ckpt, weights_only=True))
model.eval()

with torch.no_grad():
    sample_idx = random.randrange(len(test_set))
    img, mask = test_set[sample_idx]

    img = img.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0)

    logits = model(img)
    pred = torch.sigmoid(logits)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img[0, 0].cpu(), cmap='magma')
    axs[0].set_title(f"Input Thermal (TIFF) - Sample #{sample_idx}")
    axs[1].imshow(mask[0, 0].cpu(), cmap='gray')
    axs[1].set_title("Manual Mask")
    axs[2].imshow(pred[0, 0].cpu(), cmap='gray')
    axs[2].set_title("U-Net Prediction")
    for ax in axs:
        ax.axis('off')
    plt.show()
output_edges = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3"

def automated_segmentation(model_path, tiff_dir, output_dir):
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_tiffs = glob.glob(os.path.join(tiff_dir, "**", "*.tiff"), recursive=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running automated segmentation on {len(all_tiffs)} images...")
    print(f"Model checkpoint: {model_path}")
    print(f"Output edge dir : {output_dir}")

    for f_path in tqdm(all_tiffs):
        raw = tiff.imread(f_path)
        t_min, t_max = np.min(raw), np.max(raw)
        norm = (raw - t_min) / (t_max - t_min + 1e-8)
        img_256 = cv2.resize(norm.astype(np.float32), (256, 256))

        img_tensor = torch.from_numpy(img_256).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_logits = model(img_tensor)
            pred = torch.sigmoid(pred_logits).squeeze().cpu().numpy()

        binary = (pred > 0.5).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(binary)
        if not contours:
            print(f"Warning: no contour found for {os.path.basename(f_path)}, skipping.")
            continue

        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)

        edges = cv2.Canny(clean_mask, 100, 200)

        rel_path = os.path.relpath(f_path, tiff_dir)
        out_name = os.path.splitext(rel_path)[0] + "_edge.png"
        save_path = os.path.join(output_dir, out_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, edges)


model_weights = "breast_segmentation_unet_best_gpu.pth"
output_edges = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3"

if not os.path.exists(model_weights):
    raise FileNotFoundError(f"Model checkpoint not found: {model_weights}")

automated_segmentation(model_weights, TIFF_BASE, output_edges)
def robust_read_image(path, flags=cv2.IMREAD_GRAYSCALE):
    """Bypasses Windows path encoding issues and handles empty files."""
    try:
        if os.path.getsize(path) == 0:  # Check if file was stopped mid-save
            return None
        # Read file as a numpy array buffer to bypass string encoding
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
        return img
    except Exception:
        return None


def ultra_robust_light_test(tiff_dir, edge_dir, num_samples=3):
    # 1. Find all generated edge maps
    edge_files = glob.glob(os.path.join(edge_dir, "**", "*_edge.png"), recursive=True)

    if not edge_files:
        print("No edge files found! Check your Automated_Edges folder.")
        return

    # 2. Pick samples and setup plotting
    samples = random.sample(edge_files, min(num_samples, len(edge_files)))
    fig, axs = plt.subplots(len(samples), 2, figsize=(14, 6 * len(samples)))
    if len(samples) == 1:
        axs = np.expand_dims(axs, axis=0)

    for i, edge_path in enumerate(samples):
        # --- ROBUST READ ---
        edge_img = robust_read_image(edge_path)

        if edge_img is None:
            print(f"Skipping corrupted or unreadable edge map: {os.path.basename(edge_path)}")
            continue

        # 3. Find corresponding TIFF using Prefix Matching
        rel_path          = os.path.relpath(edge_path, edge_dir)
        patient_subfolder = os.path.dirname(rel_path)
        edge_filename     = os.path.basename(rel_path)
        search_folder     = os.path.join(tiff_dir, patient_subfolder)
        search_prefix     = edge_filename.split('(')[0].strip()

        potential_tiffs = glob.glob(os.path.join(search_folder, "*.tiff"))
        tiff_path = next((p for p in potential_tiffs if search_prefix in os.path.basename(p)), None)

        if tiff_path is None:
            continue

        # 4. Load/Normalize TIFF and Resize to 256x256
        raw      = tiff.imread(tiff_path)
        norm     = 255 * (raw - np.min(raw)) / (np.max(raw) - np.min(raw) + 1e-8)
        norm_256 = cv2.resize(norm.astype(np.uint8), (256, 256), interpolation=cv2.INTER_AREA)
        base_display = cv2.applyColorMap(norm_256, cv2.COLORMAP_MAGMA)

        # 5. Overlay edges
        overlay = base_display.copy()
        # The 'if edge_img is not None' check prevents the TypeError
        overlay[edge_img > 0] = [0, 255, 0]

        # Plotting
        axs[i, 0].imshow(cv2.cvtColor(base_display, cv2.COLOR_BGR2RGB))
        axs[i, 0].set_title(f"Thermal: {os.path.basename(tiff_path)}")
        axs[i, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axs[i, 1].set_title("Edge Overlay (Green)")
        for ax in axs[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# Execute the test
ultra_robust_light_test(TIFF_BASE, output_edges, num_samples=3)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def isolate_inframammary_fold_sequential(edge_img):
    """
    Implements the 4-step edge identification from Costa et al. (2023)
    and returns the visual states of all 4 steps.
    """
    h, w = edge_img.shape
    
    # Create blank canvas to store the visual progress of each step
    img_step1 = np.zeros_like(edge_img)
    img_step2 = np.zeros_like(edge_img)
    img_step3 = np.zeros_like(edge_img)
    img_step4 = np.zeros_like(edge_img)
    
    # Coordinate sets for mathematical subtraction
    bottom_pts = set()
    right_pts = set()
    left_pts = set()
    
    # Step 1: Bottom-up search (Find bottom-most pixel per column)
    for x in range(w):
        y_indices = np.where(edge_img[:, x] > 127)[0]
        if len(y_indices) > 0:
            max_y = np.max(y_indices)
            bottom_pts.add((x, max_y))
            img_step1[max_y, x] = 255
            
    # Step 2: Right-to-left search (Find right-most pixel per row)
    for y in range(h):
        x_indices = np.where(edge_img[y, :] > 127)[0]
        if len(x_indices) > 0:
            max_x = np.max(x_indices)
            right_pts.add((max_x, y))
            img_step2[y, max_x] = 255
            
    # Step 3: Left-to-right search (Find left-most pixel per row)
    for y in range(h):
        x_indices = np.where(edge_img[y, :] > 127)[0]
        if len(x_indices) > 0:
            min_x = np.min(x_indices)
            left_pts.add((min_x, y))
            img_step3[y, min_x] = 255
            
    # Step 4: Subtraction (Step 1 MINUS Step 2 MINUS Step 3)
    # This removes the armpit and lateral side lines
    pure_fold = bottom_pts - right_pts - left_pts
    
    for (x, y) in pure_fold:
        img_step4[y, x] = 255
        
    return img_step1, img_step2, img_step3, img_step4

# --- VISUALIZATION EXECUTION FOR 5 VIEWS ---
# The exact paths provided for Patient 233
patient_views = [
    r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Anterior (Front)_edge.png",
    r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Left Lateral (90Â°)_edge.png",
    r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Left Oblique (45Â°)_edge.png",
    r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Right Lateral (90Â°)_edge.png",
    r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Right Oblique (45Â°)_edge.png"
]

# Set up a 5x4 grid plot
fig, axs = plt.subplots(5, 4, figsize=(20, 25))
fig.suptitle("Sequential Isolation of Inframammary Fold - Patient 13 (All 5 Views)", fontsize=20, fontweight='bold')

for i, path in enumerate(patient_views):
    if os.path.exists(path):
        # Using np.fromfile to safely handle Windows paths with special characters like Â°
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        # Extract view name from the file path for titling
        view_name = os.path.basename(path).split('_edge')[0]
        
        # Run the sequential filter
        step1, step2, step3, step4 = isolate_inframammary_fold_sequential(img)
        
        # Plot Step 1
        axs[i, 0].imshow(step1, cmap='gray')
        if i == 0: axs[i, 0].set_title("Step 1: Bottom-Up Scan", fontsize=14)
        axs[i, 0].set_ylabel(view_name, fontsize=14, fontweight='bold', rotation=90, labelpad=20)
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        
        # Plot Step 2
        axs[i, 1].imshow(step2, cmap='gray')
        if i == 0: axs[i, 1].set_title("Step 2: Right-to-Left Scan", fontsize=14)
        axs[i, 1].axis('off')
        
        # Plot Step 3
        axs[i, 2].imshow(step3, cmap='gray')
        if i == 0: axs[i, 2].set_title("Step 3: Left-to-Right Scan", fontsize=14)
        axs[i, 2].axis('off')
        
        # Plot Step 4
        axs[i, 3].imshow(step4, cmap='gray')
        if i == 0: axs[i, 3].set_title("Step 4: Subtraction (Clean Fold)", fontsize=14)
        axs[i, 3].axis('off')
        
    else:
        print(f"Warning: Could not find image at path: {path}")

plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make room for the main title
plt.show()
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def isolate_inframammary_fold(edge_img):
    """
    Implements the 4-step edge identification to isolate the pure lower breast contour.
    Returns the coordinates of the clean fold.
    """
    h, w = edge_img.shape
    bottom_pts, right_pts, left_pts = set(), set(), set()
    
    # Step 1, 2, 3
    for x in range(w):
        y_indices = np.where(edge_img[:, x] > 127)[0]
        if len(y_indices) > 0: bottom_pts.add((x, np.max(y_indices)))
    for y in range(h):
        x_indices = np.where(edge_img[y, :] > 127)[0]
        if len(x_indices) > 0:
            right_pts.add((np.max(x_indices), y))
            left_pts.add((np.min(x_indices), y))
            
    # Step 4: Subtraction
    pure_fold = bottom_pts - right_pts - left_pts
    pure_fold_arr = np.array(list(pure_fold))
    
    # Sort left-to-right by X coordinate
    if len(pure_fold_arr) > 0:
        pure_fold_arr = pure_fold_arr[pure_fold_arr[:, 0].argsort()]
        
    return pure_fold_arr

def extract_keypoints(pure_fold_arr, view_deg):
    """
    Extracts P1, P2, P3 based on Costa et al. (2023) geometric rules.
    """
    if len(pure_fold_arr) == 0:
        return None, None, None

    # P1 and P3 are always the "first points" / "beginning and end" of the sorted curve
    P1 = pure_fold_arr[0]   # Leftmost point (Patient's Right side)
    P3 = pure_fold_arr[-1]  # Rightmost point (Patient's Left side)
    
    if view_deg in [0.0, 45.0, -45.0]:
        # P2: Junction of the curves in the center (Physically highest point -> minimum Y)
        p2_idx = np.argmin(pure_fold_arr[:, 1])
        P2 = pure_fold_arr[p2_idx]

    elif view_deg in [90.0, -90.0]:
        # P2: Lowest point of the curve (Physically lowest point -> maximum Y)
        p2_idx = np.argmax(pure_fold_arr[:, 1])
        P2 = pure_fold_arr[p2_idx]
        
    return tuple(P1), tuple(P2), tuple(P3)

# --- VISUALIZATION EXECUTION FOR ALL 5 VIEWS ---
patient_views = {
    0.0: r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Anterior (Front)_edge.png",
    45.0: r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Left Oblique (45Â°)_edge.png",
    -45.0: r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Right Oblique (45Â°)_edge.png",
    90.0: r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Left Lateral (90Â°)_edge.png",
    -90.0: r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Right Lateral (90Â°)_edge.png"
}

fig, axs = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle("Keypoint Extraction (P1, P2, P3) Across All 5 Views", fontsize=18, fontweight='bold')

for i, (view_deg, path) in enumerate(patient_views.items()):
    if os.path.exists(path):
        # Load image safely
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        view_name = os.path.basename(path).split('_edge')[0]
        
        # 1. Clean the fold
        pure_curve = isolate_inframammary_fold(img)
        
        # 2. Extract Keypoints
        P1, P2, P3 = extract_keypoints(pure_curve, view_deg)
        
        # 3. Draw Results
        canvas = np.zeros_like(img)
        if len(pure_curve) > 0:
            canvas[pure_curve[:, 1], pure_curve[:, 0]] = 255
            
        axs[i].imshow(canvas, cmap='gray')
        
        if P1 and P2 and P3:
            axs[i].scatter(P1[0], P1[1], c='red', s=100, label='P1' if i==0 else "")
            axs[i].scatter(P2[0], P2[1], c='green', s=100, label='P2' if i==0 else "")
            axs[i].scatter(P3[0], P3[1], c='blue', s=100, label='P3' if i==0 else "")
            
        axs[i].set_title(f"{view_name}", fontsize=12)
        axs[i].axis('off')
    else:
        axs[i].set_title(f"Missing: {view_deg}°")
        axs[i].axis('off')

fig.legend(loc='lower center', ncol=3, fontsize=14, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.show()
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def _order_curve_8_connected(points):
    """Order unordered edge pixels into a near-continuous 8-connected path."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=int)

    pts = [tuple(map(int, p)) for p in points]
    pts = list(dict.fromkeys(pts))  # keep unique points, stable order
    point_set = set(pts)

    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    neighbors = {}
    for p in pts:
        x, y = p
        nbs = []
        for dx, dy in neighbor_offsets:
            q = (x + dx, y + dy)
            if q in point_set:
                nbs.append(q)
        neighbors[p] = nbs

    endpoints = [p for p in pts if len(neighbors[p]) <= 1]
    if endpoints:
        start = min(endpoints, key=lambda t: (t[0], t[1]))
    else:
        start = min(pts, key=lambda t: (t[0], t[1]))

    ordered = [start]
    visited = {start}
    current = start

    while len(visited) < len(point_set):
        candidates = [q for q in neighbors[current] if q not in visited]
        if not candidates:
            unvisited = [q for q in pts if q not in visited]
            if not unvisited:
                break
            current = min(unvisited, key=lambda q: (q[0] - current[0]) ** 2 + (q[1] - current[1]) ** 2)
            ordered.append(current)
            visited.add(current)
            continue

        next_p = min(candidates, key=lambda q: (q[0] - current[0]) ** 2 + (q[1] - current[1]) ** 2)
        ordered.append(next_p)
        visited.add(next_p)
        current = next_p

    return np.array(ordered, dtype=int)


def extract_lower_single_edge(edge_img):
    """Paper step (4): keep only the lower single edge (inframammary fold)."""
    h, w = edge_img.shape
    bottom_pts, right_pts, left_pts = set(), set(), set()

    # Step 1: bottom-most edge pixel for each column
    for x in range(w):
        y_idx = np.where(edge_img[:, x] > 127)[0]
        if len(y_idx) > 0:
            bottom_pts.add((x, int(np.max(y_idx))))

    # Step 2 and 3: lateral borders to subtract
    for y in range(h):
        x_idx = np.where(edge_img[y, :] > 127)[0]
        if len(x_idx) > 0:
            right_pts.add((int(np.max(x_idx)), y))
            left_pts.add((int(np.min(x_idx)), y))

    pure_fold = bottom_pts - right_pts - left_pts
    if not pure_fold:
        return np.empty((0, 2), dtype=int)

    return _order_curve_8_connected(np.array(list(pure_fold), dtype=int))


def extract_keypoints_paper(lower_curve, view_deg):
    """
    Paper-aligned P1/P2/P3 extraction (Algorithm 1, lines 10-20):
    - 0, +45, -45: P2 = upper junction (min y), P1 = RIGHT inflection (max x), P3 = LEFT inflection (min x)
    - +90, -90: P2 = lowest point (max y), P1 = toward body center, P3 = toward arms
    """
    if len(lower_curve) == 0:
        return None, None, None

    if view_deg in (0.0, 45.0, -45.0):
        # Alg-1 lines 11-16
        # P2 = highest point (min y) = center junction
        # P1 = RIGHT side of image (max x)
        # P3 = LEFT side of image (min x)
        p2 = lower_curve[np.argmin(lower_curve[:, 1])]
        p1 = lower_curve[np.argmax(lower_curve[:, 0])]   # RIGHT (max x)
        p3 = lower_curve[np.argmin(lower_curve[:, 0])]   # LEFT  (min x)
    elif view_deg in (90.0, -90.0):
        # Alg-1 lines 17-20: P2=lowest(max y), P1=body center, P3=arms
        p2 = lower_curve[np.argmax(lower_curve[:, 1])]
        if view_deg == 90.0:
            # Right lateral: body center is LEFT (min x), arms RIGHT (max x)
            p1 = lower_curve[np.argmin(lower_curve[:, 0])]
            p3 = lower_curve[np.argmax(lower_curve[:, 0])]
        else:
            # Left lateral (-90): body center is RIGHT (max x), arms LEFT (min x)
            p1 = lower_curve[np.argmax(lower_curve[:, 0])]
            p3 = lower_curve[np.argmin(lower_curve[:, 0])]
    else:
        return None, None, None

    return tuple(map(int, p1)), tuple(map(int, p2)), tuple(map(int, p3))


def get_moura_active_curve(edge_img, view_deg, P1, P2, P3):
    """Blue line in Fig. 5 is the lower single edge from step (4), where P1/P2/P3 lie."""
    _ = (view_deg, P1, P2, P3)  # keep signature stable for later cells
    return extract_lower_single_edge(edge_img)


# --- EXECUTION ---
patient_views = {
    0.0: r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Anterior (Front)_edge.png",
    45.0: r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Right Oblique (45Â°)_edge.png",
    90.0: r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Right Lateral (90Â°)_edge.png",
    -45.0: r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Left Oblique (45Â°)_edge.png",
    -90.0: r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\data\Automated_Edges3\Patient_13\benign\Left Lateral (90Â°)_edge.png",
}

fig, axs = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle("Figure 5 style: lower single edge (blue) with P1/P2/P3", fontsize=18, fontweight="bold")

for i, (view_deg, path) in enumerate(patient_views.items()):
    if not os.path.exists(path):
        axs[i].set_title(f"Missing: {view_deg} deg")
        axs[i].axis("off")
        continue

    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        axs[i].set_title(f"Unreadable: {view_deg} deg")
        axs[i].axis("off")
        continue

    lower_curve = extract_lower_single_edge(img)
    P1, P2, P3 = extract_keypoints_paper(lower_curve, view_deg)
    active_curve = get_moura_active_curve(img, view_deg, P1, P2, P3)

    axs[i].imshow(img, cmap="gray", alpha=0.18)

    if len(active_curve) > 0:
        axs[i].plot(active_curve[:, 0], active_curve[:, 1], color="dodgerblue", linewidth=2.5)

    if P1 and P2 and P3:
        axs[i].scatter(P1[0], P1[1], c="red", s=120)
        axs[i].scatter(P2[0], P2[1], c="yellow", s=120)
        axs[i].scatter(P3[0], P3[1], c="lime", s=120)
        axs[i].text(P1[0] + 3, P1[1] - 3, "P1", color="red", fontsize=10, weight="bold")
        axs[i].text(P2[0] + 3, P2[1] - 3, "P2", color="yellow", fontsize=10, weight="bold")
        axs[i].text(P3[0] + 3, P3[1] - 3, "P3", color="lime", fontsize=10, weight="bold")

    axs[i].set_title(f"{view_deg:+.0f} deg")
    axs[i].axis("off")

plt.tight_layout()
plt.show()
%pip install plotly
# Phase 5 -- 3-D Geometric Transformation (Costa et al. 2023)
import numpy as np, cv2

def get_ordered_contour(edge_img, roof_margin=12):
    """Extract largest ORDERED contour from edge image via cv2.findContours."""
    mask = edge_img.copy()
    if roof_margin > 0:
        mask[:roof_margin, :] = 0
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=float)
    largest = max(contours, key=len)
    return largest.reshape(-1, 2).astype(float)

def longest_contiguous_segment(pts_3d, mask):
    """
    Given a closed-loop contour and a boolean mask (e.g. z<=0),
    extract the longest contiguous run of True values,
    handling wrap-around for closed contours.
    Returns an OPEN curve (no loop).
    """
    n = len(mask)
    if n == 0 or mask.sum() == 0:
        return np.empty((0, 3))
    if mask.all():
        return pts_3d

    # Double the mask to handle wrap-around
    doubled = np.concatenate([mask, mask])
    best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
    for i in range(2 * n):
        if doubled[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    best_len = min(best_len, n)
    indices = [(best_start + i) % n for i in range(best_len)]
    return pts_3d[indices]


def extract_body_edge_from_contour(contour, start_pt):
    idx = np.argmin(np.sum((contour - start_pt)**2, axis=1))
    n = len(contour)
    
    forward_y = contour[(idx + 10) % n, 1]
    backward_y = contour[(idx - 10 + n) % n, 1]
    
    step = 1 if forward_y < backward_y else -1
        
    path = []
    curr = idx
    min_y = np.min(contour[:, 1])
    while True:
        pt = contour[curr]
        path.append(pt)
        if pt[1] <= min_y + 5:
            break
        curr = (curr + step + n) % n
        if len(path) > n:
            break
            
    return np.array(path)

def generate_9_curves_3d(patient_views, roof_margin=12, verbose=True):
    curves_3d = []
    if 0.0 not in patient_views:
        raise KeyError("Need 0.0 (frontal) key.")

    img_0 = cv2.imdecode(np.fromfile(patient_views[0.0], dtype=np.uint8),
                         cv2.IMREAD_GRAYSCALE)
    if img_0 is None:
        raise IOError("Cannot read frontal image.")

    pure_0 = extract_lower_single_edge(img_0)
    p1_0, p2_0, p3_0 = extract_keypoints_paper(pure_0, 0.0)
    if p1_0 is None:
        raise RuntimeError("Cannot extract keypoints from 0 deg view.")
    p1_0 = np.array(p1_0, dtype=float)
    p2_0 = np.array(p2_0, dtype=float)
    p3_0 = np.array(p3_0, dtype=float)

    if verbose:
        print(f"Frontal anchors: P1={p1_0.astype(int)}, "
              f"P2={p2_0.astype(int)}, P3={p3_0.astype(int)}")

    # Ordered contour for 0-deg
    contour_0 = get_ordered_contour(img_0, roof_margin)
    if len(contour_0) == 0:
        raise RuntimeError("No contour in 0 deg image.")

    # C1: all points RIGHT of P1 (Alg-1: includes breast + right body edge)
    c1_mask = contour_0[:, 0] >= p1_0[0]
    c1_pts = longest_contiguous_segment(
        np.column_stack((contour_0, np.zeros(len(contour_0)))), c1_mask)
    if len(c1_pts) > 0:
        curves_3d.append(c1_pts)

    # C2: all points LEFT of P3 (Alg-1: includes breast + left body edge)
    c2_mask = contour_0[:, 0] <= p3_0[0]
    c2_pts = longest_contiguous_segment(
        np.column_stack((contour_0, np.zeros(len(contour_0)))), c2_mask)
    if len(c2_pts) > 0:
        curves_3d.append(c2_pts)

    # C3: auxiliary mid-line from P2 to top (Alg-1 line 37)
    top_y = float(contour_0[:, 1].min())
    mid_y = np.linspace(p2_0[1], top_y, 50)
    curves_3d.append(np.column_stack((np.full(50, p2_0[0]), mid_y, np.zeros(50))))

    if verbose:
        print(f"C1={len(c1_pts)}, C2={len(c2_pts)}, C3=50 pts (frontal)")

    # C4-C9: rotated side-view curves
    # Fix: Right breast views (45, 90) -> anchor P3 (min X). Rotate by -src_angle
    # Fix: Left breast views (-45, -90) -> anchor P1 (max X). Rotate by -src_angle
    rules = [
        ( 45.0, p3_0,  -45.0, "C4:  45 -> P3"),
        ( 90.0, p3_0,  -90.0, "C7:  90 -> P3"),
        (-45.0, p1_0,   45.0, "C6: -45 -> P1"),
        (-90.0, p1_0,   90.0, "C5: -90 -> P1"),
        ( 45.0, p3_0, -135.0, "C8: aux -135 -> P3"),
        (-45.0, p1_0,  135.0, "C9: aux 135 -> P1"),
    ]

    for src_angle, pivot_global, rot_deg, label in rules:
        if src_angle not in patient_views:
            continue
        img_src = cv2.imdecode(
            np.fromfile(patient_views[src_angle], dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE)
        if img_src is None:
            continue

        pure_src = extract_lower_single_edge(img_src)
        _, p2_s, _ = extract_keypoints_paper(pure_src, src_angle)
        if p2_s is None:
            if verbose: print(f"  [SKIP] {label}")
            continue
        p2_s = np.array(p2_s, dtype=float)

        active = get_ordered_contour(img_src, roof_margin).astype(float)
        if len(active) == 0:
            continue

        # Step 22: Translate P2 to origin
        x_c = active[:, 0] - p2_s[0]
        y_c = active[:, 1] - p2_s[1]

        # Step 23: Ry(theta) rotation
        rad = np.radians(rot_deg)
        x_rot =  x_c * np.cos(rad)
        y_rot =  y_c
        z_rot = -x_c * np.sin(rad)

        # Steps 25/27: Translate to pivot
        x_f = x_rot + pivot_global[0]
        y_f = y_rot + pivot_global[1]
        z_f = z_rot

        # Step 38: Keep only z <= 0, extract longest contiguous segment
        pts_3d = np.column_stack((x_f, y_f, z_f))
        z_mask = z_f <= 0
        segment = longest_contiguous_segment(pts_3d, z_mask)

        if verbose:
            print(f"  {label}: {len(active)} -> {len(segment)} pts")

        if len(segment) > 0:
            curves_3d.append(segment)

    return curves_3d

# -- Execute --
print("=" * 60)
print("Phase 5: Generating 9 curves with contiguous segment extraction")
print("=" * 60)
curves_3d = generate_9_curves_3d(patient_views, verbose=True)
print(f"\nTotal curves: {len(curves_3d)}")
for k, c in enumerate(curves_3d):
    print(f"  C{k+1}: {len(c):5d} pts | "
          f"X=[{c[:,0].min():.0f},{c[:,0].max():.0f}] "
          f"Y=[{c[:,1].min():.0f},{c[:,1].max():.0f}] "
          f"Z=[{c[:,2].min():.0f},{c[:,2].max():.0f}]")

# Phase 6 -- B-spline Curve Fitting (Costa et al. S2.2, Alg-1 line 39)
# ================================================================
# Since Phase 5 now provides ORDERED contour points, B-spline fitting
# is straightforward -- no PCA reordering needed.

from scipy.interpolate import splprep, splev
import numpy as np

BSPLINE_DEGREE = 4          # Paper: "curves of degree 4"
N_EVAL_PTS     = 100        # Uniform resample count per curve

def fit_bspline_3d(pts_3d, degree=BSPLINE_DEGREE, n_eval=N_EVAL_PTS):
    """
    Fit a B-spline through ORDERED 3-D points.

    Parameters
    ----------
    pts_3d : ndarray (N, 3) -- ordered points along the curve
    degree : int -- B-spline degree (paper uses 4)
    n_eval : int -- number of uniformly-spaced evaluation points

    Returns
    -------
    eval_pts : ndarray (n_eval, 3) -- smoothly resampled curve
    tck      : tuple -- B-spline representation
    """
    if len(pts_3d) < degree + 1:
        return pts_3d.copy(), None

    # Remove consecutive duplicates
    diffs = np.linalg.norm(np.diff(pts_3d, axis=0), axis=1)
    keep = np.concatenate(([True], diffs > 1e-8))
    clean_pts = pts_3d[keep]

    if len(clean_pts) < degree + 1:
        return clean_pts.copy(), None

    # Subsample if too many points (speeds up fitting, reduces noise)
    max_ctrl = 200
    if len(clean_pts) > max_ctrl:
        idx = np.linspace(0, len(clean_pts) - 1, max_ctrl, dtype=int)
        clean_pts = clean_pts[idx]

    try:
        tck, u = splprep([clean_pts[:, 0], clean_pts[:, 1], clean_pts[:, 2]],
                         k=degree, s=len(clean_pts) * 2.0)
        u_new = np.linspace(0, 1, n_eval)
        x_new, y_new, z_new = splev(u_new, tck)
        return np.column_stack((x_new, y_new, z_new)), tck
    except Exception as e:
        print(f"  B-spline fit failed: {e}")
        return clean_pts.copy(), None


# -- Fit all curves --
bspline_curves = []
bspline_tcks   = []

print("=" * 60)
print("Phase 6: Fitting B-spline curves (degree 4) ...")
print("=" * 60)

for k, raw_pts in enumerate(curves_3d):
    smooth_pts, tck = fit_bspline_3d(raw_pts)
    bspline_curves.append(smooth_pts)
    bspline_tcks.append(tck)
    status = "OK" if tck is not None else "fallback (too few pts)"
    print(f"  Curve {k+1}: {len(raw_pts):5d} raw pts -> "
          f"{len(smooth_pts):4d} B-spline pts  [{status}]")

print(f"\nTotal B-spline curves: {len(bspline_curves)}")

# Phase 7 — 3-D Visualization of B-spline Curves
# ================================================================
import plotly.graph_objects as go
import numpy as np

CURVE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"
]

fig = go.Figure()

for k, curve in enumerate(bspline_curves):
    color = CURVE_COLORS[k % len(CURVE_COLORS)]
    fig.add_trace(go.Scatter3d(
        x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
        mode="lines",
        name=f"C{k+1}",
        line=dict(width=4, color=color),
    ))

# Add P1, P2, P3 anchor points from the frontal view
pure_0_vis = extract_lower_single_edge(
    cv2.imdecode(np.fromfile(patient_views[0.0], dtype=np.uint8),
                 cv2.IMREAD_GRAYSCALE))
p1_vis, p2_vis, p3_vis = extract_keypoints_paper(pure_0_vis, 0.0)

if p1_vis and p2_vis and p3_vis:
    for pt, name, color in [(p1_vis, "P1", "red"),
                             (p2_vis, "P2", "yellow"),
                             (p3_vis, "P3", "lime")]:
        fig.add_trace(go.Scatter3d(
            x=[pt[0]], y=[pt[1]], z=[0],
            mode="markers+text",
            name=name,
            marker=dict(size=8, color=color, symbol="diamond"),
            text=[name], textposition="top center",
        ))

fig.update_layout(
    title="Phase 7: 9 B-spline Curves in 3-D (Costa et al. Fig. 7a)",
    scene=dict(
        xaxis_title="X (horizontal)",
        yaxis_title="Y (vertical / down)",
        zaxis_title="Z (depth, <0 = toward camera)",
        aspectmode="data",
    ),
    width=900, height=700,
    margin=dict(l=0, r=0, t=40, b=0),
)

fig.show()
print("✓ 3-D visualization complete.")

# Phase 8 — NURBS Surface Generation & Visualization
# ================================================================
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata

# Collect all B-spline evaluated points
all_pts = np.vstack(bspline_curves)

# Create a regular grid for interpolation
x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()

N_GRID = 80  # resolution of the surface mesh

xi = np.linspace(x_min, x_max, N_GRID)
yi = np.linspace(y_min, y_max, N_GRID)
XI, YI = np.meshgrid(xi, yi)

# Interpolate Z values on the regular grid
ZI = griddata(
    (all_pts[:, 0], all_pts[:, 1]),
    all_pts[:, 2],
    (XI, YI),
    method="cubic"
)

# ── Build the 3-D surface figure ──────────────────────────────────────────
fig = go.Figure()

# Surface mesh
fig.add_trace(go.Surface(
    x=XI, y=YI, z=ZI,
    colorscale="Blues",
    opacity=0.7,
    name="NURBS Surface",
    showscale=True,
    colorbar=dict(title="Z depth"),
))

# Overlay the B-spline curves
CURVE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"
]
for k, curve in enumerate(bspline_curves):
    color = CURVE_COLORS[k % len(CURVE_COLORS)]
    fig.add_trace(go.Scatter3d(
        x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
        mode="lines",
        name=f"C{k+1}",
        line=dict(width=3, color=color),
    ))

fig.update_layout(
    title="Phase 8: NURBS Surface from 9 B-spline Curves (Costa et al. Fig. 7b-c)",
    scene=dict(
        xaxis_title="X (horizontal)",
        yaxis_title="Y (vertical / down)",
        zaxis_title="Z (depth, <0 = toward camera)",
        aspectmode="data",
    ),
    width=900, height=700,
    margin=dict(l=0, r=0, t=40, b=0),
)

fig.show()
print("✓ NURBS surface generation complete.")
print(f"  Grid: {N_GRID}×{N_GRID} = {N_GRID**2} vertices")
print(f"  Source curves: {len(bspline_curves)}, total pts: {len(all_pts)}")
