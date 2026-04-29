import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import cv2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


class ThermalDataset(Dataset):
    def __init__(self, root_dirs, img_size=(64, 64)):
        self.data = []
        self.img_size = img_size

        for path, ds_name, label in root_dirs:
            if not os.path.exists(path):
                continue
            for img_name in os.listdir(path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.data.append(
                        {
                            "path": os.path.join(path, img_name),
                            "dataset": ds_name,
                            "label": label,
                        }
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = cv2.imread(item["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Gagal membaca gambar: {item['path']}")
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        return img_tensor, item["label"], item["dataset"]


def get_norm_layer(channels, norm_type="bn"):
    if norm_type == "bn":
        return nn.BatchNorm2d(channels, eps=1e-4)
    if norm_type == "gn":
        groups = 8 if channels >= 8 and channels % 8 == 0 else 1
        return nn.GroupNorm(groups, channels, eps=1e-4)
    raise ValueError("norm_type must be bn or gn")


class ResDown(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super().__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_out, norm_type=norm_type)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size, 1, kernel_size // 2)
        self.skip = nn.Conv2d(channel_in, channel_out, 1, 2)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        residual = self.skip(x)
        x = self.act_fnc(self.norm1(x))
        x = self.conv1(x)
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class ResUp(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2, norm_type="bn"):
        super().__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_out, norm_type=norm_type)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size, 1, kernel_size // 2)
        self.skip = nn.Conv2d(channel_in, channel_out, 1, 1)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        residual = self.skip(self.up_nn(x))
        x = self.up_nn(self.act_fnc(self.norm1(x)))
        x = self.conv1(x)
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super().__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)
        self.conv1 = nn.Conv2d(channel_in, channel_in, kernel_size, 1, kernel_size // 2)
        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.act_fnc = nn.ELU()
        self.skip = channel_in == channel_out
        self.bttl_nk = channel_in // 2

    def forward(self, x_in):
        x = self.act_fnc(self.norm1(x_in))
        x_cat = self.conv1(x)
        x = x_cat[:, : self.bttl_nk]
        if self.skip:
            residual = x_in
        else:
            residual = x_cat[:, self.bttl_nk :]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class Encoder(nn.Module):
    def __init__(self, channels, ch=64, blocks=(1, 2, 4, 8), latent_channels=128, num_res_blocks=1, norm_type="bn", deep_model=False):
        super().__init__()
        self.conv_in = nn.Conv2d(channels, blocks[0] * ch, 3, 1, 1)
        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [2 * blocks[-1]]
        self.layer_blocks = nn.ModuleList([])

        for w_in, w_out in zip(widths_in, widths_out):
            if deep_model:
                self.layer_blocks.append(ResBlock(w_in * ch, w_in * ch, norm_type=norm_type))
            self.layer_blocks.append(ResDown(w_in * ch, w_out * ch, norm_type=norm_type))

        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_out[-1] * ch, widths_out[-1] * ch, norm_type=norm_type))

        self.conv_mu = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.conv_log_var = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.act_fnc = nn.ELU()

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, sample=False):
        x = self.conv_in(x)
        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)
        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)
        if self.training or sample:
            z = self.sample(mu, log_var)
        else:
            z = mu
        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(self, channels, ch=64, blocks=(1, 2, 4, 8), latent_channels=128, num_res_blocks=1, norm_type="bn", deep_model=False):
        super().__init__()
        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [2 * blocks[-1]])[::-1]
        self.conv_in = nn.Conv2d(latent_channels, widths_in[0] * ch, 1, 1)
        self.layer_blocks = nn.ModuleList([])

        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_in[0] * ch, widths_in[0] * ch, norm_type=norm_type))

        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResUp(w_in * ch, w_out * ch, norm_type=norm_type))
            if deep_model:
                self.layer_blocks.append(ResBlock(w_out * ch, w_out * ch, norm_type=norm_type))

        self.conv_out = nn.Conv2d(blocks[0] * ch, channels, 5, 1, 2)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)
        return torch.tanh(self.conv_out(x))


class VAE(nn.Module):
    def __init__(self, channel_in=1, ch=64, blocks=(1, 2, 4, 8), latent_channels=128, num_res_blocks=1, norm_type="gn", deep_model=False):
        super().__init__()
        self.encoder = Encoder(
            channel_in,
            ch=ch,
            blocks=blocks,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_type=norm_type,
            deep_model=deep_model,
        )
        self.decoder = Decoder(
            channel_in,
            ch=ch,
            blocks=blocks,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_type=norm_type,
            deep_model=deep_model,
        )

    def forward(self, x):
        encoding, mu, log_var = self.encoder(x)
        recon = self.decoder(encoding)
        return recon, mu, log_var


class VGG19FeatureLoss(nn.Module):
    def __init__(self, channel_in=3, width=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, width, 3, 1, 1)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)
        self.conv3 = nn.Conv2d(width, 2 * width, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * width, 2 * width, 3, 1, 1)
        self.conv5 = nn.Conv2d(2 * width, 4 * width, 3, 1, 1)
        self.conv6 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv7 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv8 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv9 = nn.Conv2d(4 * width, 8 * width, 3, 1, 1)
        self.conv10 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv11 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv12 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv13 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv14 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv15 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv16 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def _feature_loss(x):
        return (x[: x.shape[0] // 2] - x[x.shape[0] // 2 :]).pow(2).mean()

    def forward(self, x):
        x = self.conv1(x)
        loss = self._feature_loss(x)
        x = self.conv2(self.relu(x))
        loss += self._feature_loss(x)
        x = self.mp(self.relu(x))
        x = self.conv3(x)
        loss += self._feature_loss(x)
        x = self.conv4(self.relu(x))
        loss += self._feature_loss(x)
        x = self.mp(self.relu(x))
        x = self.conv5(x)
        loss += self._feature_loss(x)
        x = self.conv6(self.relu(x))
        loss += self._feature_loss(x)
        x = self.conv7(self.relu(x))
        loss += self._feature_loss(x)
        x = self.conv8(self.relu(x))
        loss += self._feature_loss(x)
        x = self.mp(self.relu(x))
        x = self.conv9(x)
        loss += self._feature_loss(x)
        x = self.conv10(self.relu(x))
        loss += self._feature_loss(x)
        x = self.conv11(self.relu(x))
        loss += self._feature_loss(x)
        x = self.conv12(self.relu(x))
        loss += self._feature_loss(x)
        x = self.mp(self.relu(x))
        x = self.conv13(x)
        loss += self._feature_loss(x)
        x = self.conv14(self.relu(x))
        loss += self._feature_loss(x)
        x = self.conv15(self.relu(x))
        loss += self._feature_loss(x)
        x = self.conv16(self.relu(x))
        loss += self._feature_loss(x)
        return loss / 16


def kl_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def denormalize(img_tensor):
    return torch.clamp((img_tensor * 0.5) + 0.5, 0.0, 1.0)


def visualize_reconstructions(model, data_loader, device, output_dir, num_samples=5):
    model.eval()
    with torch.no_grad():
        data, labels, _ = next(iter(data_loader))
        data = data.to(device)
        recon, _, _ = model(data)

    n = min(num_samples, data.size(0))
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for i in range(n):
        axes[0, i].imshow(denormalize(data[i]).cpu().squeeze(0), cmap="inferno")
        axes[0, i].set_title(f"Original\n{labels[i]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(denormalize(recon[i]).cpu().squeeze(0), cmap="inferno")
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis("off")

    recon_path = os.path.join(output_dir, "vae_reconstructions.png")
    plt.tight_layout()
    plt.savefig(recon_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def train_vae():
    base_rg = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\DatasetRG_Watershed"
    base_dmr = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\DatasetDMR-IR_Watershed"
    output_dir = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Samakan label biner antar dataset:
    # RG: benign->Normal, malignant->Abnormal
    # DMR: Normal->Normal, Abnormal->Abnormal
    class_mapping = {
        "RG": {
            "base": os.path.join(base_rg, "anterior"),
            "folders": {"benign": "Normal", "malignant": "Abnormal"},
        },
        "DMR": {
            "base": os.path.join(base_dmr, "Anterior"),
            "folders": {"Normal": "Normal", "Abnormal": "Abnormal"},
        },
    }

    target_paths = []
    for ds_name, config in class_mapping.items():
        for folder_name, label_name in config["folders"].items():
            target_paths.append((os.path.join(config["base"], folder_name), ds_name, label_name))

    dataset = ThermalDataset(target_paths)
    if len(dataset) < 2:
        raise ValueError("Dataset terlalu kecil untuk train/validation split.")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(channel_in=1, ch=32, blocks=(1, 2, 4, 8), latent_channels=128, num_res_blocks=1, norm_type="gn", deep_model=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    feature_scale = 0.0
    feature_extractor = None
    if feature_scale > 0:
        feature_extractor = VGG19FeatureLoss(channel_in=3).to(device)

    num_epochs = 100
    patience = 5
    wait = 0
    best_val_loss = float("inf")
    best_state = None
    train_losses = []
    val_losses = []

    print(f"Memulai pelatihan pada {device}...")
    for epoch in range(num_epochs):
        model.train()
        train_total = 0.0
        for data, _, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            mse_loss = F.mse_loss(recon, data)
            kl = kl_loss(mu, logvar)
            loss = mse_loss + kl

            if feature_extractor is not None:
                feat_in = torch.cat((recon.repeat(1, 3, 1, 1), data.repeat(1, 3, 1, 1)), dim=0)
                loss = loss + feature_scale * feature_extractor(feat_in)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_total += loss.item()

        avg_train_loss = train_total / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for data, _, _ in val_loader:
                data = data.to(device)
                recon, mu, logvar = model(data)
                mse_loss = F.mse_loss(recon, data)
                kl = kl_loss(mu, logvar)
                vloss = mse_loss + kl
                val_total += vloss.item()

        avg_val_loss = val_total / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                os.path.join(output_dir, "best_cnnvae_checkpoint.pth"),
            )
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping di epoch {epoch+1} (patience={patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    loss_path = os.path.join(output_dir, "cnnvae_train_val_loss.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    model.eval()
    all_mu = []
    all_labels = []
    all_sources = []
    with torch.no_grad():
        for data, label, source in all_loader:
            _, mu, _ = model(data.to(device))
            mu = mu.mean(dim=(2, 3))
            all_mu.append(mu.cpu().numpy())
            all_labels.extend(label)
            all_sources.extend(source)

    latent_features = np.concatenate(all_mu, axis=0)
    print("Menjalankan t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    latent_2d = tsne.fit_transform(latent_features)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=latent_2d[:, 0], y=latent_2d[:, 1], hue=all_labels, palette="viridis", s=30)
    plt.title("Ruang Laten berdasarkan Diagnosis")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=latent_2d[:, 0], y=latent_2d[:, 1], hue=all_sources, palette="Set2", s=30)
    plt.title("Ruang Laten berdasarkan Asal Dataset")

    tsne_path = os.path.join(output_dir, "cnnvae_latent_tsne.png")
    plt.tight_layout()
    plt.savefig(tsne_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return model, train_loader, device, output_dir


if __name__ == "__main__":
    model, train_loader, device, output_dir = train_vae()
    visualize_reconstructions(model, train_loader, device, output_dir, num_samples=5)