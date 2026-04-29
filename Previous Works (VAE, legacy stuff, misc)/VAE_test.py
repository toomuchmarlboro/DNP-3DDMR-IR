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

# ==============================================================================
# 1. DATASET LOADER (CUSTOM)
# ==============================================================================

class ThermalDataset(Dataset):
    def __init__(self, root_dirs, img_size=(64, 64)):
        self.data = []
        self.img_size = img_size
        
        # root_dirs: daftar tuple (path, dataset_name, label_name)
        for path, ds_name, label in root_dirs:
            if not os.path.exists(path):
                continue
            for img_name in os.listdir(path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append({
                        'path': os.path.join(path, img_name),
                        'dataset': ds_name,
                        'label': label
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Load citra grayscale
        img = cv2.imread(item['path'], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        # Normalisasi ke [0, 1]
        img = img.astype(np.float32) / 255.0
        # Ubah ke tensor format [C, H, W]
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        
        return img_tensor, item['label'], item['dataset']

# VAE: Variational Autoencoder for latent space
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        
        # Encoder: Kompresi citra ke distribusi laten
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Vektor Laten (mu dan log-variance)
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # Decoder: Rekonstruksi dari ruang laten
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), # 64x64
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """Terapkan z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        
        de_input = self.decoder_input(z)
        de_input = de_input.view(-1, 128, 8, 8)
        reconstruction = self.decoder(de_input)
        
        return reconstruction, mu, logvar

# 3. FUNGSI KERUGIAN (LOSS FUNCTION)
def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """Gabungan Reconstruction Loss dan KL Divergence"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence: Formula tertutup untuk Gaussian
    kld_loss = -0.8 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss

# 4. TRAINING DAN VISUALISASI
def train_vae():
    # Definisikan jalur dataset lokal Anda
    base_rg = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\DatasetRG_Watershed"
    base_dmr = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\DatasetDMR-IR_Watershed"
    output_dir = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\outputs"
    os.makedirs(output_dir, exist_ok=True)

    target_paths = [
        (os.path.join(base_rg, "anterior", "benign"), "RG", "Normal"),
        (os.path.join(base_rg, "anterior", "malignant"), "RG", "Abnormal"),
        (os.path.join(base_dmr, "Anterior", "Normal"), "DMR", "Normal"),
        (os.path.join(base_dmr, "Anterior", "Abnormal"), "DMR", "Abnormal"),
    ]

    dataset = ThermalDataset(target_paths)
    if len(dataset) < 2:
        raise ValueError("Dataset terlalu kecil untuk train/validation split.")

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    all_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100
    patience = 5
    wait = 0
    best_val_loss = float("inf")
    best_state = None

    train_losses, val_losses = [], []

    print(f"Memulai pelatihan pada {device}...")
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_total = 0.0
        for data, _, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = vae_loss_function(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_total += loss.item()
        avg_train_loss = train_total / len(train_dataset)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for data, _, _ in val_loader:
                data = data.to(device)
                recon, mu, logvar = model(data)
                vloss = vae_loss_function(recon, data, mu, logvar)
                val_total += vloss.item()
        avg_val_loss = val_total / len(val_dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # Early stopping (patience=5)
        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping di epoch {epoch+1} (patience={patience}).")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Plot loss train/val (show + save)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    loss_path = os.path.join(output_dir, "vae_train_val_loss.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Ekstraksi ruang laten untuk t-SNE
    model.eval()
    all_mu, all_labels, all_sources = [], [], []
    with torch.no_grad():
        for data, label, source in all_loader:
            _, mu, _ = model(data.to(device))
            all_mu.append(mu.cpu().numpy())
            all_labels.extend(label)
            all_sources.extend(source)

    latent_features = np.concatenate(all_mu, axis=0)

    print("Menjalankan t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    latent_2d = tsne.fit_transform(latent_features)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=latent_2d[:, 0], y=latent_2d[:, 1], hue=all_labels, palette='viridis')
    plt.title("Ruang Laten berdasarkan Diagnosis")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=latent_2d[:, 0], y=latent_2d[:, 1], hue=all_sources, palette='Set2')
    plt.title("Ruang Laten berdasarkan Asal Dataset")

    tsne_path = os.path.join(output_dir, "vae_latent_tsne.png")
    plt.tight_layout()
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return model, train_loader, device, output_dir


def visualize_reconstructions(model, data_loader, device, output_dir, num_samples=5):
    model.eval()
    with torch.no_grad():
        data, _, _ = next(iter(data_loader))
        data = data.to(device)
        recon, _, _ = model(data)

    n = min(num_samples, data.size(0))
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for i in range(n):
        axes[0, i].imshow(data[i].cpu().squeeze(0), cmap='inferno')
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon[i].cpu().squeeze(0), cmap='inferno')
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis("off")

    recon_path = os.path.join(output_dir, "vae_reconstructions.png")
    plt.tight_layout()
    plt.savefig(recon_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    model, train_loader, device, output_dir = train_vae()
    visualize_reconstructions(model, train_loader, device, output_dir, num_samples=5)