import os, random
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Safe for remote SSH
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

from models import UNet, Encoder2D, Decoder3D
from train import build_patient_groups, PatientDataset, CFG

def generate_3d_plots(n=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    out_dir = Path("3d_renders")
    out_dir.mkdir(exist_ok=True)
    
    # 1. Load U-Net
    unet = UNet().to(device)
    unet.load_state_dict(torch.load(CFG["unet_ckpt"], map_location=device, weights_only=False))
    unet.eval()
    
    # 2. Load 3DBreastNet best weights
    enc = Encoder2D().to(device)
    dec = Decoder3D().to(device)
    ckpt_path = Path(CFG["ckpt_dir"]) / "3dbreastnet_best.pth"
    
    if not ckpt_path.exists():
        print(f"Error: No trained model found at {ckpt_path}.")
        return
        
    # FIX: weights_only=False is required for PyTorch 2.6+ to load the complex checkpoint dict
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc.load_state_dict(ckpt["enc"])
    dec.load_state_dict(ckpt["dec"])
    enc.eval(); dec.eval()
    
    # 3. Get patients
    groups = build_patient_groups(CFG["tiff_base"])
    if len(groups) == 0:
        print("No patients found.")
        return
        
    selected = random.sample(groups, min(n, len(groups)))
    dataset = PatientDataset(selected, unet, device)
    
    print(f"\nGenerating 3D surface models for {len(selected)} patients...")
    
    for i in range(len(dataset)):
        item = dataset[i]
        pid, label = item["patient_id"], item["label"]
        m5 = item["masks_5ch"].unsqueeze(0).to(device)
        
        # 4. Inference
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            vol = dec(enc(m5))
        vol_np = vol[0, 0].cpu().numpy()
        
        # 5. Extract 3D Surface
        try:
            verts, faces, normals, values = marching_cubes(vol_np, level=0.5)
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                            cmap='hot', lw=0, antialiased=True, alpha=0.8)
            ax.set_title(f"Patient: {pid} | Label: {label}\nReconstructed 3D Surface")
            ax.view_init(elev=20, azim=45)
            
            plt.tight_layout()
            out_path = out_dir / f"{pid}_3d_surface.png"
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"Saved: {out_path}")
            
        except ValueError as e:
            print(f"Could not generate mesh for {pid}: {e}")

if __name__ == "__main__":
    generate_3d_plots()
