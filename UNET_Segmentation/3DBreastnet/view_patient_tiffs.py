import os
from pathlib import Path
import tifffile
import matplotlib.pyplot as plt
import numpy as np

# Absolute path to the dataset
DATA_DIR = Path(r"../DNP-3DDMR-IR/data/organized_by_patient")

def get_view_key(filename):
    n = filename.lower()
    if "right later" in n: return "RL"
    if "right obli"  in n: return "RO"
    if "frontal" in n or "anterior" in n: return "F"
    if "left obliq"  in n: return "LO"
    if "left later"  in n: return "LL"
    return None

def view_patient_tiffs():
    print("=" * 60)
    print("    BREAST THERMOGRAPHY 5-VIEW TIFF VIEWER")
    print("=" * 60)
    
    if not DATA_DIR.exists():
        print(f"Error: Dataset directory not found at {DATA_DIR}")
        return

    while True:
        patient_id = input("\nEnter Patient ID (e.g., 'Patient_233') or 'exit' to quit: ").strip()
        if patient_id.lower() in ['exit', 'quit', 'q']:
            break
            
        if not patient_id.lower().startswith('patient_'):
            # Auto-correct if they just type "233"
            patient_id = f"Patient_{patient_id}"

        # Find the patient directory
        patient_path = DATA_DIR / patient_id
        if not patient_path.exists():
            print(f"Error: Could not find patient folder '{patient_id}' inside the dataset.")
            continue
            
        # The dataset is structured as Patient_ID / Label / *.tiff
        # Let's find the subfolder (e.g., 'benign' or 'malignant' or 'healthy')
        subfolders = [d for d in patient_path.iterdir() if d.is_dir()]
        if not subfolders:
            print(f"Error: No label folder found inside {patient_id}.")
            continue
            
        label_folder = subfolders[0]
        label = label_folder.name
        
        # Collect the 5 TIFFs
        views = {}
        for tiff_file in label_folder.glob("*.tiff"):
            vk = get_view_key(tiff_file.name)
            if vk:
                views[vk] = tiff_file
                
        ordered_keys = ["RL", "RO", "F", "LO", "LL"]
        missing = [k for k in ordered_keys if k not in views]
        
        if missing:
            print(f"Warning: Patient {patient_id} is missing views: {missing}")
            
        found_keys = [k for k in ordered_keys if k in views]
        
        if not found_keys:
            print(f"No valid TIFF views found for {patient_id}.")
            continue
            
        print(f"Loading {len(found_keys)} views for {patient_id} ({label})...")
        
        # Plotting
        fig, axes = plt.subplots(1, len(found_keys), figsize=(4 * len(found_keys), 5))
        if len(found_keys) == 1:
            axes = [axes]
            
        fig.suptitle(f"{patient_id} ({label}) - Absolute Temperature TIFFs", fontsize=16)
        
        for ax, vk in zip(axes, found_keys):
            tiff_path = views[vk]
            # Load raw absolute temperature
            img = tifffile.imread(str(tiff_path)).astype(np.float32)
            
            # Plot
            im = ax.imshow(img, cmap='inferno')
            ax.set_title(vk)
            ax.axis('off')
            
            # Add colorbar for each view
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_title("°C", fontsize=10)
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    view_patient_tiffs()
