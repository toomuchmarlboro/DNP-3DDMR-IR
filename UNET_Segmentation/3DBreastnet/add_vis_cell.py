import json

NOTEBOOK_PATH = r"c:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\DNP-3DDMR-IR\UNET_Segmentation\3DBreastnet\breastnet3d_v6.ipynb"

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

vis_code = """import matplotlib.pyplot as plt
import numpy as np

# Build the prior on CPU for visualization
prior_tensor = build_dual_ellipsoid_prior(128, 'cpu')
prior_np = prior_tensor.squeeze().numpy()  # Shape: (128, 128, 128)

# We visualize 3 orthogonal slices through the center of the volume
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Frontal Slice (Depth = 64) - Coronal plane
axes[0].imshow(prior_np[64, :, :], cmap='hot', vmin=0, vmax=1)
axes[0].set_title('Frontal Slice (Depth = 64)')
axes[0].set_xlabel('Width (X)')
axes[0].set_ylabel('Height (Y)')

# 2. Transverse Slice (Height = 70) - Axial plane
axes[1].imshow(prior_np[:, 70, :], cmap='hot', vmin=0, vmax=1)
axes[1].set_title('Transverse Slice (Height = 70)')
axes[1].set_xlabel('Width (X)')
axes[1].set_ylabel('Depth (Z)')

# 3. Sagittal Slice (Width = 38) - Right Breast center
axes[2].imshow(prior_np[:, :, 38], cmap='hot', vmin=0, vmax=1)
axes[2].set_title('Sagittal Slice (Width = 38)')
axes[2].set_xlabel('Height (Y)')
axes[2].set_ylabel('Depth (Z)')

plt.tight_layout()
plt.show()
"""

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + '\\n' for line in vis_code.split('\\n')][:-1]
}

# Find where to insert it (after the USE_SHAPE_PRIOR cell)
target_idx = -1
for i, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', []))
    if 'USE_SHAPE_PRIOR = True' in src:
        target_idx = i
        break

if target_idx != -1:
    nb['cells'].insert(target_idx + 1, new_cell)
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Added visualization cell.")
else:
    print("Could not find the target cell to insert after.")
