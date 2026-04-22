import numpy as np
import cv2
import pyvista as pv
from datasets import load_dataset
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

# =============================================================================
# MODUL 1: SEGMENTASI (WATERSHED REFINEMENT)
# =============================================================================
def apply_watershed_mask(img_array):
    """Membersihkan background dan mengisolasi jaringan payudara."""
    img_8bit = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    cv2.watershed(cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR), markers)
    return (markers > 1).astype(np.uint8)



# =============================================================================
# MODUL 2: SPACE CARVING & THERMAL FUSION
# =============================================================================
def reconstruct_thermal_volume(patient_id, grid_size=128):
    print(f"Memuat data DMR-IR untuk Pasien {patient_id}...")
    ds = load_dataset("SemilleroCV/DMR-IR", split="train")
    
    # Mapping sudut DMR-IR (0:Frontal, 1:R45, 2:R90, 3:L45, 4:L90)
    view_angles = {"0": 0, "1": 45, "2": 90, "3": -45, "4": -90}
    views = {}
    
    # Koleksi 5 view unik
    patient_rows = [r for r in ds if str(r['patient_id']) == str(patient_id)]
    for r in patient_rows:
        v_code = str(r['view']).strip()
        if v_code in view_angles and v_code not in views:
            img = np.array(r['image'])
            views[v_code] = {
                'thermal': img,
                'mask': apply_watershed_mask(img),
                'angle': view_angles[v_code]
            }

    # Inisialisasi Grid Voxel
    voxels = np.ones((grid_size, grid_size, grid_size), dtype=bool)
    thermal_volume = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    hits = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

    print(f"Memahat volume dari {len(views)} sudut pandang...")
    for code, data in views.items():
        angle = data['angle']
        m_res = cv2.resize(data['mask'], (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
        t_res = cv2.resize(data['thermal'], (grid_size, grid_size))
        
        # Extrusion & Projection
        vol_mask = np.repeat((m_res > 0)[:, :, np.newaxis], grid_size, axis=2)
        vol_thermal = np.repeat(t_res[:, :, np.newaxis], grid_size, axis=2)
        
        # Rotasi terhadap sumbu vertikal (Y-axis pada grid)
        if angle != 0:
            rotated_mask = rotate(vol_mask, angle, axes=(0, 2), reshape=False, order=0) > 0.5
            rotated_thermal = rotate(vol_thermal, angle, axes=(0, 2), reshape=False, order=0)
        else:
            rotated_mask, rotated_thermal = vol_mask, vol_thermal
            
        # Perpotongan Visual Hull
        voxels &= rotated_mask
        thermal_volume += (rotated_thermal * rotated_mask)
        hits += rotated_mask.astype(np.uint8)

    # Fusion & Clipping
    thermal_volume[hits > 0] /= hits[hits > 0]
    thermal_volume[~voxels] = 0
    thermal_volume[:, :, :grid_size//4] = 0  # Chest Wall Clipping

    return thermal_volume

# =============================================================================
# MODUL 3: VISUALISASI PYVISTA
# =============================================================================
def visualize_model(volume):
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape)
    grid.spacing = (1, 1, 1)
    grid.point_data["Temperature"] = volume.flatten(order="F")

    dataset = grid.threshold(5.0) 
    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.add_volume(dataset, cmap="inferno", opacity="linear", shade=True)
    plotter.add_mesh_clip_plane(dataset, cmap="inferno") # Untuk membedah isi dalam
    plotter.show()

if __name__ == "__main__":
    vol = reconstruct_thermal_volume("194")
    visualize_model(vol)