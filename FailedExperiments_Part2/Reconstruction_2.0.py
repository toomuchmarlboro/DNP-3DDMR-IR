import numpy as np
import cv2
import matplotlib.pyplot as plt
from datasets import load_dataset
from scipy.ndimage import rotate, binary_dilation

# =============================================================================
# PHASE 1.6 — THERMAL SPACE CARVING WITH WATERSHED REFINEMENT
# =============================================================================

def apply_watershed_mask(img_array):
    """Membersihkan background menggunakan algoritma Watershed."""
    # Normalisasi ke 8-bit untuk OpenCV
    img_8bit = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 1. Thresholding (Otsu)
    _, thresh = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 3. Identifikasi area pasti (Foreground & Background)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
    
    # 4. Marker Labelling
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # 5. Watershed
    img_color = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    
    # Hasil mask: Area selain background (-1 dan 1 adalah background/border)
    refined_mask = np.where(markers > 1, 1, 0).astype(np.uint8)
    return refined_mask

def thermal_space_carving(views_dict, grid_size=64):
    """Membangun 3D Hull lengkap dengan Thermal Map (Texture)."""
    voxels = np.ones((grid_size, grid_size, grid_size), dtype=bool)
    # Array untuk menyimpan warna (R, G, B, A)
    colors = np.zeros((grid_size, grid_size, grid_size, 4))
    
    view_angles = {"Frontal": 0, "Right 45°": 45, "Right 90°": 90, "Left 45°": -45, "Left 90°": -90}
    
    for name, data in views_dict.items():
        angle = view_angles[name]
        mask = data['mask']
        thermal = data['thermal'] # Citra IR asli
        
        # Resize dan Centering (Gunakan logika centering dari diskusi sebelumnya)
        mask_res = cv2.resize(mask, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
        thermal_res = cv2.resize(thermal, (grid_size, grid_size))
        
        # Extrusion & Projection
        mask_binary = mask_res > 0
        vol_mask = np.repeat(mask_binary[:, :, np.newaxis], grid_size, axis=2)
        
        # Map warna thermal (Normalize ke 0-1 untuk Colormap)
        # Gunakan 'inferno' colormap untuk estetika medis
        thermal_norm = cv2.normalize(thermal_res, None, 0, 1, cv2.NORM_MINMAX)
        color_map = plt.get_cmap('inferno')(thermal_norm)
        vol_colors = np.repeat(color_map[:, :, np.newaxis, :], grid_size, axis=2)

        if angle != 0:
            rotated_mask = rotate(vol_mask, angle, axes=(0, 2), reshape=False, order=0)
            rotated_colors = rotate(vol_colors, angle, axes=(0, 2), reshape=False, order=0)
        else:
            rotated_mask = vol_mask
            rotated_colors = vol_colors
            
        # Carving
        voxels = voxels & (rotated_mask > 0.5)
        # Update warna pada voxel yang tersisa
        colors[voxels] = rotated_colors[voxels]

    return voxels, colors

# =============================================================================
# MAIN EXECUTION - FIXED FOR SUBPLOT ERROR
# =============================================================================
if __name__ == "__main__":
    print("Memuat dataset DMR-IR...")
    ds_dict = load_dataset("SemilleroCV/DMR-IR")
    full_ds = []
    for split in ds_dict.keys(): 
        full_ds.extend(ds_dict[split])

    TARGET_ID = 194
    patient_data = [r for r in full_ds if str(r['patient_id']) == str(TARGET_ID)]
    mapping = {"0": "Frontal", "1": "Right 45°", "2": "Right 90°", "3": "Left 45°", "4": "Left 90°"}
    
    views_processed = {}
    plt.figure(figsize=(15, 6))
    
    idx_plot = 1
    # Urutkan agar protokol 'Static' diproses lebih dulu jika ada
    patient_data.sort(key=lambda x: x.get('protocol', ''), reverse=True)

    for rec in patient_data:
        v_raw = str(rec['view']).strip()
        if v_raw in mapping:
            name = mapping[v_raw]
            
            # KUNCI PERBAIKAN: Jika view ini sudah diproses, lewati
            if name in views_processed:
                continue
            
            # Batasi agar tidak melebihi 5 view unik
            if idx_plot > 5:
                break
                
            thermal_img = np.array(rec['image'])
            
            # 1. Jalankan Watershed
            refined_mask = apply_watershed_mask(thermal_img)
            views_processed[name] = {'mask': refined_mask, 'thermal': thermal_img}
            
            # Output Visualisasi (Maksimal 10 subplot: 1-5 ori, 6-10 masked)
            plt.subplot(2, 5, idx_plot)
            plt.imshow(thermal_img, cmap='gray')
            plt.title(f"Original {name}")
            plt.axis('off')
            
            plt.subplot(2, 5, idx_plot + 5)
            masked_viz = cv2.bitwise_and(thermal_img, thermal_img, mask=refined_mask)
            plt.imshow(masked_viz, cmap='inferno')
            plt.title(f"Masked {name}")
            plt.axis('off')
            
            idx_plot += 1
            
    plt.tight_layout()
    plt.show()
    plt.savefig("C:\\Users\\LENOVO THINKPAD T14\\Documents\\PROPOSAL TA\\files\\Rodriguez-Guerrero Dataset\\Breast Thermography\\3D Reconstruction\\outputs")

    # 2. Jalankan 3D Thermal Reconstruction
    #if len(views_processed) >= 3:
    #    print("\nMenampilkan 3D Thermal Digital Twin...")
    #    hull_3d, thermal_colors = thermal_space_carving(views_processed, grid_size=64)
    #    
    #    fig = plt.figure(figsize=(10, 8))
    #    ax = fig.add_subplot(111, projection='3d')
    #    ax.voxels(hull_3d, facecolors=thermal_colors, alpha=0.8)
    #    ax.set_title(f"3D Thermal Reconstruction - Patient {TARGET_ID}")
    #    plt.show()