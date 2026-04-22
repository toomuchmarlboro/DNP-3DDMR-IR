import numpy as np
import cv2
import matplotlib.pyplot as plt
from datasets import load_dataset
from scipy.ndimage import rotate
import os

# =============================================================================
# PHASE 1.5 — MULTI-VIEW SPACE CARVING (VISUAL HULL) - NUMERIC ROBUST VERSION
# =============================================================================

def extract_patient_data_robust(full_dataset, target_id):
    if not full_dataset:
        return {}

    # 1. Deteksi Kolom
    first_rec = full_dataset[0]
    id_key = next((k for k in ['patient_id', 'id', 'PatientID'] if k in first_rec.keys()), 'patient_id')
    view_key = next((k for k in ['view', 'view_name', 'View'] if k in first_rec.keys()), 'view')

    # 2. Filter Pasien
    target_id_str = str(target_id).strip()
    patient_data = [rec for rec in full_dataset if str(rec.get(id_key, '')).strip() == target_id_str]
    
    if not patient_data:
        print(f" ERROR: ID {target_id} tidak ditemukan.")
        return {}

    # 3. Mapping dengan Normalisasi + Kode Numerik
    # Berdasarkan log debug ID 194: 0=Front, 1=R45, 2=R90, 3=L45, 4=L90
    mapping = {
        "Frontal": ["0", "frontal", "front", "anterior", "f"],
        "Right 45°": ["1", "right45", "r45", "rightlateral45"],
        "Right 90°": ["2", "right90", "r90", "rightlateral90"],
        "Left 45°": ["3", "left45", "l45", "leftlateral45"],
        "Left 90°": ["4", "left90", "l90", "leftlateral90"]
    }
    
    found_views = {}
    print(f"\n[DIAGNOSIS] Menganalisis ID {target_id} ({len(patient_data)} record ditemukan):")
    
    for rec in patient_data:
        v_raw = str(rec.get(view_key, '')).strip()
        # Normalisasi: kecilkan semua, hapus spasi, hapus simbol derajat
        v_clean = v_raw.lower().replace(" ", "").replace("°", "").replace("deg", "")
        
        protocol = str(rec.get('protocol', 'Static')).strip()
        
        matched_name = None
        for standard_name, aliases in mapping.items():
            if v_clean in aliases:
                matched_name = standard_name
                break
        
        if matched_name:
            # Prioritaskan protokol 'Static' jika ada duplikasi
            if matched_name not in found_views or protocol == "Static":
                mask = np.array(rec['segmentation_mask'])
                found_views[matched_name] = (mask > 0).astype(np.uint8)
                print(f"  [FOUND] '{v_raw}' -> '{matched_name}'")
        else:
            print(f"  [DEBUG] '{v_raw}' (clean: '{v_clean}') tidak cocok dengan target.")
            
    return found_views

def space_carving_3d(views_dict, grid_size=128):
    # 1. Mulai dengan grid penuh (True)
    voxels = np.ones((grid_size, grid_size, grid_size), dtype=bool)
    
    view_configs = {
        "Frontal": 0, "Right 45°": 45, "Right 90°": 90, "Left 45°": -45, "Left 90°": -90
    }

    print(f"\n[DEBUG] Memulai pahatan pada grid {grid_size}^3...")

    for name, mask in views_dict.items():
        angle = view_configs[name]
        
        # --- PERBAIKAN DIMENSI ---
        # Memaksa mask menjadi 2D (menghilangkan channel dimensi jika ada)
        if mask.ndim > 2:
            mask = np.squeeze(mask)
            if mask.ndim > 2: # Jika masih > 2 (misal RGB), ambil satu channel
                mask = mask[:, :, 0]
        
        # --- LOGIKA PENENGAHAN (CENTERING) ---
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            print(f"  [!] WARNING: Mask untuk {name} kosong! Melewati.")
            continue
            
        # Sekarang pasti hanya ada 2 values (y, x)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Crop bagian payudaranya saja
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
        
        # Letakkan di tengah canvas grid_size x grid_size
        canvas = np.zeros((grid_size, grid_size), dtype=np.uint8)
        h, w = cropped_mask.shape
        
        # Resize jika hasil crop lebih besar dari target grid
        if h > grid_size or w > grid_size:
            scale = min(grid_size/h, grid_size/w)
            h_new, w_new = int(h*scale), int(w*scale)
            cropped_mask = cv2.resize(cropped_mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
            h, w = h_new, w_new
            
        start_y = (grid_size - h) // 2
        start_x = (grid_size - w) // 2
        canvas[start_y:start_y+h, start_x:start_x+w] = cropped_mask
        
        # 2. Extrusion (Ubah 2D jadi 3D balok)
        mask_binary = canvas > 0
        vol_mask = np.repeat(mask_binary[:, :, np.newaxis], grid_size, axis=2)
        
        # 3. Rotate (Putar sesuai sudut kamera)
        if angle != 0:
            rotated_mask = rotate(vol_mask, angle, axes=(0, 2), reshape=False, order=0)
        else:
            rotated_mask = vol_mask
            
        # 4. Intersection (Proses Memahat)
        rotated_mask_bool = (rotated_mask > 0.5)
        
        # Cek apakah hasil AND membuat voxel jadi nol
        new_voxels = voxels & rotated_mask_bool
        if np.sum(new_voxels) == 0:
            print(f"  [!] Sudut {angle}° ({name}) tidak overlap! Melewati agar 3D tidak kosong.")
        else:
            voxels = new_voxels
            print(f"  [✓] {name}: Voxel tersisa = {np.sum(voxels)}")

    return voxels

if __name__ == "__main__":
    print("Memuat dataset DMR-IR...")
    try:
        # Muat semua split agar ID 194 ketemu
        ds_dict = load_dataset("SemilleroCV/DMR-IR")
        full_ds = []
        for split in ds_dict.keys():
            full_ds.extend(ds_dict[split])
        print(f"Total data: {len(full_ds)} records.")
    except Exception as e:
        print(f"Gagal: {e}"); full_ds = []

    TARGET_ID = 194 
    views = extract_patient_data_robust(full_ds, TARGET_ID)

    if len(views) < 3:
        print(f"\n[FAILED] View tidak cukup untuk ID {TARGET_ID}.")
    else:
        # Gunakan 64 jika RAM laptop berat, 128 untuk detail lebih tinggi
        GRID_RES = 32 
        hull_3d = space_carving_3d(views, grid_size=GRID_RES)
        
        print("\n[DISPLAY] Merender hasil 3D...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(hull_3d, facecolors='salmon', edgecolor='k', alpha=0.1)
        ax.set_title(f"3D Visual Hull - Patient {TARGET_ID}")
        
        # Simpan untuk Phase 2
        np.save(f"hull_{TARGET_ID}.npy", hull_3d)
        print(f"File tersimpan: hull_{TARGET_ID}.npy")
        plt.show()