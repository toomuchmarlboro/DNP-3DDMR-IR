import numpy as np
import cv2
import pyvista as pv
from datasets import load_dataset

# =============================================================================
# PHASE 1.5 — ROBUST PATIENT DATA EXTRACTION
# =============================================================================
def extract_patient_data_robust(full_dataset, target_id):
    if not full_dataset:
        return {}

    id_key = 'patient_id'
    view_key = 'view'

    target_id_str = str(target_id).strip()
    patient_data = [rec for rec in full_dataset if str(rec.get(id_key, '')).strip() == target_id_str]
    
    if not patient_data:
        print(f" ERROR: ID {target_id} tidak ditemukan.")
        return {}

    mapping = {
        "Frontal": ["0", "frontal", "front", "anterior", "f"],
        "Right 45°": ["1", "right45", "r45", "rightlateral45", "3"], # DMR-IR indices vary
        "Left 45°": ["4", "left45", "l45", "leftlateral45"]
    }
    
    found_views = {}
    print(f"\n[DIAGNOSIS] Menganalisis ID {target_id} ({len(patient_data)} record ditemukan):")
    
    for rec in patient_data:
        v_raw = str(rec.get(view_key, '')).strip()
        v_clean = v_raw.lower().replace(" ", "").replace("°", "").replace("deg", "")
        
        thermal = np.array(rec['image'])
        
        matched_name = None
        for standard_name, aliases in mapping.items():
            if v_clean in aliases:
                matched_name = standard_name
                break
        
        if matched_name:
            # Store both thermal and mask (DMR-IR usually has 'segmentation_mask' or we generate it)
            mask = (thermal > thermal.min() + (thermal.max()-thermal.min())*0.1).astype(np.uint8)
            found_views[matched_name] = {'thermal': thermal, 'mask': mask}
            print(f"  [FOUND] '{v_raw}' -> '{matched_name}'")
            
    return found_views

# =============================================================================
# PHASE 1.6 — SPACE CARVING ENGINE (VOXEL-WISE)
# =============================================================================
def build_3d_model_fixed(found_views, depth=180):
    if "Frontal" not in found_views:
        print("Error: Frontal view is mandatory.")
        return None

    # 1. Dimensions based on the Frontal Image
    h, w = found_views["Frontal"]['mask'].shape
    
    # Initialize: Start with a solid block (True)
    voxels = np.ones((h, w, depth), dtype=bool)
    
    # Thermal data and view counter
    thermal_volume = np.zeros((h, w, depth), dtype=np.float32)
    view_count = np.zeros((h, w, depth), dtype=np.uint8)

    # DMR-IR Protocol Angles (0, 45, -45)
    # Note: Adjust names to match your extract_patient_data_robust output
    angle_map = {"Frontal": 0, "Right 45°": 45, "Left 45°": -45}
    
    # CRITICAL: Center of Rotation (The 'Spine' of the patient)
    cx, cz = w // 2, depth // 2

    print("Executing Strict Visual Hull Intersection...")

    # 2. Iterate through the 3D Grid
    for z in range(depth):
        rz = z - cz  # Depth relative to center
        for x in range(w):
            rx = x - cx  # Width relative to center
            
            # Check this voxel against EVERY available camera
            for name, data in found_views.items():
                if name not in angle_map: continue
                
                # Convert angle to radians
                theta = np.radians(angle_map[name])
                
                # THE FIX: Rotate the 3D coordinate back to the 2D Camera Pixel
                # x_pixel = center + (x_rel * cos - z_rel * sin)
                tx = int(cx + (rx * np.cos(theta) + rz * np.sin(theta)))
                
                # Is the projected pixel inside the 2D image?
                if 0 <= tx < w:
                    # Get the vertical 'Body' mask for this X position
                    mask_column = data['mask'][:, tx] > 0
                    
                    # STRICT INTERSECTION (The Carving)
                    # Voxel = Voxel AND Camera_Observation
                    voxels[:, x, z] &= mask_column
                    
                    # Accumulate thermal data only for voxels that are still 'Body'
                    survived = voxels[:, x, z]
                    if survived.any():
                        thermal_volume[survived, x, z] += data['thermal'][survived, tx]
                        view_count[survived, x, z] += 1
                else:
                    # If the voxel is outside a camera's field of view, we carve it
                    voxels[:, x, z] = False

    # 3. Finalize Surface Temperatures
    # Average the heat from overlapping views
    final_mask = (view_count > 0) & voxels
    thermal_volume[final_mask] /= view_count[final_mask]
    
    # Fill internal tissue with 37°C (Core Body Temp) for Pennes Equation
    thermal_volume[voxels & (view_count == 0)] = 37.0
    
    # Remove any isolated noise/floating voxels
    thermal_volume[~voxels] = 0

    return thermal_volume

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    TARGET_ID = 194
    print("Loading SemilleroCV/DMR-IR...")
    
    # Load dataset
    ds_dict = load_dataset("SemilleroCV/DMR-IR")
    full_ds = []
    for split in ds_dict.keys():
        full_ds.extend(ds_dict[split])

    # 1. Run Robust Extraction
    views = extract_patient_data_robust(full_ds, TARGET_ID)

    if views:
        # 2. Run Reconstruction
        vol_data = build_3d_model_fixed(views)
        
        # 3. Visualize with PyVista
        if vol_data is not None:
            grid = pv.ImageData()
            grid.dimensions = np.array(vol_data.shape)
            grid.spacing = (1, 1, 1)
            grid.point_data["Thermal"] = vol_data.flatten(order="F")

            plotter = pv.Plotter()
            # Use a slice to see the internal heat and tumor location
            plotter.add_mesh_slice(grid, normal='z', generate_triangles=True, cmap="inferno")
            # Or use a clip plane for interactive 3D cutting
            plotter.add_mesh_clip_plane(grid, scalars="Thermal", cmap="inferno")
            plotter.show()