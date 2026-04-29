import os
import numpy as np
import cv2
import pyvista as pv

def integrate_multiview_volume():
    # --- 1. CONFIGURATION ---
    base_path = r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction\DatasetRG_Watershed"
    category = "benign"
    patient_id = "IIR0001" 

    # Load 3 views
    img_ant = cv2.imread(os.path.join(base_path, "anterior", category, f"{patient_id}_anterior.png"), 0)
    img_l = cv2.imread(os.path.join(base_path, "oblleft", category, f"{patient_id}_oblleft.png"), 0)
    img_r = cv2.imread(os.path.join(base_path, "oblright", category, f"{patient_id}_oblright.png"), 0)

    if img_ant is None or img_l is None or img_r is None:
        print("Missing views. Ensure all 3 images exist in the folders.")
        return

    # Dimensions
    h, w = img_ant.shape
    depth = 150 # The Z-axis depth of the 3D volume
    
    # --- 2. INITIALIZE 3D VOXEL GRID ---
    # Each voxel stores a thermal intensity value
    volume = np.zeros((h, w, depth), dtype=np.float32)
    
    # Pre-calculate rotation constants
    angle = np.radians(45)
    cos45 = np.cos(angle)
    sin45 = np.sin(angle)
    
    # Center point for the rotation
    cx, cz = w // 2, depth // 2

    print(f"Integrating multi-view data for {patient_id}...")

    # --- 3. SPACE CARVING & THERMAL FUSION ---
    for z in range(depth):
        for x in range(w):
            # Calculate coordinates for the oblique views based on 45-degree rotation
            # We transform the (x, z) coordinates to see where they land in the side views
            rx = x - cx
            rz = z - cz
            
            # Rotation Matrix applied to X and Z
            x_left = int(cx + (rx * cos45 - rz * sin45))
            x_right = int(cx + (rx * cos45 + rz * sin45))
            
            # Check if the rotated coordinates are within the image bounds
            if 0 <= x_left < w and 0 <= x_right < w:
                # INTEGRATION LOGIC:
                # A voxel is valid only if it is 'active' (non-black) in all three views
                # This is the "Visual Hull" intersection
                mask_ant = img_ant[:, x] > 5
                mask_l = img_l[:, x_left] > 5
                mask_r = img_r[:, x_right] > 5
                
                valid_y = mask_ant & mask_l & mask_r
                
                # FUSE DATA: Average the thermal values from available views
                # This integrates the multi-view perspectives into a single voxel value
                fused_thermal = (img_ant[:, x].astype(float) + 
                                 img_l[:, x_left].astype(float) + 
                                 img_r[:, x_right].astype(float)) / 3.0
                
                volume[valid_y, x, z] = fused_thermal[valid_y]

    # --- 4. VISUALIZATION ---
    print("Generating Integrated 3D Patient Model...")
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape)
    grid.spacing = (1, 1, 1)
    grid.point_data["Integrated_Thermal_Data"] = volume.flatten(order="F")

    plotter = pv.Plotter()
    plotter.set_background("black")
    
    # Volume rendering (shows the solid 3D mass)
    plotter.add_volume(grid, cmap="inferno", opacity="linear", shade=True)
    
    # Clipping plane for inspection
    plotter.add_mesh_clip_plane(grid, scalars="Integrated_Thermal_Data", cmap="inferno")
    
    plotter.add_text(f"INTEGRATED 3D MODEL: {patient_id}", font_size=12)
    plotter.show()

if __name__ == "__main__":
    integrate_multiview_volume()