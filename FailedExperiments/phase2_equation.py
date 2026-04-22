import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def solve_pennes_numpy(a, b, c, tumor_pos, tumor_r, res=2.0):
    """
    Menghitung distribusi suhu 3D menggunakan Finite Difference Method.
    a, b, c: Semi-axes ellipsoid (mm)
    tumor_pos: (x, y, z) relatif terhadap pusat basis (mm)
    res: Resolusi grid (mm/voxel)
    """
    # 1. Inisialisasi Grid
    nx, ny, nz = int(2*a/res)+2, int(2*b/res)+2, int(c/res)+2
    T = np.full((nx, ny, nz), 37.0) # Suhu inti tubuh
    
    # Grid koordinat
    x, y, z = np.indices((nx, ny, nz))
    xf, yf, zf = (x - nx/2)*res, (y - ny/2)*res, z*res
    
    # 2. Masking Jaringan
    breast_mask = (xf/a)**2 + (yf/b)**2 + (zf/c)**2 <= 1.0
    dist_tumor = np.sqrt((xf-tumor_pos[0])**2 + (yf-tumor_pos[1])**2 + (zf-tumor_pos[2])**2)
    tumor_mask = dist_tumor <= tumor_r
    
    # 3. Parameter Fisiologis (Agnelli 2011)
    K = np.where(tumor_mask, 0.75, 0.5)    # Conductivity
    P = np.where(tumor_mask, 7992.4, 1998.1) # Perfusion
    Q = np.where(tumor_mask, 42000, 4200)    # Metabolic
    
    # 4. Iterative Solver (Jacobi)
    h2 = res**2
    Ta = 25.0 # Ambient temp
    h_conv = 10.0 # Heat transfer coefficient
    
    print(f"Simulasi Bioheat untuk volume {nx}x{ny}x{nz}...")
    for _ in range(150): # Iterasi hingga stabil
        T_old = T.copy()
        
        # Interior voxels (Pennes FDM update)
        denom = (6*K + h2*P)
        T[1:-1, 1:-1, 1:-1] = (K[1:-1, 1:-1, 1:-1] * (
            T_old[2:, 1:-1, 1:-1] + T_old[:-2, 1:-1, 1:-1] +
            T_old[1:-1, 2:, 1:-1] + T_old[1:-1, :-2, 1:-1] +
            T_old[1:-1, 1:-1, 2:] + T_old[1:-1, 1:-1, :-2]
        ) + h2 * (P[1:-1, 1:-1, 1:-1]*37.0 + Q[1:-1, 1:-1, 1:-1])) / denom[1:-1, 1:-1, 1:-1]
        
        # Pastikan suhu di luar payudara tidak dihitung (Boundary)
        T[~breast_mask] = Ta
        
    return T, breast_mask

def project_to_5_views(T_vol, a, b, c, res):
    """
    Memproyeksikan suhu permukaan ke 5 sudut DMR-IR (0, +/-45, +/-90)
    """
    surface_temp = np.max(T_vol, axis=2) # Sederhananya mengambil suhu terluar
    
    angles = [0, 45, 90, -45, -90]
    names = ["Front", "R45", "R90", "L45", "L90"]
    
    plt.figure(figsize=(15, 3))
    for i, angle in enumerate(angles):
        plt.subplot(1, 5, i+1)
        # Rotasi simulasi sederhana (dalam riset asli gunakan rotasi 3D voxel)
        rotated = np.roll(surface_temp, int(angle/res), axis=1) 
        plt.imshow(rotated, cmap='inferno')
        plt.title(names[i])
        plt.axis('off')
    plt.show()

# --- Main Execution ---
df = pd.read_csv('geometry.csv')
sample = df.iloc[0] # Ambil pasien pertama

# Simulasi tumor di kuadran atas (Upper Outer Quadrant)
tumor_xyz = (20, 20, 15) # mm dari pusat
tumor_r = 10.0 # 1 cm radius

T_volume, mask = solve_pennes_numpy(sample['a_mm'], sample['b_mm'], sample['c_mm'], tumor_xyz, tumor_r)
project_to_5_views(T_volume, sample['a_mm'], sample['b_mm'], sample['c_mm'], res=2.0)