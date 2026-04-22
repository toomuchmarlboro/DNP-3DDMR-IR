"""
===========================================================================
  3D THERMAL PATIENT RECONSTRUCTION FOR ONCOLOGY ANALYSIS
  -------------------------------------------------------
  Pipeline:
    1. Data Acquisition    — DMR-IR dataset (SemilleroCV/HuggingFace)
    2. Segmentation        — Marker-Controlled Watershed Transform
    3. 3D Reconstruction   — Visual Hull via Space Carving (Voxel Grid)
    4. Thermal Fusion      — Surface temperature mapping
    5. Bioheat Solver      — Steady-state Pennes Bioheat Equation (FD)
    6. Visualization       — PyVista + Matplotlib summary figure
===========================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, label as nd_label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import os

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PATIENT_ID     = 194
GRID_SIZE      = (64, 64, 64)        # (Nx, Ny, Nz) voxels
VOXEL_SIZE_M   = 0.005               # 5 mm/voxel → 32 cm torso span
CORE_TEMP      = 37.0                # °C arterial / deep core

# Camera angle per view code (degrees, rotation around Y-axis)
VIEW_ANGLES = {0: 0.0, 1: 45.0, 2: -45.0, 3: 90.0, 4: -90.0}
VIEW_NAMES  = {0: "Frontal", 1: "Right_45°", 2: "Left_45°",
               3: "Right_90°", 4: "Left_90°"}

# ── Pennes Bioheat Equation parameters ──────────────────────────────────────
K_TISSUE   = 0.50     # W/(m·K)   thermal conductivity of tissue
RHO_B      = 1060.0   # kg/m³     blood density
C_B        = 3600.0   # J/(kg·K)  blood specific heat capacity
OMEGA_B    = 0.0005   # 1/s       blood perfusion rate
T_ARTERIAL = 37.0     # °C        arterial blood temperature
Q_METABOLIC= 420.0    # W/m³      basal metabolic heat generation

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────

def load_patient_views(patient_id: int) -> dict:
    """
    Attempt to load from HuggingFace DMR-IR dataset across all splits.
    Uses robust string matching to catch various naming conventions.
    Falls back to physics-based synthetic generation if unavailable.
    """
    print(f"\n{'─'*70}")
    print(f"  STEP 1 │ DATA ACQUISITION   (Patient {patient_id})")
    print(f"{'─'*70}")

    try:
        from datasets import load_dataset
        print("  ► Loading SemilleroCV/DMR-IR from HuggingFace (ALL splits) …")
        
        # Load the entire DatasetDict (train, test, validation)
        ds_dict = load_dataset("SemilleroCV/DMR-IR", trust_remote_code=True)

        # Dictionary mapping dataset string labels to our internal 0-4 keys
        view_mapper = {
            "frontal": 0, "front": 0, "anterior": 0, "0": 0,
            "right_oblique": 1, "rightoblique": 1, "r45": 1, "1": 1,
            "left_oblique": 2, "leftoblique": 2, "l45": 2, "2": 2,
            "right_lateral": 3, "rightlateral": 3, "r90": 3, "3": 3,
            "left_lateral": 4, "leftlateral": 4, "l90": 4, "4": 4
        }

        views = {}
        patient_found_count = 0

        # Loop through every split (train, test, validation)
        for split_name, ds_split in ds_dict.items():
            for sample in ds_split:
                # Robustly extract ID (catch variations like "194", "P194", "ID_194")
                pid = str(sample.get('patient_id') or sample.get('ID') or sample.get('id', '')).strip()
                
                if str(patient_id) not in pid:
                    continue
                
                patient_found_count += 1

                # Robustly extract View Code
                v_raw = str(sample.get('view') or sample.get('position') or sample.get('view_code', ''))
                v_clean = v_raw.lower().replace(" ", "").replace("°", "")

                # Match the string to our internal codes (0, 1, 2, 3, 4)
                internal_code = None
                for key, val in view_mapper.items():
                    if key in v_clean:
                        internal_code = val
                        break

                if internal_code is not None and internal_code not in views:
                    img = sample.get('image') or sample.get('thermal_image')
                    if img is not None:
                        views[internal_code] = {
                            'image': np.array(img),
                            'label': sample.get('label', 0),
                            'angle': VIEW_ANGLES[internal_code],
                            'name' : VIEW_NAMES[internal_code],
                            'source': 'dataset'
                        }

        if views:
            print(f"    ✓ Found {len(views)} real views for Patient {patient_id}: {[v['name'] for v in views.values()]}")
            return views
        elif patient_found_count > 0:
            print(f"    ✗ Patient {patient_id} found {patient_found_count} times, but no recognized view angles matched.")
        else:
            print(f"    ✗ Patient {patient_id} was not found in any dataset split.")

    except Exception as e:
        print(f"    ✗ Dataset load error: {e}")

    print("  ► Generating physics-based synthetic thermal data …")
    return _generate_synthetic_views(patient_id)


def _generate_synthetic_views(patient_id: int) -> dict:
    """
    Synthesise realistic IR thermal views of a female torso
    with a sub-surface heat source (simulated tumour).
    The temperature pattern follows a smoothed Pennes forward solution.
    """
    np.random.seed(patient_id % 1000)
    H, W = 128, 96
    views = {}

    # Tumour ground-truth position (normalised [-1,1])
    tumour_world = np.array([0.15, 0.10, 0.05])   # slightly right, upper
    tumour_depth = 0.20                             # normalised depth into torso
    tumour_radius = 0.12
    tumour_delta_T = 4.5                            # °C above core at surface

    for code, angle_deg in VIEW_ANGLES.items():
        img = np.full((H, W), 28.0, dtype=np.float32)   # ambient background

        # ── Torso silhouette via ellipse ───────────────────────────────
        cx, cy = W / 2, H * 0.55
        ra, rb = W * 0.36, H * 0.40                      # semi-axes
        Y_g, X_g = np.mgrid[0:H, 0:W]
        ellipse = ((X_g - cx)**2 / ra**2 + (Y_g - cy)**2 / rb**2)
        body_mask_bool = ellipse < 1.0

        # ── Skin temperature gradient (edges cooler) ───────────────────
        dist_norm = np.clip(1.0 - ellipse, 0, 1)
        skin_T    = 34.5 + 2.0 * dist_norm**0.5           # 34.5 → 36.5 °C
        img[body_mask_bool] = skin_T[body_mask_bool]

        # ── Project tumour onto this camera view ───────────────────────
        theta = np.radians(angle_deg)
        Ry = np.array([[ np.cos(theta), 0, np.sin(theta)],
                        [ 0,             1, 0            ],
                        [-np.sin(theta), 0, np.cos(theta)]])
        t_rotated = Ry @ tumour_world

        # Map to pixel space
        tx_px = (t_rotated[0] + 1) / 2 * (W - 1)
        ty_px = (1 - (t_rotated[1] + 1) / 2) * (H - 1)

        # z-component indicates whether tumour faces this camera
        visibility = np.abs(t_rotated[2])   # 0→fully visible, 1→side-on

        # Surface heat spot (attenuated by depth and visibility angle)
        sigma = ra * (0.25 + tumour_depth * 0.5)
        heat_spot = tumour_delta_T * np.exp(
            -((X_g - tx_px)**2 + (Y_g - ty_px)**2) / (2 * sigma**2)
        ) * np.clip(1.0 - visibility * 0.4, 0.3, 1.0)

        img[body_mask_bool] += heat_spot[body_mask_bool]

        # ── Sensor noise ───────────────────────────────────────────────
        img += np.random.normal(0, 0.12, (H, W))

        views[code] = {
            'image' : img,
            'label' : 1,         # malignant
            'angle' : angle_deg,
            'name'  : VIEW_NAMES[code],
            'source': 'synthetic'
        }

    print(f"    ✓ Synthetic views generated ({len(views)} × {H}×{W} px)")
    return views


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — WATERSHED SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalise arbitrary-range image to uint8 [0, 255]."""
    f = img.astype(np.float32)
    rng = float(f.max()) - float(f.min())
    if rng > 0:
        return ((f - f.min()) / rng * 255).astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)


def _parse_thermal(img_array: np.ndarray):
    """Extract (gray_uint8, temp_map_celsius) from any IR image format."""
    if img_array.ndim == 3:
        if img_array.shape[2] >= 3:
            # RGB thermal pseudo-colour → convert to grey intensity
            gray_u8 = cv2.cvtColor(
                img_array[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            temp = gray_u8.astype(np.float32) / 255.0 * 15.0 + 30.0
        else:
            gray_u8 = _to_uint8(img_array[:, :, 0])
            temp = img_array[:, :, 0].astype(np.float32)
    else:
        arr = img_array.astype(np.float32)
        temp = arr
        # Re-scale to Celsius if raw range is outside 25–50 °C
        if arr.max() > 100 or arr.max() < 1:
            temp = arr / arr.max() * 15.0 + 30.0 if arr.max() > 0 else arr + 30.0
        gray_u8 = _to_uint8(arr)

    return gray_u8, temp.astype(np.float32)


def watershed_segment(img_array: np.ndarray):
    """
    Marker-Controlled Watershed segmentation.

    Returns
    -------
    mask    : uint8 ndarray  (1 = body, 0 = background)
    temp_C  : float32 ndarray  temperature map in Celsius
    """
    gray_u8, temp_C = _parse_thermal(img_array)

    # ── 1. Denoise ─────────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(gray_u8, (7, 7), 2.0)

    # ── 2. Otsu foreground / background seed ───────────────────────────
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)

    # ── 3. Distance transform → sure foreground markers ────────────────
    dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    fg_thresh = 0.35 * dist.max() if dist.max() > 0 else 1.0
    _, sure_fg = cv2.threshold(dist, fg_thresh, 255, 0)
    sure_fg    = sure_fg.astype(np.uint8)

    # ── 4. Sure background (heavy dilation) ────────────────────────────
    sure_bg = cv2.dilate(closed, kernel, iterations=4)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # ── 5. Markers ─────────────────────────────────────────────────────
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1                # reserve 0 for unknown
    markers[unknown == 255] = 0

    # ── 6. Watershed ───────────────────────────────────────────────────
    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(bgr, markers.copy())

    body_raw = np.zeros_like(gray_u8, dtype=np.uint8)
    body_raw[markers_ws > 1] = 1

    # ── 7. Post-process: fill holes + largest CC ────────────────────────
    body_filled = ndimage.binary_fill_holes(body_raw).astype(np.uint8)
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(
        body_filled, connectivity=8)

    if n_lbl > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        body_filled = (lbl_map == largest).astype(np.uint8)

    return body_filled, temp_C


def segment_all_views(patient_views: dict) -> dict:
    print(f"\n{'─'*70}")
    print(f"  STEP 2 │ WATERSHED SEGMENTATION")
    print(f"{'─'*70}")
    segmented = {}
    for code, v in sorted(patient_views.items()):
        mask, temp = watershed_segment(v['image'])
        pct = 100.0 * mask.sum() / mask.size
        print(f"    {v['name']:12s}  coverage={pct:5.1f}%  "
              f"T=[{temp[mask>0].min():.1f}, {temp[mask>0].max():.1f}] °C")
        segmented[code] = {**v, 'mask': mask, 'temp_C': temp}
    return segmented


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SPACE CARVING (VISUAL HULL)
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_matrix_y(angle_deg: float) -> np.ndarray:
    """4 × 4 homogeneous rotation matrix around the Y-axis."""
    t   = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]], dtype=np.float32)


def _project_voxels(coords_norm: np.ndarray,
                    angle_deg: float,
                    H: int, W: int):
    """
    Orthographic projection of N normalised voxel centres onto the
    camera image plane at the given Y-rotation angle.

    Parameters
    ----------
    coords_norm : (N, 3)  in [-1, 1]³
    Returns:  u (N,), v (N,) — integer pixel coordinates clamped to [0, W-1] × [0, H-1]
    """
    R = _rotation_matrix_y(angle_deg)[:3, :3]
    rotated = (R @ coords_norm.T).T          # (N, 3)

    # Orthographic: camera looks along +Z, image plane = XY
    u = np.round((rotated[:, 0] + 1) / 2 * (W - 1)).astype(int)
    v = np.round((1 - (rotated[:, 1] + 1) / 2) * (H - 1)).astype(int)

    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    return u, v


def space_carving(segmented_views: dict,
                  grid_size: tuple = GRID_SIZE):
    """
    Visual Hull via strict Boolean intersection.
    Voxel is TISSUE iff it projects inside the body mask in every view.
    """
    print(f"\n{'─'*70}")
    print(f"  STEP 3 │ SPACE CARVING  (grid {grid_size[0]}³)")
    print(f"{'─'*70}")

    Nx, Ny, Nz = grid_size
    total = Nx * Ny * Nz

    # Normalised voxel centre coordinates in [-1, 1]
    xs = np.linspace(-1, 1, Nx)
    ys = np.linspace(-1, 1, Ny)
    zs = np.linspace(-1, 1, Nz)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')   # (Nx, Ny, Nz)
    coords_flat = np.stack(
        [XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)      # (N, 3)

    voxel_flat = np.ones(total, dtype=bool)  # start: all tissue

    for code, v in sorted(segmented_views.items()):
        mask = v['mask']
        H, W = mask.shape
        u, proj_v = _project_voxels(coords_flat, v['angle'], H, W)

        in_mask = mask[proj_v, u].astype(bool)
        voxel_flat &= in_mask

        remaining = voxel_flat.sum()
        print(f"    ✓ Carved with {v['name']:12s} → {remaining:6d} voxels "
              f"({100*remaining/total:.1f}%) remain")

    voxel_grid = voxel_flat.reshape(grid_size)

    # ── Clean-up: remove isolated specks ──────────────────────────────
    labeled, n_cc = nd_label(voxel_grid)
    if n_cc > 1:
        sizes = np.array([
            (labeled == i).sum() for i in range(1, n_cc + 1)])
        largest_id = sizes.argmax() + 1
        voxel_grid = labeled == largest_id
        print(f"    ✓ Kept largest connected component "
              f"({voxel_grid.sum()} voxels after cleanup)")

    coord_grid = np.stack(
        [XX, YY, ZZ], axis=-1)   # (Nx, Ny, Nz, 3)

    return voxel_grid.astype(bool), coord_grid


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — THERMAL FUSION
# ─────────────────────────────────────────────────────────────────────────────

def fuse_thermal_data(voxel_grid: np.ndarray,
                      coord_grid: np.ndarray,
                      segmented_views: dict,
                      grid_size: tuple = GRID_SIZE):
    """
    Map 2-D surface temperatures onto 3-D surface voxels.
    Interior voxels are initialised to CORE_TEMP.
    Temperature is averaged across all views projecting onto each voxel.
    """
    print(f"\n{'─'*70}")
    print(f"  STEP 4 │ THERMAL FUSION")
    print(f"{'─'*70}")

    T_vol   = np.full(grid_size, CORE_TEMP, dtype=np.float32)
    T_sum   = np.zeros(grid_size, dtype=np.float32)
    T_count = np.zeros(grid_size, dtype=np.int32)

    # Surface = tissue voxels that lose at least one neighbour after erosion
    eroded  = binary_erosion(voxel_grid, iterations=1)
    surface = voxel_grid & ~eroded

    # Indices of surface voxels
    surf_idx = np.argwhere(surface)           # (M, 3)
    surf_coords = coord_grid[
        surf_idx[:, 0], surf_idx[:, 1], surf_idx[:, 2]]   # (M, 3)

    for code, v in sorted(segmented_views.items()):
        H, W = v['mask'].shape
        temp_C = v['temp_C']

        u, pv = _project_voxels(surf_coords, v['angle'], H, W)
        sampled = temp_C[pv, u]               # (M,)

        for m, (ix, iy, iz) in enumerate(surf_idx):
            T_sum[ix, iy, iz]   += sampled[m]
            T_count[ix, iy, iz] += 1

    valid = T_count > 0
    T_vol[valid] = T_sum[valid] / T_count[valid]
    T_vol[~voxel_grid] = np.nan               # outside body → NaN

    print(f"    Surface voxels  : {surface.sum():,}")
    print(f"    Interior voxels : {(voxel_grid & ~surface).sum():,}")
    surf_T = T_vol[surface]
    print(f"    Surface T range : {np.nanmin(surf_T):.2f} – "
          f"{np.nanmax(surf_T):.2f} °C  (mean {np.nanmean(surf_T):.2f})")

    return T_vol, surface


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — PENNES BIOHEAT EQUATION  (Steady-State FD Solver)
# ─────────────────────────────────────────────────────────────────────────────
#
#   ∇·(k∇T) − ρb·cb·ωb·(T − Ta) + Qm = 0
#
#   Discretised on a uniform Cartesian grid:
#   k/dx² · Σ_neighbours T_i  −  (6·k/dx² + ρb·cb·ωb)·T  +  ρb·cb·ωb·Ta + Qm = 0
#
#   Rearranging for T at each interior voxel:
#   T = [k/dx²·Σ T_neighbours + ρb·cb·ωb·Ta + Qm]  /  [6·k/dx² + ρb·cb·ωb]
#
#   Boundary condition: surface voxels fixed to fused surface temperatures.
# ─────────────────────────────────────────────────────────────────────────────

def solve_pennes_bioheat(T_init: np.ndarray,
                         voxel_grid: np.ndarray,
                         surface: np.ndarray,
                         n_iter: int = 200,
                         tol: float = 1e-5):
    """
    Iterative Gauss-Seidel solver for steady-state Pennes Bioheat Equation.

    Parameters
    ----------
    T_init    : (Nx, Ny, Nz) initial temperature field
    voxel_grid: (Nx, Ny, Nz) tissue mask
    surface   : (Nx, Ny, Nz) surface voxel mask (fixed BCs)

    Returns
    -------
    T_solved  : (Nx, Ny, Nz) converged temperature field
    tumor_mask: (Nx, Ny, Nz) predicted tumour / hot-spot mask
    """
    print(f"\n{'─'*70}")
    print(f"  STEP 5 │ PENNES BIOHEAT SOLVER")
    print(f"{'─'*70}")

    dx   = VOXEL_SIZE_M
    α    = K_TISSUE / dx**2             # conduction coefficient
    β    = RHO_B * C_B * OMEGA_B       # blood perfusion coefficient
    denom = 6.0 * α + β                # denominator for GS update
    rhs_const = β * T_ARTERIAL + Q_METABOLIC   # constant RHS term

    T = T_init.copy()
    T[np.isnan(T)] = CORE_TEMP          # initialise NaN (non-tissue) to core

    interior = voxel_grid & ~surface

    print(f"    k={K_TISSUE} W/(m·K) | ωb={OMEGA_B} 1/s | "
          f"Qm={Q_METABOLIC} W/m³ | dx={dx*1000:.0f} mm")
    print(f"    α={α:.2e}  β={β:.4f}  denom={denom:.4f}")

    T_BC = T.copy()    # snapshot of surface BCs

    # ── Optional: synthesise a spatially-varying Qm (embedded tumour source)
    # This models a malignant nodule with ~10× elevated metabolic activity.
    # The centroid is estimated from the surface temperature hot-spot.
    Nx, Ny, Nz = voxel_grid.shape
    Qm_field = np.full(voxel_grid.shape, Q_METABOLIC, dtype=np.float32)

    # Locate surface temperature peak → project inward for tumour centroid
    T_surf_only = np.where(surface, T_BC, -np.inf)
    if T_surf_only.max() > CORE_TEMP + 0.5:
        pk_idx = np.unravel_index(np.argmax(T_surf_only), T_surf_only.shape)
        # Place hot-spot 20% of the way toward the body centre
        centre = np.array([Nx//2, Ny//2, Nz//2], dtype=float)
        pk     = np.array(pk_idx, dtype=float)
        tumour_vox = (pk + (centre - pk) * 0.20).astype(int)
        tumour_vox = np.clip(tumour_vox, 0,
                             np.array([Nx-1, Ny-1, Nz-1]))
        r_vox = 4.0   # ~2 cm radius tumour
        Ygr, Xgr, Zgr = np.mgrid[0:Nx, 0:Ny, 0:Nz]
        dist_sq = ((Xgr - tumour_vox[0])**2 +
                   (Ygr - tumour_vox[1])**2 +
                   (Zgr - tumour_vox[2])**2)
        tumour_region = dist_sq < r_vox**2
        Qm_field[tumour_region & voxel_grid] = Q_METABOLIC * 12.0
        print(f"    Tumour hot-source placed at voxel "
              f"({tumour_vox[0]}, {tumour_vox[1]}, {tumour_vox[2]})  "
              f"({tumour_region.sum()} voxels  ×12 Qm)")

    for it in range(1, n_iter + 1):
        T_old = T.copy()

        # ── Gauss-Seidel sweep (vectorised interior slice) ─────────────
        T[1:-1, 1:-1, 1:-1] = np.where(
            interior[1:-1, 1:-1, 1:-1],
            (α * (T[2:,  1:-1, 1:-1] + T[:-2, 1:-1, 1:-1] +
                  T[1:-1, 2:,  1:-1] + T[1:-1, :-2, 1:-1] +
                  T[1:-1, 1:-1, 2:  ] + T[1:-1, 1:-1, :-2])
             + β * T_ARTERIAL + Qm_field[1:-1, 1:-1, 1:-1]) / denom,
            T[1:-1, 1:-1, 1:-1]
        )

        # ── Restore surface boundary conditions ────────────────────────
        T[surface] = T_BC[surface]

        # ── Convergence check ──────────────────────────────────────────
        diff = np.abs(T - T_old)
        max_d = diff[voxel_grid].max()

        if it % 50 == 0:
            print(f"    iter {it:4d}/{n_iter}  max Δ = {max_d:.2e} °C")

        if max_d < tol:
            print(f"    ✓ Converged at iteration {it}  (max Δ = {max_d:.2e} °C)")
            break

    # ── Mask non-tissue ────────────────────────────────────────────────
    T_out = T.copy()
    T_out[~voxel_grid] = np.nan

    # ── Tumour / hot-spot detection ────────────────────────────────────
    T_thresh   = CORE_TEMP + 1.0         # > 38.0 °C ≡ suspicious
    tumor_mask = (T_out > T_thresh) & voxel_grid

    print(f"\n    ── Post-solve summary ──────────────────────────────────")
    print(f"    Tissue T range  : "
          f"{np.nanmin(T_out):.2f} – {np.nanmax(T_out):.2f} °C")
    print(f"    Hot-spot thresh : {T_thresh:.1f} °C")
    print(f"    Hot-spot voxels : {tumor_mask.sum():,}")
    if tumor_mask.sum() > 0:
        com = ndimage.center_of_mass(tumor_mask)
        print(f"    Centroid voxel  : ({com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f})")
        T_grid = GRID_SIZE[0] / 2
        depth_norm = abs(com[2] - T_grid) / T_grid
        depth_cm   = depth_norm * GRID_SIZE[2] * VOXEL_SIZE_M * 100
        print(f"    Est. depth      : {depth_cm:.1f} cm from body centre")

    return T_out, tumor_mask


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

# ── Colour palette ─────────────────────────────────────────────────────────
_DARK_BG   = '#0d0d14'
_PANEL_BG  = '#12121e'
_ACCENT    = '#00e5ff'
_RED_WARM  = '#ff4b4b'
_TEXT      = '#dce3ec'
_MUTED     = '#8899aa'

_thermal_cmap = matplotlib.colormaps['inferno']
_risk_cmap    = LinearSegmentedColormap.from_list(
    'risk', [(0, 0, 0, 0), '#ff9900', '#ff3300', '#cc0000'])


def _ax_style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(_PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a3e')
    ax.tick_params(colors=_MUTED, labelsize=7)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=8, pad=4, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, color=_MUTED, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=_MUTED, fontsize=7)


def _colorbar(mappable, ax, label=''):
    cb = plt.colorbar(mappable, ax=ax, fraction=0.038, pad=0.03)
    cb.ax.yaxis.set_tick_params(color=_MUTED, labelsize=6)
    cb.set_label(label, color=_MUTED, fontsize=6)
    cb.outline.set_edgecolor('#2a2a3e')
    return cb


def visualise_all(patient_views: dict,
                  segmented_views: dict,
                  voxel_grid: np.ndarray,
                  T_vol: np.ndarray,
                  tumor_mask: np.ndarray,
                  patient_id: int = PATIENT_ID,
                  output_dir: str = '/mnt/user-data/outputs'):
    """
    Comprehensive 5-section matplotlib dashboard:
      A — Raw thermal views
      B — Watershed segmentation masks
      C — 3-D projection views (frontal / sagittal / axial slices)
      D — Bioheat temperature field cross-sections
      E — Tumour risk overlay + statistics
    """
    print(f"\n{'─'*70}")
    print(f"  STEP 6 │ VISUALISATION")
    print(f"{'─'*70}")

    Nx, Ny, Nz = GRID_SIZE
    T_clean = np.where(voxel_grid, np.nan_to_num(T_vol, nan=CORE_TEMP), 0.0)

    n_views = len(patient_views)
    cols_per_row = max(n_views, 5)

    fig = plt.figure(figsize=(22, 22), facecolor=_DARK_BG)
    fig.text(0.5, 0.98,
             f"3D Thermal Digital Twin — Patient {patient_id} "
             f"│ Oncology Analysis Dashboard",
             ha='center', va='top', color=_ACCENT,
             fontsize=15, fontweight='bold', family='monospace')
    fig.text(0.5, 0.965,
             "Pennes Bioheat Equation  •  Space-Carving Visual Hull  "
             "•  Marker-Controlled Watershed",
             ha='center', va='top', color=_MUTED, fontsize=9)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SECTION A — Raw thermal views
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for i, (code, v) in enumerate(sorted(patient_views.items())):
        ax = fig.add_subplot(6, cols_per_row, i + 1)
        img = v['image']
        if img.ndim == 3:
            im = ax.imshow(img[:, :, :3])
        else:
            im = ax.imshow(img, cmap='inferno', vmin=28, vmax=42)
        _colorbar(im, ax, '°C')
        _ax_style(ax, title=f"A{i+1}  {v['name']}")
        ax.axis('off')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SECTION B — Watershed masks
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for i, (code, v) in enumerate(sorted(segmented_views.items())):
        ax = fig.add_subplot(6, cols_per_row, cols_per_row + i + 1)
        mask = v['mask']
        temp = v['temp_C']
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        t_n = (temp - 30) / 15.0
        t_n = np.clip(t_n, 0, 1)
        rgba = _thermal_cmap(t_n)
        overlay[:, :, :3] = rgba[:, :, :3]
        overlay[:, :, 3]  = mask.astype(float) * 0.9 + 0.1
        ax.imshow(overlay)
        # Contour
        if mask.max() > 0:
            ax.contour(mask, levels=[0.5], colors=[_ACCENT], linewidths=0.8, alpha=0.9)
        _ax_style(ax, title=f"B{i+1}  Mask  {v['name']}")
        ax.axis('off')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SECTION C — Visual Hull projections
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    c_row = 2 * cols_per_row + 1  # starting subplot index for row 3

    # Frontal (XY) depth projection
    ax_f = fig.add_subplot(6, 3, 7)
    proj_xy = voxel_grid.sum(axis=2).astype(float)
    proj_xy[proj_xy == 0] = np.nan
    im_f = ax_f.imshow(proj_xy.T, cmap='viridis', origin='lower')
    _colorbar(im_f, ax_f, 'depth [vox]')
    _ax_style(ax_f, "C1  Frontal Projection (Visual Hull)", "X vox", "Y vox")

    # Sagittal (YZ) projection
    ax_s = fig.add_subplot(6, 3, 8)
    proj_yz = voxel_grid.sum(axis=0).astype(float)
    proj_yz[proj_yz == 0] = np.nan
    im_s = ax_s.imshow(proj_yz.T, cmap='viridis', origin='lower')
    _colorbar(im_s, ax_s, 'depth [vox]')
    _ax_style(ax_s, "C2  Sagittal Projection (Visual Hull)", "Y vox", "Z vox")

    # Axial (XZ) projection
    ax_a = fig.add_subplot(6, 3, 9)
    proj_xz = voxel_grid.sum(axis=1).astype(float)
    proj_xz[proj_xz == 0] = np.nan
    im_a = ax_a.imshow(proj_xz.T, cmap='viridis', origin='lower')
    _colorbar(im_a, ax_a, 'depth [vox]')
    _ax_style(ax_a, "C3  Axial Projection (Visual Hull)", "X vox", "Z vox")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SECTION D — Bioheat temperature cross-sections
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    vmin_T, vmax_T = 35.5, 40.5

    # Axial slice at tumour centroid Z
    if tumor_mask.sum() > 0:
        com = ndimage.center_of_mass(tumor_mask)
        slx = int(np.clip(com[0], 0, Nx-1))
        sly = int(np.clip(com[1], 0, Ny-1))
        slz = int(np.clip(com[2], 0, Nz-1))
    else:
        slx, sly, slz = Nx//2, Ny//2, Nz//2

    ax_d1 = fig.add_subplot(6, 3, 10)
    sl_z = T_clean[:, :, slz].copy()
    sl_z[~voxel_grid[:, :, slz]] = np.nan
    im_d1 = ax_d1.imshow(sl_z.T, cmap='inferno', origin='lower',
                          vmin=vmin_T, vmax=vmax_T)
    if tumor_mask[:, :, slz].sum() > 0:
        ax_d1.contour(tumor_mask[:, :, slz].T, levels=[0.5],
                      colors=[_RED_WARM], linewidths=1.2, linestyles='--')
    _colorbar(im_d1, ax_d1, '°C')
    _ax_style(ax_d1, f"D1  Axial Slice  z={slz}  (Bioheat)", "X", "Y")

    ax_d2 = fig.add_subplot(6, 3, 11)
    sl_x = T_clean[slx, :, :].copy()
    sl_x[~voxel_grid[slx, :, :]] = np.nan
    im_d2 = ax_d2.imshow(sl_x.T, cmap='inferno', origin='lower',
                          vmin=vmin_T, vmax=vmax_T)
    if tumor_mask[slx, :, :].sum() > 0:
        ax_d2.contour(tumor_mask[slx, :, :].T, levels=[0.5],
                      colors=[_RED_WARM], linewidths=1.2, linestyles='--')
    _colorbar(im_d2, ax_d2, '°C')
    _ax_style(ax_d2, f"D2  Sagittal Slice  x={slx}  (Bioheat)", "Y", "Z")

    ax_d3 = fig.add_subplot(6, 3, 12)
    sl_y = T_clean[:, sly, :].copy()
    sl_y[~voxel_grid[:, sly, :]] = np.nan
    im_d3 = ax_d3.imshow(sl_y.T, cmap='inferno', origin='lower',
                          vmin=vmin_T, vmax=vmax_T)
    if tumor_mask[:, sly, :].sum() > 0:
        ax_d3.contour(tumor_mask[:, sly, :].T, levels=[0.5],
                      colors=[_RED_WARM], linewidths=1.2, linestyles='--')
    _colorbar(im_d3, ax_d3, '°C')
    _ax_style(ax_d3, f"D3  Coronal Slice  y={sly}  (Bioheat)", "X", "Z")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  SECTION E — Risk map + histogram + stats panel
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # E1 — Tumour risk overlay (frontal)
    ax_e1 = fig.add_subplot(6, 3, 13)
    tissue_front = voxel_grid.sum(axis=2).astype(float)
    risk_front   = tumor_mask.sum(axis=2).astype(float)
    tissue_front[tissue_front == 0] = np.nan
    risk_front[risk_front   == 0] = np.nan

    ax_e1.imshow(tissue_front.T, cmap='bone', origin='lower', alpha=0.7, aspect='auto')
    if not np.all(np.isnan(risk_front)):
        ax_e1.imshow(risk_front.T, cmap=_risk_cmap, origin='lower',
                     alpha=0.9, vmin=0.1, aspect='auto')
    patch_risk = mpatches.Patch(color='#ff4400', label=f'Hot-spot ({tumor_mask.sum()} vox)')
    ax_e1.legend(handles=[patch_risk], loc='upper right',
                  fontsize=6, framealpha=0.3, labelcolor=_TEXT)
    _ax_style(ax_e1, "E1  Tumour Risk Map (Frontal)")

    # E2 — Temperature histogram
    ax_e2 = fig.add_subplot(6, 3, 14)
    T_tissue = T_clean[voxel_grid]
    T_tissue = T_tissue[T_tissue > 0]
    ax_e2.hist(T_tissue, bins=60, color='#f97316', edgecolor='none',
               alpha=0.85, density=True, label='Tissue temperature')
    ax_e2.axvline(CORE_TEMP,       color='#22d3ee', lw=1.5, ls='--', label=f'Core ({CORE_TEMP:.1f}°C)')
    ax_e2.axvline(CORE_TEMP + 1.0, color=_RED_WARM,  lw=1.5, ls='--', label='Risk threshold (38.0°C)')
    ax_e2.set_xlim([35, 42])
    ax_e2.legend(fontsize=6, framealpha=0.2, labelcolor=_TEXT)
    _ax_style(ax_e2, "E2  Temperature Distribution (Bioheat)", "T (°C)", "Density")

    # E3 — Stats panel
    ax_e3 = fig.add_subplot(6, 3, 15)
    ax_e3.set_facecolor('#0d1117')
    ax_e3.axis('off')
    T_surf = np.nanmean(T_vol[~binary_erosion(voxel_grid, iterations=1) & voxel_grid])
    T_max  = float(np.nanmax(T_vol[voxel_grid])) if voxel_grid.any() else np.nan
    T_mean = float(np.nanmean(T_vol[voxel_grid])) if voxel_grid.any() else np.nan
    if tumor_mask.sum() > 0:
        com    = ndimage.center_of_mass(tumor_mask)
        depth_cm = abs(com[2] - Nz/2) * VOXEL_SIZE_M * 100
        vol_cm3  = tumor_mask.sum() * (VOXEL_SIZE_M*100)**3
    else:
        depth_cm, vol_cm3 = 0.0, 0.0

    lines = [
        ("PATIENT",         f"{patient_id}",         _ACCENT),
        ("Views processed", f"{n_views}",             _TEXT),
        ("Voxel grid",      f"{GRID_SIZE[0]}³",        _TEXT),
        ("Voxel size",      f"{VOXEL_SIZE_M*10:.0f} mm",  _TEXT),
        ("Tissue voxels",   f"{voxel_grid.sum():,}",  _TEXT),
        ("",                "",                        _TEXT),
        ("THERMAL",         "─────────────",          _ACCENT),
        ("Surface T mean",  f"{T_surf:.2f} °C",       _TEXT),
        ("Interior T mean", f"{T_mean:.2f} °C",       _TEXT),
        ("Peak T",          f"{T_max:.2f} °C",        _RED_WARM if T_max > 38.5 else _TEXT),
        ("",                "",                        _TEXT),
        ("TUMOUR ANALYSIS", "─────────────",          _ACCENT),
        ("Hot-spot voxels", f"{tumor_mask.sum():,}",  _RED_WARM if tumor_mask.sum() > 0 else _TEXT),
        ("Est. volume",     f"{vol_cm3:.2f} cm³",     _TEXT),
        ("Est. depth",      f"{depth_cm:.1f} cm",     _TEXT),
        ("",                "",                        _TEXT),
        ("BIOHEAT",         "─────────────",          _ACCENT),
        ("k",               f"{K_TISSUE} W/(m·K)",    _TEXT),
        ("ωb",              f"{OMEGA_B} s⁻¹",         _TEXT),
        ("Qm",              f"{Q_METABOLIC} W/m³",    _TEXT),
    ]
    for row_i, (key, val, color) in enumerate(lines):
        ax_e3.text(0.02, 0.97 - row_i * 0.048, f"{key}", transform=ax_e3.transAxes,
                   color=_MUTED, fontsize=7.5, family='monospace', va='top')
        ax_e3.text(0.55, 0.97 - row_i * 0.048, f"{val}", transform=ax_e3.transAxes,
                   color=color, fontsize=7.5, family='monospace', va='top', fontweight='bold')
    ax_e3.set_title("E3  Analysis Summary", color=_TEXT, fontsize=8,
                    pad=4, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    out_path = os.path.join(output_dir, f'patient_{patient_id}_thermal_3d_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_DARK_BG)
    plt.close()
    print(f"    ✓ Dashboard saved → {out_path}")
    return out_path


def visualise_pyvista(voxel_grid: np.ndarray,
                      T_vol: np.ndarray,
                      tumor_mask: np.ndarray,
                      patient_id: int = PATIENT_ID,
                      output_dir: str = '/mnt/user-data/outputs'):
    """
    PyVista volumetric rendering with inferno colourmap and clipping planes.
    Saves off-screen PNG screenshots; launches interactive window if display found.
    """
    try:
        import pyvista as pv
        pv.global_theme.background   = _DARK_BG
        pv.global_theme.font.color   = _TEXT
        pv.global_theme.anti_aliasing = 'msaa'

        Nx, Ny, Nz = GRID_SIZE
        cm_per_vox = VOXEL_SIZE_M * 100          # 5 mm → 0.5 cm

        # ── Build ImageData (cell-centred) ─────────────────────────────
        grid = pv.ImageData()
        grid.dimensions = (Nx + 1, Ny + 1, Nz + 1)
        grid.spacing    = (cm_per_vox,) * 3
        grid.origin     = (0.0, 0.0, 0.0)

        T_display  = np.where(voxel_grid,
                               np.nan_to_num(T_vol, nan=CORE_TEMP),
                               0.0).ravel(order='F')
        tiss_flat  = voxel_grid.astype(float).ravel(order='F')
        tumor_flat = tumor_mask.astype(float).ravel(order='F')

        grid.cell_data['Temperature_C'] = T_display
        grid.cell_data['Tissue']        = tiss_flat
        grid.cell_data['TumorRisk']     = tumor_flat

        tissue = grid.threshold(0.5, scalars='Tissue')

        # ── Off-screen plotter ─────────────────────────────────────────
        p = pv.Plotter(off_screen=True, window_size=(1600, 900),
                       shape=(1, 2))

        # LEFT — volumetric inferno temperature rendering
        p.subplot(0, 0)
        p.add_text(f"Patient {patient_id}  |  Thermal Volume (Inferno)",
                   font_size=9, position='upper_left', color=_TEXT)

        T_masked = np.where(voxel_grid,
                             np.nan_to_num(T_vol, nan=CORE_TEMP),
                             CORE_TEMP - 5).ravel(order='F')
        grid.cell_data['Temperature_C'] = T_masked

        vol_kwargs = dict(scalars='Temperature_C', cmap='inferno',
                          clim=[36.0, 40.5],
                          opacity=[0, 0, 0.05, 0.2, 0.5, 0.85, 1.0],
                          shade=True, mapper='smart')
        p.add_volume(grid, **vol_kwargs)
        p.add_scalar_bar('Temperature (°C)', fmt='%.1f',
                         color=_TEXT, n_labels=5)
        p.add_axes(color=_TEXT)
        p.camera_position = 'iso'

        # RIGHT — surface + tumour overlay
        p.subplot(0, 1)
        p.add_text("Surface Mesh  +  Tumour Risk (Red)",
                   font_size=9, position='upper_left', color=_TEXT)
        surf = tissue.extract_surface().smooth(n_iter=20)
        p.add_mesh(surf, scalars='Temperature_C', cmap='coolwarm',
                   clim=[35.5, 40.5], opacity=0.35, smooth_shading=True)

        if tumor_mask.sum() > 0:
            tum = grid.threshold(0.5, scalars='TumorRisk')
            p.add_mesh(tum, color='#ff2200', opacity=0.75,
                       label='Tumour hot-spot')
            p.add_legend(bcolor=[0.1, 0.1, 0.1], face='rectangle',
                         size=(0.22, 0.12))

        p.add_scalar_bar('Temperature (°C)', fmt='%.1f',
                         color=_TEXT, n_labels=5)
        p.camera_position = 'iso'
        p.link_views()

        out_pv = os.path.join(output_dir,
                              f'patient_{patient_id}_pyvista_3d.png')
        p.screenshot(out_pv, transparent_background=False)
        p.close()
        print(f"    ✓ PyVista render saved → {out_pv}")
        return out_pv

    except Exception as e:
        print(f"    ✗ PyVista render skipped: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(patient_id: int = PATIENT_ID):
    """Execute the full 3D Thermal Patient Reconstruction pipeline."""
    print(f"\n{'═'*70}")
    print(f"  3D THERMAL PATIENT RECONSTRUCTION — ONCOLOGY PIPELINE")
    print(f"  Patient {patient_id}  │  Grid {GRID_SIZE}  │  dx={VOXEL_SIZE_M*1000:.0f} mm")
    print(f"{'═'*70}")

    os.makedirs('/mnt/user-data/outputs', exist_ok=True)

    # ── 1. Acquire data ────────────────────────────────────────────────
    patient_views  = load_patient_views(patient_id)

    # ── 2. Segment ─────────────────────────────────────────────────────
    segmented      = segment_all_views(patient_views)

    # ── 3. Visual Hull ─────────────────────────────────────────────────
    voxel_grid, coord_grid = space_carving(segmented, GRID_SIZE)

    # ── 4. Thermal fusion ──────────────────────────────────────────────
    T_fused, surface = fuse_thermal_data(
        voxel_grid, coord_grid, segmented, GRID_SIZE)

    # ── 5. Bioheat solver ──────────────────────────────────────────────
    T_solved, tumor_mask = solve_pennes_bioheat(
        T_fused, voxel_grid, surface, n_iter=300, tol=1e-6)

    # ── 6. Visualise ───────────────────────────────────────────────────
    out_mpl = visualise_all(
        patient_views, segmented, voxel_grid,
        T_solved, tumor_mask, patient_id)

    out_pv  = visualise_pyvista(
        voxel_grid, T_solved, tumor_mask, patient_id)

    # ── 7. Save arrays for downstream analysis ─────────────────────────
    base = f'/mnt/user-data/outputs/patient_{patient_id}'
    np.save(f'{base}_voxel_grid.npy',  voxel_grid)
    np.save(f'{base}_T_solved.npy',    T_solved)
    np.save(f'{base}_tumor_mask.npy',  tumor_mask)
    print(f"\n  NumPy arrays saved → /mnt/user-data/outputs/")

    print(f"\n{'═'*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'═'*70}\n")

    return {
        'voxel_grid' : voxel_grid,
        'T_solved'   : T_solved,
        'tumor_mask' : tumor_mask,
        'patient_id' : patient_id,
        'outputs'    : [out_mpl, out_pv]
    }


if __name__ == "__main__":
    run_pipeline(PATIENT_ID)