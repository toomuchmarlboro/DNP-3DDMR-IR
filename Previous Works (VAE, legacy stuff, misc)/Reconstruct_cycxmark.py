"""
reconstruct_3d_breast.py
========================
3D breast volume reconstruction from 5 masked radiometric thermography views.

Compatible with output of watershed_background_removal.py:
  {stem}__mask.png              -- binary silhouette (uint8, 0/255)
  {stem}__mask.npy              -- binary silhouette (uint8, 0/1)
  {stem}__masked_radiometric.npy-- raw thermal float array (background=0)

Author : generated pipeline
Usage  :
    python reconstruct_3d_breast.py \
        --input-root  data/organized_by_patient_watershed \
        --output-root data/reconstruction_output \
        --voxel-res   64 \
        --projection  orthographic \
        --carving-mode soft \
        --min-views   3 \
        --method      baseline \
        --save-debug  true

    # Dry-run (first patient only):
    python reconstruct_3d_breast.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import center_of_mass as scipy_center_of_mass

# ---------------------------------------------------------------------------
# Optional heavy imports – graceful degradation
# ---------------------------------------------------------------------------
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    from skimage.measure import marching_cubes
    from skimage.morphology import binary_closing, binary_opening, remove_small_objects
    from skimage.morphology import ball, disk
    from skimage.measure import label as sk_label
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize as MplNorm
    from matplotlib.cm import ScalarMappable
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

VIEW_NAMES: List[str] = [
    "Anterior (Front)",
    "Left Oblique (45°)",
    "Right Oblique (45°)",
    "Left Lateral (90°)",
    "Right Lateral (90°)",
]

# Yaw angles in degrees for each canonical view (patient-centred, right-hand)
VIEW_YAWS: Dict[str, float] = {
    "Anterior (Front)":    0.0,
    "Left Oblique (45°)": -45.0,
    "Right Oblique (45°)": 45.0,
    "Left Lateral (90°)": -90.0,
    "Right Lateral (90°)": 90.0,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("reconstruct3d")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

LOG = setup_logger()


def resolve_relative_path(path_value: Path, script_dir: Path) -> Path:
    """Resolve a relative path against the current working directory first, then the script directory."""
    if path_value.is_absolute():
        return path_value

    cwd_candidate = (Path.cwd() / path_value).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (script_dir / path_value).resolve()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ViewData:
    """Single-view input bundle."""
    view_name: str
    yaw_deg: float
    mask: np.ndarray          # (H, W) uint8 0/1
    radiometric: Optional[np.ndarray]  # (H, W) float, None if unavailable
    mask_path: str
    radiometric_path: str
    confidence: float = 1.0   # 0-1, reduced for fallback/missing views


@dataclass
class PatientRecord:
    patient_id: str
    available_views: List[str] = field(default_factory=list)
    mask_paths: Dict[str, str] = field(default_factory=dict)
    radiometric_paths: Dict[str, str] = field(default_factory=dict)
    image_shapes: Dict[str, list] = field(default_factory=dict)
    voxel_grid_shape: list = field(default_factory=list)
    reconstruction_status: str = "pending"
    metrics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_image_any(path: Path) -> np.ndarray:
    """Load image from TIFF, NPY, PNG in native precision."""
    suf = path.suffix.lower()
    if suf in (".tif", ".tiff"):
        if HAS_TIFFFILE:
            return tifffile.imread(str(path)).astype(np.float32)
        if HAS_PIL:
            return np.array(PILImage.open(path)).astype(np.float32)
    if suf == ".npy":
        return np.load(str(path)).astype(np.float32)
    if suf == ".png":
        if HAS_CV2:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img.astype(np.float32)
        if HAS_PIL:
            return np.array(PILImage.open(path)).astype(np.float32)
    raise IOError(f"Cannot load: {path}")


def load_mask(path: Path, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load binary mask as uint8 0/1, optionally resize to target_shape (H,W)."""
    arr = load_image_any(path)
    if arr.ndim == 3:
        arr = arr[..., 0]          # take first channel
    mask = (arr > 0).astype(np.uint8)
    if target_shape is not None and mask.shape != target_shape:
        LOG.warning("Mask shape %s != radiometric shape %s → resizing mask",
                    mask.shape, target_shape)
        if HAS_CV2:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        elif HAS_PIL:
            pil = PILImage.fromarray(mask)
            pil = pil.resize((target_shape[1], target_shape[0]),
                             PILImage.NEAREST)
            mask = np.array(pil)
    return mask.astype(np.uint8)


def clean_mask(mask: np.ndarray,
               min_size: int = 200,
               fill_holes: bool = True,
               keep_largest: bool = True) -> np.ndarray:
    """Morphological cleanup of binary mask."""
    if not HAS_SKIMAGE:
        return mask
    m = mask.astype(bool)
    m = binary_opening(m, disk(3))
    if fill_holes:
        from scipy.ndimage import binary_fill_holes
        m = binary_fill_holes(m)
    m = remove_small_objects(m, min_size=min_size)
    if keep_largest:
        labeled = sk_label(m)
        if labeled.max() > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            m = (labeled == sizes.argmax())
    return m.astype(np.uint8)


# ---------------------------------------------------------------------------
# Dynamic Principal Point – Hybrid Center Detection
# ---------------------------------------------------------------------------

def get_mask_center(
    mask: np.ndarray,
    radiometric: Optional[np.ndarray],
    fiducial_cold_percentile: float = 2.0,
    fiducial_search_cols: float = 0.20,
    fiducial_search_rows_start: float = 0.65,
    fiducial_min_pixels: int = 10,
    fiducial_min_contrast: float = 0.90,
) -> Tuple[float, float]:
    """
    Compute the dynamic principal point (cx, cy) for a single view.

    This solves the "hourglass artefact" caused by assuming a fixed image
    centre (W/2, H/2) when the breast silhouette is off-centre due to
    hand-held clinical camera positioning.

    WHAT IS THE PRINCIPAL POINT (cx, cy)?
    ======================================
    cx, cy is the image coordinate onto which the 3D origin (0,0,0) projects.
    Geometrically this is the pixel position of the breast centroid in the
    image.  Setting it correctly ensures that the voxel grid is aligned with
    the breast in every view, preventing the hourglass collapse.

    WHY THE BREAST CoM — NOT THE FIDUCIAL TAPE?
    ============================================
    Although the black tape at the epigastrium (ulu hati) is a visible
    anatomical landmark, it is anatomically BELOW the breast centre
    (typically 80-120 px lower in a 240-row image).  Using the tape position
    as cy shifts the entire 3D reconstruction downward by that amount, making
    the hourglass WORSE, not better.

    The fiducial tape IS useful for cross-view REGISTRATION (finding a fixed
    3D point shared across all camera poses), but implementing that requires
    knowing the tape's 3D coordinates relative to the breast model origin and
    updating the coordinate system accordingly — a separate step not handled
    here.

    For fixing off-centre principal point, the MASK CENTER-OF-MASS (CoM) is
    the correct approach: it tells us exactly where the breast silhouette
    centre appears in each view's image plane, which is the correct image
    position of the projection of the anatomical breast centre.

    The fiducial is still detected and LOGGED as a diagnostic (it can help
    identify if the tape is visible in a given view), but it is NOT used as
    the projection principal point.

    PRIORITY CHAIN
    ==============
    Priority 1 — Mask Center-of-Mass (PRIMARY, geometrically correct):
        scipy.ndimage.center_of_mass of the binary breast silhouette mask.
        Returns the image (row, col) of the weighted centroid of all
        foreground pixels.  This is exactly where the breast centre projects.

    Priority 2 — Geometric image centre (FALLBACK for empty mask only):
        (W/2, H/2).  Only reached if the mask has no foreground pixels at all.

    FIDUCIAL TAPE DIAGNOSTIC (logged but not used as principal point):
        Detects the cold tape cluster in the lower-centre of the radiometric
        image and logs its position as a DEBUG message.  Can be used
        separately for cross-view alignment if desired.

    Parameters
    ----------
    mask                     : (H, W) uint8 binary mask (0/1)
    radiometric              : (H, W) float32 temperature array or None
                               (used only for fiducial diagnostic logging)
    fiducial_cold_percentile : percentile threshold for tape detection
    fiducial_search_cols     : half-width of tape search zone as fraction of W
    fiducial_search_rows_start : upper boundary of tape search as fraction of H
    fiducial_min_pixels      : minimum cold cluster size to log as tape
    fiducial_min_contrast    : tape_threshold / region_median must be <= this
                               (guards against false-positives in uniform images)

    Returns
    -------
    (cx, cy) : float tuple — principal point in pixel coordinates
               cx = column index (horizontal), cy = row index (vertical)
    """
    H, W = mask.shape

    # ----------------------------------------------------------------
    # Fiducial tape diagnostic (logging only — NOT used as cx/cy)
    # ----------------------------------------------------------------
    if radiometric is not None:
        r0 = int(fiducial_search_rows_start * H)
        c0 = int((0.5 - fiducial_search_cols) * W)
        c1 = int((0.5 + fiducial_search_cols) * W)
        r0 = max(0, min(r0, H - 1))
        c0 = max(0, min(c0, W - 1))
        c1 = max(c0 + 1, min(c1, W))

        search_region = radiometric[r0:H, c0:c1].copy()
        nonzero_vals  = search_region[search_region > 0.0]

        if len(nonzero_vals) >= fiducial_min_pixels:
            threshold     = float(np.percentile(nonzero_vals,
                                                fiducial_cold_percentile))
            region_median = float(np.median(nonzero_vals))
            contrast_ok   = (region_median > 0.0 and
                             threshold <= fiducial_min_contrast * region_median)

            if contrast_ok:
                cold_binary = (search_region > 0.0) & (search_region <= threshold)
                n_cold = int(cold_binary.sum())
                if n_cold >= fiducial_min_pixels:
                    rows_idx, cols_idx = np.where(cold_binary)
                    cy_fid = float(np.mean(rows_idx)) + r0
                    cx_fid = float(np.mean(cols_idx)) + c0
                    LOG.debug(
                        "    [fiducial-diag] tape detected at "
                        "cx=%.1f cy=%.1f (cluster=%d px, contrast=%.3f) "
                        "— logged only, NOT used as principal point",
                        cx_fid, cy_fid, n_cold,
                        threshold / max(region_median, 1e-9),
                    )
                else:
                    LOG.debug("    [fiducial-diag] cold cluster too small (%d px)", n_cold)
            else:
                LOG.debug("    [fiducial-diag] contrast guard failed (uniform image)")

    # ----------------------------------------------------------------
    # Priority 1: Mask Center-of-Mass  (correct principal point)
    # ----------------------------------------------------------------
    if mask.sum() > 0:
        com = scipy_center_of_mass(mask.astype(np.float32))
        # center_of_mass returns (row, col) — note order
        cy_com, cx_com = float(com[0]), float(com[1])
        LOG.debug(
            "    [principal-point] mask CoM: cx=%.1f  cy=%.1f  "
            "(foreground px=%d)",
            cx_com, cy_com, int(mask.sum()),
        )
        return cx_com, cy_com

    # ----------------------------------------------------------------
    # Priority 2: Geometric image centre (empty mask only)
    # ----------------------------------------------------------------
    LOG.warning(
        "    [principal-point] mask is empty — using geometric centre "
        "(%.1f, %.1f)",
        W / 2.0, H / 2.0,
    )
    return float(W / 2.0), float(H / 2.0)


# ---------------------------------------------------------------------------
# Patient discovery
# ---------------------------------------------------------------------------

def discover_patients(input_root: Path) -> Dict[str, Path]:
    """
    Walk organized_by_patient_watershed and return {patient_id: patient_folder}.
    Handles both flat and class-subfolder layouts.
    """
    patients: Dict[str, Path] = {}
    for p in sorted(input_root.iterdir()):
        if p.is_dir() and p.name.startswith("Patient_"):
            patients[p.name] = p
    return patients


def find_view_files(patient_dir: Path) -> Dict[str, Dict[str, Optional[Path]]]:
    """
    Locate mask + radiometric files for each canonical view within a patient folder.
    Search all subdirectories (class subfolders like benign/malignant).

    Returns:
        {view_name: {"mask": Path|None, "radiometric": Path|None}}
    """
    result: Dict[str, Dict[str, Optional[Path]]] = {
        v: {"mask": None, "radiometric": None} for v in VIEW_NAMES
    }

    # Collect all files under patient_dir recursively
    all_files = list(patient_dir.rglob("*"))

    for view_name in VIEW_NAMES:
        # Stem of the original image == view_name (from watershed script)
        stem = view_name

        # Mask candidates
        mask_png = next((f for f in all_files
                         if f.name == f"{stem}__mask.png"), None)
        mask_npy = next((f for f in all_files
                         if f.name == f"{stem}__mask.npy"), None)
        # Radiometric masked array
        rad_npy  = next((f for f in all_files
                         if f.name == f"{stem}__masked_radiometric.npy"), None)
        # Original TIFF (fallback)
        orig_tif = next((f for f in all_files
                         if f.stem == stem and f.suffix.lower() in (".tif", ".tiff")), None)

        result[view_name]["mask"]        = mask_npy or mask_png
        result[view_name]["radiometric"] = rad_npy or orig_tif

    return result


# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------

def yaw_rotation_matrix(yaw_deg: float) -> np.ndarray:
    """3×3 rotation matrix for yaw around the Y (vertical) axis."""
    theta = np.deg2rad(yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)


def orthographic_project(
    points_3d: np.ndarray,
    yaw_deg: float,
    image_h: int,
    image_w: int,
    scale: float = 1.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> np.ndarray:
    """
    Orthographic projection: rotate by yaw, drop depth axis.

    Uses a Dynamic Principal Point (cx, cy) so that off-centre breast
    positions from hand-held clinical capture are handled correctly.
    When cx/cy are None the classical fixed-centre formula is used as
    a safe default (backward-compatible).

    Mapping with dynamic principal point
    -------------------------------------
        pixels_per_unit = (image_w - 1) / (2.0 * scale)
        col = x_cam * pixels_per_unit + cx
        row = -y_cam * pixels_per_unit + cy    # Y flipped (image Y goes down)

    Returns pixel coordinates (N, 2) as float32: [row, col].
    """
    R = yaw_rotation_matrix(yaw_deg)
    rotated = points_3d @ R.T            # (N, 3)
    x_cam = rotated[:, 0]                # horizontal -> col
    y_cam = rotated[:, 1]                # vertical   -> row (flipped)

    if cx is None or cy is None:
        # Legacy fixed-centre formula (fallback / backward-compat)
        col = (x_cam / scale + 1.0) * 0.5 * (image_w - 1)
        row = (1.0 - (y_cam / scale + 1.0) * 0.5) * (image_h - 1)
    else:
        # Dynamic principal point formula.
        # IMPORTANT: horizontal and vertical pixel densities differ on
        # non-square images — use image_w for cols and image_h for rows.
        ppu_w = (image_w - 1) / (2.0 * scale)   # pixels per world unit, horizontal
        ppu_h = (image_h - 1) / (2.0 * scale)   # pixels per world unit, vertical
        col = x_cam * ppu_w + cx
        row = -y_cam * ppu_h + cy               # negative: image Y goes down

    return np.stack([row, col], axis=1).astype(np.float32)


def perspective_project(
    points_3d: np.ndarray,
    yaw_deg: float,
    image_h: int,
    image_w: int,
    focal_scale: float = 2.0,
    scale: float = 1.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> np.ndarray:
    """
    Perspective projection with configurable focal length and dynamic
    principal point.

    Uses a Dynamic Principal Point (cx, cy) to handle off-centre breast
    positions from hand-held clinical cameras.
    When cx/cy are None the classical fixed-centre formula is used as
    a safe default (backward-compatible).

    Mapping with dynamic principal point (after perspective divide)
    ---------------------------------------------------------------
        pixels_per_unit = (image_w - 1) / (2.0 * scale)
        col = x_proj * pixels_per_unit + cx
        row = -y_proj * pixels_per_unit + cy   # Y flipped

    Returns pixel coordinates (N, 2) as float32: [row, col].
    """
    R = yaw_rotation_matrix(yaw_deg)
    rotated = points_3d @ R.T
    x_cam = rotated[:, 0]
    y_cam = rotated[:, 1]
    z_cam = rotated[:, 2]

    f = focal_scale * scale
    cam_z = f + scale

    denom = cam_z - z_cam
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)

    # Perspective-divided (projected) camera-plane coordinates
    x_proj = f * x_cam / denom
    y_proj = f * y_cam / denom

    if cx is None or cy is None:
        # Legacy fixed-centre formula (fallback / backward-compat)
        col = (x_proj / scale + 1.0) * 0.5 * (image_w - 1)
        row = (1.0 - (y_proj / scale + 1.0) * 0.5) * (image_h - 1)
    else:
        # Dynamic principal point formula.
        # IMPORTANT: horizontal and vertical pixel densities differ on
        # non-square images — use image_w for cols and image_h for rows.
        ppu_w = (image_w - 1) / (2.0 * scale)   # pixels per world unit, horizontal
        ppu_h = (image_h - 1) / (2.0 * scale)   # pixels per world unit, vertical
        col = x_proj * ppu_w + cx
        row = -y_proj * ppu_h + cy              # negative: image Y goes down

    return np.stack([row, col], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Voxel grid helpers
# ---------------------------------------------------------------------------

def build_voxel_coords(res: int, scale: float = 1.0) -> np.ndarray:
    """
    Build (res^3, 3) array of 3D world-space coords for a cubic voxel grid.
    Coordinates span [-scale, +scale] in each axis.
    """
    lin = np.linspace(-scale, scale, res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')  # (res, res, res)
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (N, 3)
    return coords


def sample_mask_at_pixels(
    mask: np.ndarray,          # (H, W) uint8 0/1
    pixels: np.ndarray         # (N, 2) float [row, col]
) -> np.ndarray:
    """
    Nearest-neighbour lookup of mask values at (possibly fractional) pixel coords.
    Returns (N,) uint8.
    """
    H, W = mask.shape
    rows = np.clip(np.round(pixels[:, 0]).astype(np.int32), 0, H - 1)
    cols = np.clip(np.round(pixels[:, 1]).astype(np.int32), 0, W - 1)
    return mask[rows, cols]


def sample_radiometric_at_pixels(
    radiometric: np.ndarray,   # (H, W) float
    pixels: np.ndarray,        # (N, 2) float [row, col]
    valid_only: bool = True
) -> np.ndarray:
    """
    Bilinear-ish (floor) sampling of radiometric at pixel positions.
    Returns (N,) float32; zeros where out-of-bounds or background.
    """
    H, W = radiometric.shape
    rows = np.clip(np.floor(pixels[:, 0]).astype(np.int32), 0, H - 1)
    cols = np.clip(np.floor(pixels[:, 1]).astype(np.int32), 0, W - 1)
    vals = radiometric[rows, cols].astype(np.float32)
    return vals


# ---------------------------------------------------------------------------
# Core reconstruction: Visual Hull Voxel Carving + Thermal Fusion
# ---------------------------------------------------------------------------

def voxel_carving(
    views: List[ViewData],
    voxel_res: int = 128,
    projection: str = "orthographic",
    carving_mode: str = "soft",
    min_views: int = 3,
    scale: float = 1.0,
    focal_scale: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Silhouette-based visual hull voxel carving with thermal fusion.

    Parameters
    ----------
    views        : list of ViewData (loaded masks + radiometric)
    voxel_res    : cubic grid side length
    projection   : "orthographic" | "perspective"
    carving_mode : "strict" (all views) | "soft" (min_views)
    min_views    : minimum views for soft mode
    scale        : half-extent of voxel grid in world units
    focal_scale  : perspective focal multiplier

    Returns
    -------
    occupancy     : (R, R, R) bool array
    temp_volume   : (R, R, R) float32, fused temperature
    vote_count    : (R, R, R) uint8, how many views voted this voxel in
    """
    R = voxel_res
    N_vox = R ** 3
    LOG.info("Building %d³ voxel grid → %d voxels", R, N_vox)

    coords = build_voxel_coords(R, scale=scale)  # (N_vox, 3)

    # Accumulators
    vote_in  = np.zeros(N_vox, dtype=np.int16)    # number of views voting "inside"
    vote_tot = np.zeros(N_vox, dtype=np.int16)    # number of views used
    temp_sum  = np.zeros(N_vox, dtype=np.float64)
    temp_wsum = np.zeros(N_vox, dtype=np.float64)
    temp_stack: List[np.ndarray] = []             # for median fusion

    for vd in views:
        H, W = vd.mask.shape

        # --- Dynamic Principal Point (Hybrid Alignment) ---
        cx, cy = get_mask_center(vd.mask, vd.radiometric)
        LOG.debug(
            "  [carving] view='%s'  principal_point=(cx=%.1f, cy=%.1f)  "
            "image=(%d x %d)",
            vd.view_name, cx, cy, W, H
        )

        if projection == "perspective":
            pix = perspective_project(coords, vd.yaw_deg, H, W,
                                      focal_scale=focal_scale, scale=scale,
                                      cx=cx, cy=cy)
        else:
            pix = orthographic_project(coords, vd.yaw_deg, H, W,
                                       scale=scale, cx=cx, cy=cy)

        inside = sample_mask_at_pixels(vd.mask, pix)  # (N_vox,) 0/1
        vote_in  += inside.astype(np.int16)
        vote_tot += 1

        if vd.radiometric is not None:
            temps = sample_radiometric_at_pixels(vd.radiometric, pix)
            temp_sum  += (temps * inside * vd.confidence).astype(np.float64)
            temp_wsum += (inside * vd.confidence).astype(np.float64)
            temp_stack.append(np.where(inside.astype(bool), temps, np.nan))

    # --- Carving decision ---
    if carving_mode == "strict":
        occupancy_flat = (vote_in == vote_tot)
    else:  # soft
        occupancy_flat = (vote_in >= min_views)

    # --- Temperature fusion: weighted mean + median ---
    temp_wmean = np.where(temp_wsum > 0, temp_sum / np.maximum(temp_wsum, 1e-9),
                          0.0).astype(np.float32)

    if temp_stack:
        stack = np.stack(temp_stack, axis=0)  # (V, N_vox)
        with np.errstate(all='ignore'):
            temp_median = np.nanmedian(stack, axis=0).astype(np.float32)
        temp_median = np.nan_to_num(temp_median, nan=0.0)
    else:
        temp_median = temp_wmean.copy()

    # Fuse: average of weighted mean and median
    temp_fused = np.where(occupancy_flat, 0.5 * (temp_wmean + temp_median), 0.0)

    occupancy  = occupancy_flat.reshape(R, R, R)
    temp_volume = temp_fused.reshape(R, R, R).astype(np.float32)
    vote_count = vote_in.reshape(R, R, R).astype(np.uint8)

    return occupancy.astype(bool), temp_volume, vote_count


# ---------------------------------------------------------------------------
# Probabilistic occupancy fusion (Method C)
# ---------------------------------------------------------------------------

def probabilistic_occupancy(
    views: List[ViewData],
    voxel_res: int,
    projection: str = "orthographic",
    scale: float = 1.0,
    focal_scale: float = 2.0,
    prior: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bayesian occupancy fusion: each view converts mask silhouette into
    a ray-based likelihood and updates a shared probability volume.

    Returns
    -------
    prob_volume  : (R, R, R) float32 in [0, 1] – occupancy probability
    uncertainty  : (R, R, R) float32 – binary entropy H(p)
    """
    R = voxel_res
    N_vox = R ** 3
    coords = build_voxel_coords(R, scale=scale)

    log_odds = np.full(N_vox, np.log(prior / (1 - prior + 1e-9)),
                       dtype=np.float64)

    p_inside = 0.85   # P(in mask | occupied)
    p_outside = 0.10  # P(in mask | empty)  — some silhouette ambiguity

    for vd in views:
        H, W = vd.mask.shape

        # --- Dynamic Principal Point (Hybrid Alignment) ---
        cx, cy = get_mask_center(vd.mask, vd.radiometric)
        LOG.debug(
            "  [prob] view='%s'  principal_point=(cx=%.1f, cy=%.1f)  "
            "image=(%d x %d)",
            vd.view_name, cx, cy, W, H
        )

        if projection == "perspective":
            pix = perspective_project(coords, vd.yaw_deg, H, W,
                                      focal_scale=focal_scale, scale=scale,
                                      cx=cx, cy=cy)
        else:
            pix = orthographic_project(coords, vd.yaw_deg, H, W,
                                       scale=scale, cx=cx, cy=cy)

        inside = sample_mask_at_pixels(vd.mask, pix).astype(bool)

        # Log-likelihood update
        ll_occ_in   = np.log(p_inside    + 1e-9)
        ll_emp_in   = np.log(p_outside   + 1e-9)
        ll_occ_out  = np.log(1 - p_inside  + 1e-9)
        ll_emp_out  = np.log(1 - p_outside + 1e-9)

        ll_ratio = np.where(inside,
                            ll_occ_in  - ll_emp_in,
                            ll_occ_out - ll_emp_out)
        log_odds += ll_ratio * vd.confidence

    probs = 1.0 / (1.0 + np.exp(-log_odds))
    probs = probs.reshape(R, R, R).astype(np.float32)

    eps = 1e-9
    entropy = -(probs * np.log(probs + eps) +
                (1 - probs) * np.log(1 - probs + eps))
    entropy = entropy.astype(np.float32)

    return probs, entropy


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection-over-Union for binary masks."""
    p = pred_mask.astype(bool).ravel()
    g = gt_mask.astype(bool).ravel()
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union > 0 else 0.0


def reproject_occupancy(
    occupancy: np.ndarray,   # (R, R, R) bool
    vd: ViewData,
    voxel_res: int,
    projection: str = "orthographic",
    scale: float = 1.0,
    focal_scale: float = 2.0,
) -> np.ndarray:
    """
    Project occupancy volume back onto view plane -> binary silhouette image.

    Uses the same Dynamic Principal Point (get_mask_center) as the carving
    functions so that the reprojection IoU is computed consistently.

    Returns (H, W) uint8 reprojected mask.
    """
    R = voxel_res
    H, W = vd.mask.shape
    coords = build_voxel_coords(R, scale=scale)
    occ_flat = occupancy.ravel().astype(bool)

    # Reuse the same principal point as was used during carving
    cx, cy = get_mask_center(vd.mask, vd.radiometric)

    if projection == "perspective":
        pix = perspective_project(coords, vd.yaw_deg, H, W,
                                  focal_scale=focal_scale, scale=scale,
                                  cx=cx, cy=cy)
    else:
        pix = orthographic_project(coords, vd.yaw_deg, H, W,
                                   scale=scale, cx=cx, cy=cy)

    reproj = np.zeros((H, W), dtype=np.uint8)
    rows = np.clip(np.round(pix[:, 0]).astype(np.int32), 0, H - 1)
    cols = np.clip(np.round(pix[:, 1]).astype(np.int32), 0, W - 1)
    reproj[rows[occ_flat], cols[occ_flat]] = 1

    # morphological closing to fill projection gaps
    if HAS_SKIMAGE:
        reproj = binary_closing(reproj.astype(bool), disk(3)).astype(np.uint8)

    return reproj


def compute_quality_metrics(
    occupancy: np.ndarray,
    temp_volume: np.ndarray,
    views: List[ViewData],
    voxel_res: int,
    projection: str = "orthographic",
    scale: float = 1.0,
    focal_scale: float = 2.0,
) -> Dict:
    """Compute reprojection IoU, thermal variance, and volume plausibility."""
    metrics: Dict = {}

    # --- Reprojection IoU per view ---
    ious = []
    for vd in views:
        reproj = reproject_occupancy(occupancy, vd, voxel_res, projection,
                                     scale, focal_scale)
        iou = compute_iou(reproj, vd.mask)
        ious.append(iou)
        metrics[f"iou_{vd.view_name}"] = round(iou, 4)

    metrics["mean_iou"] = round(float(np.mean(ious)), 4) if ious else 0.0

    # --- Thermal variance ---
    active = temp_volume[occupancy]
    if len(active) > 0:
        metrics["temp_mean"]   = round(float(np.mean(active)), 4)
        metrics["temp_std"]    = round(float(np.std(active)), 4)
        metrics["temp_median"] = round(float(np.median(active)), 4)
        median_abs_dev = np.median(np.abs(active - np.median(active)))
        metrics["temp_MAD"]    = round(float(median_abs_dev), 4)
    else:
        metrics.update({"temp_mean": 0, "temp_std": 0,
                         "temp_median": 0, "temp_MAD": 0})

    # --- Volume plausibility ---
    occ_count = int(occupancy.sum())
    total     = voxel_res ** 3
    fill_frac = occ_count / total

    metrics["occupied_voxels"] = occ_count
    metrics["fill_fraction"]   = round(fill_frac, 6)
    metrics["empty_volume"]    = occ_count == 0

    if occ_count > 0 and HAS_SKIMAGE:
        # Disconnected anatomy check
        labeled = sk_label(occupancy)
        n_comp = labeled.max()
        metrics["n_connected_components"] = int(n_comp)
        metrics["disconnected_anatomy"]   = bool(n_comp > 3)

        # Aspect ratio
        rz = np.any(occupancy, axis=(0, 1))
        ry = np.any(occupancy, axis=(0, 2))
        rx = np.any(occupancy, axis=(1, 2))
        dims = [rz.sum(), ry.sum(), rx.sum()]
        dims = [max(d, 1) for d in dims]
        ar = max(dims) / min(dims)
        metrics["aspect_ratio"]         = round(float(ar), 3)
        metrics["unrealistic_aspect_ratio"] = bool(ar > 5.0)
    else:
        metrics["n_connected_components"] = 0
        metrics["disconnected_anatomy"]   = False
        metrics["aspect_ratio"]           = 0.0
        metrics["unrealistic_aspect_ratio"] = False

    return metrics


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_npy(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)


def save_nifti(arr: np.ndarray, path: Path) -> None:
    if not HAS_NIBABEL:
        LOG.warning("nibabel not installed – skipping NIfTI export")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4))
    nib.save(nii, str(path))
    LOG.info("Saved NIfTI: %s", path)


def save_mesh_ply(occupancy: np.ndarray,
                  temp_volume: np.ndarray,
                  path: Path,
                  level: float = 0.5) -> None:
    if not HAS_SKIMAGE:
        LOG.warning("scikit-image not installed – skipping PLY export")
        return
    from skimage.measure import marching_cubes
    occ_float = occupancy.astype(np.float32)
    try:
        verts, faces, normals, _ = marching_cubes(occ_float, level=level)
    except Exception as e:
        LOG.warning("Marching cubes failed: %s", e)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    # Colour by temperature
    vi = np.round(verts).astype(int)
    vi[:, 0] = np.clip(vi[:, 0], 0, temp_volume.shape[0] - 1)
    vi[:, 1] = np.clip(vi[:, 1], 0, temp_volume.shape[1] - 1)
    vi[:, 2] = np.clip(vi[:, 2], 0, temp_volume.shape[2] - 1)
    temps_v = temp_volume[vi[:, 0], vi[:, 1], vi[:, 2]]

    # Write ASCII PLY
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float temperature\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")
        for (x, y, z), t in zip(verts, temps_v):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {t:.4f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    LOG.info("Saved PLY mesh: %s", path)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_qa_panel(
    views: List[ViewData],
    occupancy: np.ndarray,
    temp_volume: np.ndarray,
    patient_id: str,
    out_dir: Path,
    voxel_res: int,
    projection: str = "orthographic",
    scale: float = 1.0,
) -> None:
    """Save a 2D QA panel: radiometric | mask | reprojection | residual."""
    if not HAS_MPL:
        LOG.warning("matplotlib not available – skipping QA panel")
        return

    n_views = len(views)
    fig, axes = plt.subplots(n_views, 4,
                              figsize=(16, 4 * n_views),
                              squeeze=False)

    for row, vd in enumerate(views):
        H, W = vd.mask.shape

        # Col 0: radiometric
        if vd.radiometric is not None:
            rad_vis = vd.radiometric.copy().astype(np.float32)
            rad_vis[rad_vis == 0] = np.nan
            axes[row, 0].imshow(rad_vis, cmap="inferno", interpolation="nearest")
            axes[row, 0].set_title(f"{vd.view_name}\nRadiometric", fontsize=7)
        else:
            axes[row, 0].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[row, 0].set_title(f"{vd.view_name}\nRadiometric (N/A)", fontsize=7)

        # Col 1: mask
        axes[row, 1].imshow(vd.mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Input Mask", fontsize=7)

        # Col 2: reprojection
        reproj = reproject_occupancy(occupancy, vd, voxel_res, projection, scale)
        axes[row, 2].imshow(reproj, cmap="Blues", vmin=0, vmax=1)
        axes[row, 2].set_title("Reprojection", fontsize=7)

        # Col 3: residual
        residual = np.abs(vd.mask.astype(float) - reproj.astype(float))
        axes[row, 3].imshow(residual, cmap="hot", vmin=0, vmax=1)
        axes[row, 3].set_title("Residual", fontsize=7)

        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle(f"Patient {patient_id} – QA Panel", fontsize=10)
    plt.tight_layout()
    out_path = out_dir / "qa_panel.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved QA panel: %s", out_path)


def save_volume_slices(
    occupancy: np.ndarray,
    temp_volume: np.ndarray,
    patient_id: str,
    out_dir: Path,
) -> None:
    """Save 3 orthogonal mid-slice views of occupancy and temperature."""
    if not HAS_MPL:
        return
    R = occupancy.shape[0]
    mid = R // 2

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    slices_occ  = [occupancy[mid, :, :],
                   occupancy[:, mid, :],
                   occupancy[:, :, mid]]
    slices_temp = [temp_volume[mid, :, :],
                   temp_volume[:, mid, :],
                   temp_volume[:, :, mid]]
    titles = ["Sagittal (X)", "Coronal (Y)", "Axial (Z)"]

    for c, (so, st, tt) in enumerate(zip(slices_occ, slices_temp, titles)):
        axes[0, c].imshow(so, cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"Occupancy – {tt}", fontsize=8)
        axes[0, c].axis("off")

        st_masked = np.where(so, st, np.nan)
        axes[1, c].imshow(st_masked, cmap="inferno", interpolation="nearest")
        axes[1, c].set_title(f"Temperature – {tt}", fontsize=8)
        axes[1, c].axis("off")

    fig.suptitle(f"Patient {patient_id} – Volume Slices", fontsize=10)
    plt.tight_layout()
    out_path = out_dir / "volume_slices.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved volume slices: %s", out_path)


# ---------------------------------------------------------------------------
# Per-patient pipeline
# ---------------------------------------------------------------------------

def load_patient_views(
    patient_dir: Path,
    patient_id: str,
    min_views: int,
    fallback_confidence: float = 0.5,
) -> Tuple[List[ViewData], PatientRecord]:
    """
    Discover and load all available views for a patient.
    Returns (views list, PatientRecord).
    Raises ValueError if fewer than min_views are available.
    """
    record = PatientRecord(patient_id=patient_id)
    view_files = find_view_files(patient_dir)

    views: List[ViewData] = []

    for view_name in VIEW_NAMES:
        mask_path = view_files[view_name]["mask"]
        rad_path  = view_files[view_name]["radiometric"]

        if mask_path is None:
            LOG.debug("  [%s] View '%s' – mask missing, skipping",
                      patient_id, view_name)
            continue

        try:
            raw_mask = load_mask(mask_path)
            target_shape = None
            radiometric = None

            if rad_path is not None:
                try:
                    radiometric = load_image_any(rad_path).astype(np.float32)
                    if radiometric.ndim == 3:
                        radiometric = radiometric[..., 0]
                    target_shape = radiometric.shape[:2]
                except Exception as e:
                    LOG.warning("  [%s] Could not load radiometric for '%s': %s",
                                patient_id, view_name, e)
                    radiometric = None

            mask = load_mask(mask_path, target_shape=target_shape)
            mask = clean_mask(mask)

            # Check for degenerate mask
            if mask.sum() < 50:
                LOG.warning("  [%s] View '%s' mask nearly empty after cleanup",
                            patient_id, view_name)
                confidence = fallback_confidence
            else:
                confidence = 1.0

            vd = ViewData(
                view_name=view_name,
                yaw_deg=VIEW_YAWS[view_name],
                mask=mask,
                radiometric=radiometric,
                mask_path=str(mask_path),
                radiometric_path=str(rad_path) if rad_path else "",
                confidence=confidence,
            )
            views.append(vd)

            record.available_views.append(view_name)
            record.mask_paths[view_name] = str(mask_path)
            record.radiometric_paths[view_name] = str(rad_path) if rad_path else ""
            record.image_shapes[view_name] = list(mask.shape)

            LOG.info("  [%s] ✓ Loaded view '%s' mask=%s rad=%s conf=%.2f",
                     patient_id, view_name, mask.shape,
                     radiometric.shape if radiometric is not None else "N/A",
                     confidence)

        except Exception as e:
            LOG.error("  [%s] Failed to load view '%s': %s",
                      patient_id, view_name, e)
            LOG.debug(traceback.format_exc())

    if len(views) < min_views:
        raise ValueError(
            f"Patient {patient_id}: only {len(views)} views available "
            f"(need {min_views})"
        )

    return views, record


def process_patient(
    patient_id: str,
    patient_dir: Path,
    output_root: Path,
    voxel_res: int = 64,
    projection: str = "orthographic",
    carving_mode: str = "soft",
    min_views: int = 3,
    method: str = "baseline",
    save_debug: bool = True,
    scale: float = 1.0,
    focal_scale: float = 2.0,
) -> PatientRecord:
    """Full reconstruction pipeline for a single patient."""
    t0 = time.perf_counter()
    patient_out = output_root / "per_patient" / patient_id
    patient_out.mkdir(parents=True, exist_ok=True)

    debug_dir = patient_out / "debug_projections"
    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("=" * 60)
    LOG.info("Processing patient: %s", patient_id)

    # ------------------------------------------------------------------ Load
    views, record = load_patient_views(patient_dir, patient_id, min_views)

    # -------------------------------------------------------- Reconstruction
    if method == "baseline":
        LOG.info("Running baseline voxel carving (mode=%s, min_views=%d, proj=%s)",
                 carving_mode, min_views, projection)
        t1 = time.perf_counter()
        occupancy, temp_volume, vote_count = voxel_carving(
            views, voxel_res, projection, carving_mode,
            min_views, scale, focal_scale
        )
        LOG.info("Carving done in %.1f s  – occupied: %d / %d voxels",
                 time.perf_counter() - t1, occupancy.sum(), voxel_res ** 3)

    elif method == "probabilistic":
        LOG.info("Running probabilistic occupancy fusion (proj=%s)", projection)
        t1 = time.perf_counter()
        prob_vol, uncertainty = probabilistic_occupancy(
            views, voxel_res, projection, scale, focal_scale
        )
        occupancy  = prob_vol > 0.5
        temp_volume = np.zeros_like(prob_vol, dtype=np.float32)
        vote_count  = np.zeros_like(prob_vol, dtype=np.uint8)
        LOG.info("Probabilistic fusion done in %.1f s", time.perf_counter() - t1)
        if save_debug:
            save_npy(prob_vol,    debug_dir / "prob_volume.npy")
            save_npy(uncertainty, debug_dir / "uncertainty.npy")

    else:
        raise ValueError(f"Unknown method: {method}")

    record.voxel_grid_shape = list(occupancy.shape)

    # ----------------------------------------------------------- Save volumes
    save_npy(occupancy.astype(np.uint8), patient_out / "occupancy.npy")
    save_npy(temp_volume,               patient_out / "temperature_volume.npy")
    LOG.info("Saved occupancy.npy and temperature_volume.npy")

    save_nifti(occupancy.astype(np.float32), patient_out / "occupancy.nii.gz")
    save_nifti(temp_volume,                  patient_out / "temperature_volume.nii.gz")

    save_mesh_ply(occupancy, temp_volume, patient_out / "reconstruction_mesh.ply")

    # ----------------------------------------------------------- Debug saves
    if save_debug:
        save_npy(vote_count, debug_dir / "vote_count.npy")

    # ----------------------------------------------------------- QA metrics
    LOG.info("Computing quality metrics…")
    metrics = compute_quality_metrics(
        occupancy, temp_volume, views,
        voxel_res, projection, scale, focal_scale
    )
    metrics["processing_time_s"] = round(time.perf_counter() - t0, 2)
    metrics["n_views_used"] = len(views)
    metrics["method"] = method

    record.metrics = metrics
    record.reconstruction_status = "success"

    # Save metrics JSON
    with open(patient_out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    LOG.info("Metrics: IoU=%.3f  temp_mean=%.2f  fill=%.4f  t=%.1fs",
             metrics.get("mean_iou", 0),
             metrics.get("temp_mean", 0),
             metrics.get("fill_fraction", 0),
             metrics.get("processing_time_s", 0))

    # -------------------------------------------------------- Visualisation
    save_qa_panel(views, occupancy, temp_volume, patient_id,
                  patient_out, voxel_res, projection, scale)
    save_volume_slices(occupancy, temp_volume, patient_id, patient_out)

    # ---------------------------------- Save patient record JSON
    record_dict = asdict(record)
    with open(patient_out / "patient_record.json", "w") as f:
        json.dump(record_dict, f, indent=2)

    LOG.info("Patient %s done in %.1f s  [%s]",
             patient_id, time.perf_counter() - t0, record.reconstruction_status)
    return record


# ---------------------------------------------------------------------------
# Global pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_root: Path,
    output_root: Path,
    voxel_res: int = 64,
    projection: str = "orthographic",
    carving_mode: str = "soft",
    min_views: int = 3,
    method: str = "baseline",
    save_debug: bool = True,
    dry_run: bool = False,
    scale: float = 1.0,
    focal_scale: float = 2.0,
) -> None:
    """Discover all patients and run reconstruction pipeline."""
    output_root.mkdir(parents=True, exist_ok=True)

    # Attach file logger
    global LOG
    LOG = setup_logger(output_root / "pipeline.log")
    LOG.info("Pipeline start  input=%s  output=%s  res=%d  method=%s",
             input_root, output_root, voxel_res, method)

    patients = discover_patients(input_root)
    LOG.info("Discovered %d patient folders", len(patients))

    if dry_run:
        LOG.info("DRY-RUN mode: processing first patient only")
        patients = dict(list(patients.items())[:1])

    all_records: List[PatientRecord] = []
    failures = []

    for patient_id, patient_dir in patients.items():
        try:
            record = process_patient(
                patient_id=patient_id,
                patient_dir=patient_dir,
                output_root=output_root,
                voxel_res=voxel_res,
                projection=projection,
                carving_mode=carving_mode,
                min_views=min_views,
                method=method,
                save_debug=save_debug,
                scale=scale,
                focal_scale=focal_scale,
            )
            all_records.append(record)
        except Exception as e:
            LOG.error("FAILED patient %s: %s", patient_id, e)
            LOG.debug(traceback.format_exc())
            failures.append({"patient_id": patient_id, "error": str(e)})
            failed_rec = PatientRecord(patient_id=patient_id)
            failed_rec.reconstruction_status = f"failed: {e}"
            all_records.append(failed_rec)

    # ------------------------------------------------------- Summary CSV
    csv_path = output_root / "report_summary.csv"
    fieldnames = [
        "patient_id", "reconstruction_status", "n_views_used",
        "mean_iou", "fill_fraction", "temp_mean", "temp_std",
        "occupied_voxels", "n_connected_components",
        "disconnected_anatomy", "unrealistic_aspect_ratio",
        "processing_time_s", "method",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in all_records:
            row = {"patient_id": rec.patient_id,
                   "reconstruction_status": rec.reconstruction_status}
            row.update(rec.metrics)
            writer.writerow(row)
    LOG.info("Saved summary CSV: %s", csv_path)

    # ------------------------------------------------------- Aggregate stats
    success = [r for r in all_records if r.reconstruction_status == "success"]
    n_total = len(all_records)
    n_success = len(success)

    agg: Dict = {
        "total_patients": n_total,
        "success": n_success,
        "failed": n_total - n_success,
        "success_rate": round(n_success / max(n_total, 1), 3),
    }
    if success:
        ious = [r.metrics.get("mean_iou", 0) for r in success]
        times = [r.metrics.get("processing_time_s", 0) for r in success]
        agg["avg_mean_iou"]          = round(float(np.mean(ious)), 4)
        agg["avg_processing_time_s"] = round(float(np.mean(times)), 2)
        agg["failures"] = failures

    with open(output_root / "aggregate_report.json", "w") as f:
        json.dump(agg, f, indent=2)

    LOG.info("=" * 60)
    LOG.info("PIPELINE COMPLETE")
    LOG.info("  Patients processed : %d", n_total)
    LOG.info("  Successful         : %d  (%.1f%%)",
             n_success, 100 * agg["success_rate"])
    LOG.info("  Failed             : %d", n_total - n_success)
    if success:
        LOG.info("  Avg IoU            : %.4f", agg.get("avg_mean_iou", 0))
        LOG.info("  Avg time / patient : %.1f s", agg.get("avg_processing_time_s", 0))
    LOG.info("Outputs written to: %s", output_root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3D breast volume reconstruction from multi-view thermography"
    )
    p.add_argument("--input-root",    default="data/organized_by_patient_watershed",
                   help="Root folder produced by watershed_background_removal.py")
    p.add_argument("--output-root",   default="data/reconstruction_output",
                   help="Output root for all reconstructions and reports")
    p.add_argument("--voxel-res",     type=int, default=64,
                   help="Voxel grid resolution (cubic).  64=fast, 128=better")
    p.add_argument("--projection",    choices=["orthographic", "perspective"],
                   default="orthographic",
                   help="Camera projection model")
    p.add_argument("--carving-mode",  choices=["strict", "soft"],
                   default="soft",
                   help="strict=all views required; soft=min_views threshold")
    p.add_argument("--min-views",     type=int, default=3,
                   help="Minimum views for soft carving mode")
    p.add_argument("--method",        choices=["baseline", "probabilistic"],
                   default="baseline",
                   help="Reconstruction method")
    p.add_argument("--save-debug",    choices=["true", "false"], default="true",
                   help="Save intermediate debug artifacts")
    p.add_argument("--scale",         type=float, default=1.0,
                   help="Half-extent of voxel grid in world units")
    p.add_argument("--focal-scale",   type=float, default=2.0,
                   help="Perspective focal scale multiplier")
    p.add_argument("--dry-run",       action="store_true",
                   help="Process only the first patient (quick test)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    input_root  = resolve_relative_path(Path(args.input_root), script_dir)
    output_root = resolve_relative_path(Path(args.output_root), script_dir)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    run_pipeline(
        input_root   = input_root,
        output_root  = output_root,
        voxel_res    = args.voxel_res,
        projection   = args.projection,
        carving_mode = args.carving_mode,
        min_views    = args.min_views,
        method       = args.method,
        save_debug   = args.save_debug.lower() == "true",
        dry_run      = args.dry_run,
        scale        = args.scale,
        focal_scale  = args.focal_scale,
    )


if __name__ == "__main__":
    main()
