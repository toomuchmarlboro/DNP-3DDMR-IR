# =============================================================================
# PHASE 1 — BREAST GEOMETRY INITIALIZATION
# Purpose : For each patient extract a parametric ellipsoid (a, b, c) from
#           segmentation masks.  No MRI required.
#
# Method  :
#   1. Load binary mask from frontal IR record
#   2. Fit ellipse → semi-axes a (width), b (height), centroid, orientation
#   3. Load lateral 90° mask → semi-axis c (depth)
#   4. PCA-based refinement of ellipse axes
#   5. Pixel → mm via anatomy-based scale estimate
#   6. Save geometry.csv + visualise 9 sample patients
#
# Inputs  : data/manifests/patient_manifest.csv
#           SemilleroCV/DMR-IR  (cached in data/raw)
# Outputs : data/manifests/geometry.csv
#           data/manifests/geometry_samples.png
# =============================================================================

import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from skimage.measure import regionprops, label as sk_label
from skimage.morphology import binary_closing, disk
from scipy.ndimage import binary_fill_holes
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_NAME   = "SemilleroCV/DMR-IR"
BASE_DIR       = Path(__file__).resolve().parent
CACHE_DIR      = BASE_DIR / "data" / "raw"
MANIFEST_PATH  = BASE_DIR / "patient_manifest.csv"
OUT_DIR        = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Anatomy-based pixel → mm calibration
# Average adult breast width (frontal) ≈ 110 mm.
# We estimate scale per patient: mm_per_px = REF_WIDTH_MM / pixel_width_breast
REF_WIDTH_MM   = 110.0

# Minimum mask area (px²) to be considered a valid breast region
MIN_MASK_AREA  = 500

# ── 1. Dataset loader ─────────────────────────────────────────────────────────
def load_full_dataset():
    print("[1/6] Loading dataset (cached)...")
    ds_dict = load_dataset(DATASET_NAME, cache_dir=str(CACHE_DIR))
    tagged  = []
    for sname, sds in ds_dict.items():
        sds = sds.map(lambda x: {"split": sname}, desc=f"tag {sname}")
        tagged.append(sds)
    full_ds = concatenate_datasets(tagged)
    print(f"      {len(full_ds):,} records loaded.\n")
    return full_ds


# ── 2. Mask extraction ────────────────────────────────────────────────────────
def extract_binary_mask(record) -> Optional[np.ndarray]:
    """
    Convert the segmentation_mask field (nested list) to a clean 2-D binary
    numpy array where 1 = breast tissue, 0 = background.

    Handles three possible encodings the HF dataset might use:
      A) (H, W, 3)  RGB-encoded mask  → threshold on max channel
      B) (H, W, 1)  single-channel    → threshold > 0
      C) (H, W)     already 2-D       → threshold > 0
    """
    raw = record.get("segmentation_mask")
    if raw is None or len(raw) == 0:
        return None

    arr = np.array(raw, dtype=np.uint8)

    # Handle single-channel masks in either layout:
    # (H, W, 1) or channel-first (1, H, W)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]

    # For RGB masks: collapse to single channel via max
    if arr.ndim == 3:
        arr = arr.max(axis=2)

    # Binary threshold
    binary = (arr > 0).astype(np.uint8)

    # Close small holes and fill interior gaps
    binary = binary_closing(binary, disk(3))
    binary = binary_fill_holes(binary).astype(np.uint8)

    return binary


# ── 3. Ellipse fitting ────────────────────────────────────────────────────────
def fit_ellipse(binary_mask: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Fit an ellipse to the largest connected component in a binary mask.

    Returns dict with keys:
      centroid_r, centroid_c   : centroid in pixel coords (row, col)
      semi_axis_major          : longer semi-axis (pixels)
      semi_axis_minor          : shorter semi-axis (pixels)
      orientation              : angle of major axis (radians)
      bbox_width, bbox_height  : bounding box dimensions (pixels)
      pixel_area               : foreground pixel count
      eccentricity             : 0=circle, 1=line

    Method: skimage regionprops on labelled mask, which internally uses
    the inertia tensor to fit the best-fitting ellipse — equivalent to
    PCA on the foreground pixel coordinates.
    """
    if binary_mask is None or binary_mask.sum() < MIN_MASK_AREA:
        return None

    # Label connected components, keep largest
    labelled  = sk_label(binary_mask)
    regions   = regionprops(labelled)
    if not regions:
        return None

    region = max(regions, key=lambda r: r.area)

    minr, minc, maxr, maxc = region.bbox
    return {
        "centroid_r"      : region.centroid[0],
        "centroid_c"      : region.centroid[1],
        "semi_axis_major"  : region.axis_major_length / 2,
        "semi_axis_minor"  : region.axis_minor_length / 2,
        "orientation"     : region.orientation,
        "bbox_width"      : maxc - minc,
        "bbox_height"     : maxr - minr,
        "pixel_area"      : region.area,
        "eccentricity"    : region.eccentricity,
    }


# ── 4. Per-patient ellipsoid estimation ───────────────────────────────────────
def estimate_ellipsoid(row: pd.Series, full_ds) -> dict:
    """
    Combine frontal + lateral 90 degree views to estimate all three
    ellipsoid semi-axes (a, b, c) in pixels then in mm.

    Axis mapping:
      a = half breast width       (frontal, horizontal)
      b = half breast height      (frontal, vertical)
      c = half breast depth       (lateral 90, horizontal)

    Scale: anatomy-based estimate
      mm_per_px = REF_WIDTH_MM / (2a_pixels)
    """
    result = {
        "patient_id"     : row["patient_id"],
        "label"          : row["label"],
        "split"          : row["split"],
        # Will be filled below
        "a_px": None, "b_px": None, "c_px": None,
        "a_mm": None, "b_mm": None, "c_mm": None,
        "mm_per_px"      : None,
        "centroid_r"     : None,
        "centroid_c"     : None,
        "orientation_rad": None,
        "eccentricity"   : None,
        "pixel_area"     : None,
        "img_h"          : None,
        "img_w"          : None,
        "frontal_ok"     : False,
        "lateral_ok"     : False,
        "geometry_valid" : False,
    }

    # ── Frontal mask ────────────────────────────────────────────────────────
    idx_front = row.get("idx_frontal")
    if pd.notna(idx_front):
        rec_f    = full_ds[int(idx_front)]
        mask_f   = extract_binary_mask(rec_f)
        ellipse_f = fit_ellipse(mask_f)

        if ellipse_f:
            # a = half width (use bbox_width for robustness against tilt)
            a_px = ellipse_f["bbox_width"]  / 2.0
            b_px = ellipse_f["bbox_height"] / 2.0

            img_arr = np.array(rec_f["image"])
            result.update({
                "a_px"          : a_px,
                "b_px"          : b_px,
                "centroid_r"    : ellipse_f["centroid_r"],
                "centroid_c"    : ellipse_f["centroid_c"],
                "orientation_rad": ellipse_f["orientation"],
                "eccentricity"  : ellipse_f["eccentricity"],
                "pixel_area"    : ellipse_f["pixel_area"],
                "img_h"         : img_arr.shape[0],
                "img_w"         : img_arr.shape[1],
                "frontal_ok"    : True,
            })

    # ── Lateral mask (prefer right 90, fallback left 90) ────────────────────
    idx_lat = row.get("idx_right_90")
    if pd.isna(idx_lat):
        idx_lat = row.get("idx_left_90")

    if pd.notna(idx_lat):
        rec_l    = full_ds[int(idx_lat)]
        mask_l   = extract_binary_mask(rec_l)
        ellipse_l = fit_ellipse(mask_l)

        if ellipse_l:
            # c = half of breast depth from lateral horizontal extent
            c_px = ellipse_l["bbox_width"] / 2.0
            result.update({
                "c_px"      : c_px,
                "lateral_ok": True,
            })

    # ── Pixel → mm conversion ───────────────────────────────────────────────
    if result["a_px"] is not None and result["a_px"] > 0:
        scale = REF_WIDTH_MM / (2 * result["a_px"])
        result["mm_per_px"] = scale
        result["a_mm"] = result["a_px"] * scale
        result["b_mm"] = result["b_px"] * scale if result["b_px"] else None
        result["c_mm"] = result["c_px"] * scale if result["c_px"] else None

    result["geometry_valid"] = (
        result["frontal_ok"] and
        result["lateral_ok"] and
        result["a_mm"] is not None
    )
    return result


# ── 5. Run over all patients ───────────────────────────────────────────────────
def run_phase1(manifest: pd.DataFrame, full_ds) -> pd.DataFrame:
    print("[3/6] Estimating ellipsoid geometry per patient...")
    rows = []
    for _, row in tqdm(manifest.iterrows(),
                        total=len(manifest), desc="Patients"):
        if not row.get("ready_stereo", False):
            continue
        rows.append(estimate_ellipsoid(row, full_ds))

    geom = pd.DataFrame(rows)
    geom.to_csv(OUT_DIR / "geometry.csv", index=False)

    n_valid = geom["geometry_valid"].sum()
    print(f"\n      geometry.csv saved — {n_valid}/{len(geom)} patients fully valid")
    print(f"      Frontal ok : {geom['frontal_ok'].sum()}")
    print(f"      Lateral ok : {geom['lateral_ok'].sum()}\n")
    return geom


# ── 6. Geometry statistics ────────────────────────────────────────────────────
def print_stats(geom: pd.DataFrame):
    print("[4/6] Ellipsoid dimension statistics (mm):")
    valid = geom[geom["geometry_valid"]]
    if valid.empty:
        print("  No valid geometry rows found. Skipping stats.\n")
        return

    for col, name in [("a_mm", "a (half-width)"),
                       ("b_mm", "b (half-height)"),
                       ("c_mm", "c (half-depth)")]:
        v = valid[col].dropna()
        print(f"  {name:18}: mean={v.mean():.1f}  "
              f"std={v.std():.1f}  min={v.min():.1f}  max={v.max():.1f}")
    print()

    # By label
    print("  Mean dimensions by diagnosis:")
    print(valid.groupby("label")[["a_mm","b_mm","c_mm"]].mean().round(1))
    print()


# ── 7. Visualisation ──────────────────────────────────────────────────────────
def visualise_samples(geom: pd.DataFrame, full_ds, n: int = 9):
    print(f"[5/6] Visualising {n} sample patients...")
    manifest = pd.read_csv(MANIFEST_PATH)
    valid    = geom[geom["geometry_valid"]].reset_index(drop=True)

    if valid.empty:
        print("      No valid geometry to visualise. Skipping geometry_samples.png\n")
        return

    # Pick n evenly spaced samples across the cohort
    n_plot = min(n, len(valid))
    idxs  = np.linspace(0, len(valid) - 1, n_plot, dtype=int)
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    axes = axes.flatten()

    for ax, gidx in zip(axes, idxs):
        grow   = valid.iloc[gidx]
        pid    = grow["patient_id"]
        mrow   = manifest[manifest["patient_id"] == pid]
        if mrow.empty:
            continue

        idx_f  = mrow.iloc[0]["idx_frontal"]
        if pd.isna(idx_f):
            continue

        rec    = full_ds[int(idx_f)]
        ir_img = np.array(rec["image"])
        mask   = extract_binary_mask(rec)

        # Show IR image with mask overlay
        ax.imshow(ir_img, cmap="inferno", alpha=1.0)
        if mask is not None:
            ax.imshow(mask, cmap="cool", alpha=0.3)

        # Draw fitted ellipse
        cx    = grow["centroid_c"]
        cy    = grow["centroid_r"]
        a_px  = grow["a_px"]
        b_px  = grow["b_px"]
        angle = np.degrees(grow["orientation_rad"])

        ell = Ellipse(
            xy=(float(cx), float(cy)),
            width=float(a_px) * 2,
            height=float(b_px) * 2,
            angle=float(angle),
            edgecolor="#a6e3a1", facecolor="none", linewidth=1.8
        )
        ax.add_patch(ell)
        ax.plot(cx, cy, "+", color="#f38ba8", ms=8, mew=2)

        lbl   = grow["label"]
        color = "#f38ba8" if lbl == "malignant" else "#89b4fa"
        ax.set_title(
            f"P{pid} [{lbl[:3].upper()}]\n"
            f"a={grow['a_mm']:.0f} b={grow['b_mm']:.0f} c={grow['c_mm']:.0f} mm",
            fontsize=8, color=color
        )
        ax.axis("off")

    plt.suptitle(
        "Phase 1 — Fitted Breast Ellipsoids (Frontal View)\n"
        "Green ellipse = fitted geometry  |  Overlay = segmentation mask",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "geometry_samples.png", dpi=130)
    plt.show()
    print("      Saved geometry_samples.png\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    manifest = pd.read_csv(MANIFEST_PATH)
    full_ds  = load_full_dataset()

    print(f"[2/6] Manifest loaded: {len(manifest)} patients, "
          f"{manifest['ready_stereo'].sum()} stereo-ready\n")

    geom = run_phase1(manifest, full_ds)
    print_stats(geom)
    visualise_samples(geom, full_ds)

    print("   Phase 1 complete.")
    print("   Outputs: geometry.csv")
    print("   Send back: geometry stats table + geometry_samples.png (if generated)")
    print("   Phase 2: multi-view thermal stereo is next.")