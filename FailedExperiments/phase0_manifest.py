# =============================================================================
# PHASE 0 — PATIENT MANIFEST BUILDER
# Purpose : Aggregate image-level HuggingFace records into one patient-level
#           CSV. Stores dataset indices (not pixels) to keep RAM light.
#           Also exports a protocol/view inventory and data quality report.
# Run     : python phase0_manifest.py
# Outputs :
#   data/manifests/patient_manifest.csv   ← one row per patient
#   data/manifests/view_inventory.csv     ← one row per image record
#   data/manifests/quality_report.txt     ← missing-data summary
# =============================================================================

import json
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_NAME  = "SemilleroCV/DMR-IR"
CACHE_DIR     = Path("data/raw")
OUT_DIR       = Path(r"C:\Users\LENOVO THINKPAD T14\Documents\PROPOSAL TA\files\Rodriguez-Guerrero Dataset\Breast Thermography\3D Reconstruction")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# View names exactly as they appear in ClassLabel
STATIC_VIEWS  = ["Frontal", "Right 45°", "Right 90°",
                 "Left 45°",  "Left 90°"]

# Clinical metadata columns we want to carry into the manifest
CLINICAL_COLS = [
    "age_at_visit", "body_temperature", "menopause",
    "cancer_family", "biopsy", "mammography",
    "complaints", "symptoms", "signs",
    "further_informations", "medical_further_informations",
    "text"
]

# ── Step 1 : Load and merge all splits ────────────────────────────────────────
def load_all():
    print("[1/5] Loading all splits...")
    ds_dict = load_dataset(DATASET_NAME, cache_dir=str(CACHE_DIR))

    # Tag each record with its split name before merging
    tagged = []
    for split_name, split_ds in ds_dict.items():
        split_ds = split_ds.map(
            lambda x: {"split": split_name},
            desc=f"Tagging {split_name}"
        )
        tagged.append(split_ds)

    full_ds = concatenate_datasets(tagged)
    print(f"      Total records across all splits: {len(full_ds):,}\n")
    return full_ds

# ── Step 2 : Build flat view inventory (one row per image record) ─────────────
def build_inventory(full_ds):
    print("[2/5] Building per-image view inventory...")
    rows = []

    for idx in tqdm(range(len(full_ds)), desc="Scanning records"):
        rec = full_ds[idx]

        # Decode ClassLabel integers → string names
        view_name  = full_ds.features["view"].int2str(rec["view"])
        label_name = full_ds.features["label"].int2str(rec["label"])

        # Image shape (do NOT store pixels)
        img_arr   = np.array(rec["image"])
        img_shape = img_arr.shape

        # Has segmentation mask?
        seg = rec.get("segmentation_mask")
        has_mask = (seg is not None and len(seg) > 0)

        row = {
            "global_idx"    : idx,
            "patient_id"    : rec["patient_id"],
            "split"         : rec["split"],
            "label"         : label_name,
            "protocol"      : rec["protocol"],
            "view"          : view_name,
            "record"        : rec["record"],
            "role"          : rec["role"],
            "img_h"         : img_shape[0],
            "img_w"         : img_shape[1],
            "img_channels"  : img_shape[2] if img_arr.ndim == 3 else 1,
            "has_seg_mask"  : has_mask,
            "body_temp"     : rec.get("body_temperature"),
            "age_at_visit"  : rec.get("age_at_visit"),
        }
        rows.append(row)

    inv = pd.DataFrame(rows)
    inv.to_csv(OUT_DIR / "view_inventory.csv", index=False)
    print(f"      Saved view_inventory.csv ({len(inv)} rows)\n")
    return inv

# ── Step 3 : Aggregate to patient-level manifest ───────────────────────────────
def build_manifest(inv, full_ds):
    print("[3/5] Aggregating to patient-level manifest...")
    patients = []

    for pid, grp in tqdm(inv.groupby("patient_id"),
                          desc="Patients"):

        label  = grp["label"].mode()[0]   # majority vote (should be unanimous)
        split  = grp["split"].mode()[0]

        # ── Static protocol views ────────────────────────────────────────────
        static = grp[grp["protocol"].str.lower().str.contains(
                     "static", na=False)]

        static_idx = {}
        for v in STATIC_VIEWS:
            match = static[static["view"] == v]["global_idx"].tolist()
            static_idx[v] = match[0] if match else None

        n_static_views = sum(1 for v in static_idx.values()
                             if v is not None)

        # ── Dynamic protocol frames ──────────────────────────────────────────
        dynamic = grp[grp["protocol"].str.lower().str.contains(
                      "dynamic", na=False)]
        dynamic_idx = dynamic["global_idx"].tolist()

        # ── Segmentation mask availability ───────────────────────────────────
        has_mask = grp["has_seg_mask"].any()

        # ── Clinical metadata (take first non-null value per patient) ────────
        first_rec_idx = grp["global_idx"].iloc[0]
        first_rec     = full_ds[int(first_rec_idx)]
        clinical = {col: first_rec.get(col) for col in CLINICAL_COLS}

        # ── Pipeline readiness flags ─────────────────────────────────────────
        ready_stereo  = n_static_views >= 3   # need ≥3 views for stereo
        ready_dynamic = len(dynamic_idx) >= 10 # need ≥10 frames for transient
        ready_full    = ready_stereo and ready_dynamic and has_mask

        row = {
            "patient_id"        : pid,
            "split"             : split,
            "label"             : label,
            "n_total_images"    : len(grp),
            "n_static_views"    : n_static_views,
            "n_dynamic_frames"  : len(dynamic_idx),
            "has_seg_mask"      : has_mask,
            "ready_stereo"      : ready_stereo,
            "ready_dynamic"     : ready_dynamic,
            "ready_full_pipeline": ready_full,

            # Static view indices into full_ds
            "idx_frontal"       : static_idx.get("Frontal"),
            "idx_right_45"      : static_idx.get("Right 45°"),
            "idx_right_90"      : static_idx.get("Right 90°"),
            "idx_left_45"       : static_idx.get("Left 45°"),
            "idx_left_90"       : static_idx.get("Left 90°"),

            # Dynamic frame indices (stored as JSON string)
            "dynamic_idx_list"  : json.dumps(dynamic_idx),

            **clinical
        }
        patients.append(row)

    manifest = pd.DataFrame(patients)
    manifest.to_csv(OUT_DIR / "patient_manifest.csv", index=False)
    print(f"      Saved patient_manifest.csv ({len(manifest)} patients)\n")
    return manifest

# ── Step 4 : Quality Report ────────────────────────────────────────────────────
def quality_report(manifest):
    print("[4/5] Generating quality report...")

    try:
        total    = len(manifest)
        
        if total == 0:
            lines = ["DMR-IR PATIENT MANIFEST — QUALITY REPORT",
                     "=" * 50,
                     "ERROR: No patients in manifest (empty dataset)"]
        else:
            benign   = (manifest["label"] == "benign").sum()
            malig    = (manifest["label"] == "malignant").sum()
            stereo   = manifest["ready_stereo"].sum()
            dyn      = manifest["ready_dynamic"].sum()
            full     = manifest["ready_full_pipeline"].sum()
            masked   = manifest["has_seg_mask"].sum()

            lines = [
                "DMR-IR PATIENT MANIFEST — QUALITY REPORT",
                "=" * 50,
                f"Total patients            : {total}",
                f"  Benign                  : {benign} ({benign/total*100:.1f}%)",
                f"  Malignant               : {malig}  ({malig/total*100:.1f}%)",
                "",
                "PIPELINE READINESS",
                f"  Has seg mask            : {masked}  ({masked/total*100:.1f}%)",
                f"  Ready stereo (≥3 views) : {stereo}  ({stereo/total*100:.1f}%)",
                f"  Ready dynamic (≥10 fr.) : {dyn}    ({dyn/total*100:.1f}%)",
                f"  Ready FULL pipeline     : {full}   ({full/total*100:.1f}%)",
                "",
                "STATIC VIEW COMPLETENESS (per patient)",
            ]

            for v in ["idx_frontal", "idx_right_45", "idx_right_90",
                      "idx_left_45",  "idx_left_90"]:
                n = manifest[v].notna().sum()
                lines.append(f"  {v:20}: {n}/{total} ({n/total*100:.1f}%)")

            lines += [
                "",
                "DYNAMIC FRAME COUNTS",
                f"  Mean  : {manifest['n_dynamic_frames'].mean():.1f}",
                f"  Median: {manifest['n_dynamic_frames'].median():.1f}",
                f"  Min   : {manifest['n_dynamic_frames'].min()}",
                f"  Max   : {manifest['n_dynamic_frames'].max()}",
            ]

    except Exception as e:
        lines = [
            "DMR-IR PATIENT MANIFEST — QUALITY REPORT",
            "=" * 50,
            f"ERROR generating report: {type(e).__name__}",
            str(e),
        ]

    report_str = "\n".join(lines)
    print(report_str)

    with open(OUT_DIR / "quality_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"      Saved quality_report.txt\n")

    return manifest

# ── Step 5 : Summary Plot ─────────────────────────────────────────────────────
def summary_plot(manifest):
    print("\n[5/5] Generating summary plots...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Label distribution
    manifest["label"].value_counts().plot(
        kind="bar", ax=axes[0], color=["#89b4fa", "#f38ba8"])
    axes[0].set_title("Label Distribution")
    axes[0].set_xlabel(""); axes[0].tick_params(rotation=0)

    # Static view completeness
    view_cols = ["idx_frontal","idx_right_45","idx_right_90",
                 "idx_left_45", "idx_left_90"]
    completeness = manifest[view_cols].notna().mean() * 100
    completeness.index = ["Front","R45","R90","L45","L90"]
    completeness.plot(kind="bar", ax=axes[1], color="#a6e3a1")
    axes[1].set_title("Static View Completeness (%)")
    axes[1].set_ylim(0, 110); axes[1].tick_params(rotation=0)

    # Dynamic frame distribution
    manifest["n_dynamic_frames"].plot(
        kind="hist", ax=axes[2], bins=20, color="#cba6f7")
    axes[2].set_title("Dynamic Frame Count per Patient")
    axes[2].set_xlabel("n frames")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "manifest_summary.png", dpi=120)
    plt.show()
    print(f"      Saved manifest_summary.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    full_ds  = load_all()
    inv      = build_inventory(full_ds)
    manifest = build_manifest(inv, full_ds)
    quality_report(manifest)
    summary_plot(manifest)

    print("\n Manifest complete.")
    print("   Send back: quality_report.txt + manifest_summary.png")
    print("   We write Phase 1 geometry next.")