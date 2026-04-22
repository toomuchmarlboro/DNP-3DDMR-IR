# =============================================================================
# PHASE 0 — DATASET EXPLORATION
# Purpose : Pull DMR-IR from HuggingFace and map its full structure
#           BEFORE writing any manifest parser.
# Run     : python phase0_explore.py
# Output  : Prints schema, feature names, sample record, image shapes
# =============================================================================

import os
import json
from pathlib import Path
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_NAME  = "SemilleroCV/DMR-IR"
CACHE_DIR     = Path("data/raw")          # HF will cache here
OUTPUT_SCHEMA = Path("data/manifests/schema.json")
N_SAMPLES     = 3                          # how many records to inspect

# ── Pull Dataset ──────────────────────────────────────────────────────────────
def pull_dataset():
    print("[1/5] Pulling DMR-IR from HuggingFace...")
    print(f"      Cache → {CACHE_DIR.resolve()}")
    
    ds = load_dataset(
        DATASET_NAME,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True
    )
    print("      ✓ Dataset loaded.\n")
    return ds

# ── Print Top-Level Structure ─────────────────────────────────────────────────
def inspect_splits(ds):
    print("[2/5] Dataset splits:")
    for split_name, split_data in ds.items():
        print(f"      split='{split_name}' → {len(split_data)} records")
    print()

# ── Print Feature Schema ──────────────────────────────────────────────────────
def inspect_features(ds):
    print("[3/5] Feature schema (column names + types):")
    
    # Use whichever split exists
    first_split = list(ds.keys())[0]
    features = ds[first_split].features
    
    schema = {}
    for col_name, col_type in features.items():
        type_str = str(col_type)
        schema[col_name] = type_str
        print(f"      {col_name:30} → {type_str}")
    
    # Save schema to disk
    OUTPUT_SCHEMA.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SCHEMA, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"\n      Schema saved → {OUTPUT_SCHEMA}\n")
    
    return first_split, schema

# ── Inspect Sample Records ────────────────────────────────────────────────────
def inspect_samples(ds, split_name, schema):
    print(f"[4/5] Inspecting {N_SAMPLES} sample records from split='{split_name}':")
    
    img_columns  = []   # columns that contain images
    meta_columns = []   # columns that contain metadata
    
    for col, type_str in schema.items():
        if "Image" in type_str or "image" in type_str.lower():
            img_columns.append(col)
        else:
            meta_columns.append(col)
    
    print(f"\n      Image columns  : {img_columns}")
    print(f"      Meta  columns  : {meta_columns}\n")
    
    for i in range(min(N_SAMPLES, len(ds[split_name]))):
        record = ds[split_name][i]
        print(f"  ── Record {i} ──────────────────────────────")
        
        # Print metadata fields
        for col in meta_columns:
            val = record.get(col, "N/A")
            print(f"      {col:30} = {val}")
        
        # Print image info (shape, dtype, range)
        for col in img_columns:
            img = record.get(col)
            if img is not None:
                arr = np.array(img)
                print(f"      {col:30} = shape={arr.shape} dtype={arr.dtype} "
                      f"min={arr.min():.1f} max={arr.max():.1f}")
            else:
                print(f"      {col:30} = None")
        print()
    
    return img_columns, meta_columns

# ── Visualize One Patient Sample ──────────────────────────────────────────────
def visualize_sample(ds, split_name, img_columns):
    print("[5/5] Rendering sample images for visual check...")
    record = ds[split_name][0]
    
    valid_imgs = []
    valid_titles = []
    for col in img_columns:
        img = record.get(col)
        if img is not None:
            valid_imgs.append(np.array(img))
            valid_titles.append(col)
    
    if not valid_imgs:
        print("      No images to display.")
        return
    
    n = len(valid_imgs)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    
    for ax, img_arr, title in zip(axes, valid_imgs, valid_titles):
        if img_arr.ndim == 2 or img_arr.shape[2] == 1:
            ax.imshow(img_arr, cmap="inferno")   # thermal colormap
        else:
            ax.imshow(img_arr)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    
    plt.suptitle("DMR-IR — Patient 0 Sample Images", fontsize=12)
    plt.tight_layout()
    plt.savefig("data/manifests/sample_images.png", dpi=120)
    plt.show()
    print("      Figure saved → data/manifests/sample_images.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds                        = pull_dataset()
    inspect_splits(ds)
    first_split, schema       = inspect_features(ds)
    img_cols, meta_cols       = inspect_samples(ds, first_split, schema)
    visualize_sample(ds, first_split, img_cols)
    
    print("\n✅ Exploration complete.")
    print("   → Share the terminal output + schema.json")
    print("   → We write phase0_manifest.py next.")