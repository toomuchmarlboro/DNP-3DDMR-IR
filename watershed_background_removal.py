# Mask-only (default)
#######  & "C:/Users/LENOVO THINKPAD T14/AppData/Local/Python/pythoncore-3.11-64/python.exe" watershed_background_removal.py

# Mask-only + radiometric masked arrays
#######  & "C:/Users/LENOVO THINKPAD T14/AppData/Local/Python/pythoncore-3.11-64/python.exe" watershed_background_removal.py --save-radiometric-masked

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    t_min = np.nanmin(image)
    t_max = np.nanmax(image)
    if not np.isfinite(t_min) or not np.isfinite(t_max) or (t_max - t_min) <= 0:
        return np.zeros_like(image, dtype=np.uint8)
    norm = (image - t_min) / (t_max - t_min)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def load_grayscale_image(path: Path) -> np.ndarray:
    # Keep thermal intensities stable by loading with PIL first.
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # Convert color-like image to grayscale consistently.
        return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    return arr.squeeze()


def watershed_foreground_mask(norm_img: np.ndarray) -> np.ndarray:
    h, w = norm_img.shape
    elevation_map = sobel(norm_img)
    markers = np.zeros_like(norm_img, dtype=np.int32)

    # Background seeds: lower corners and optional top "neck" strip.
    markers[-1, 0] = 1
    markers[-1, -1] = 1

    neck_ratio = 0.00
    neck_limit = int(h * neck_ratio)
    if neck_limit > 0:
        markers[0:neck_limit, :] = 1

    # Foreground seed around lower center breast region.
    center_y = int(h * 0.60)
    center_x = w // 2
    rad_x = max(1, int(w * 0.05))
    rad_y = max(1, int(h * 0.05))

    y1 = max(0, center_y - rad_y)
    y2 = min(h, center_y + rad_y)
    x1 = max(0, center_x - rad_x)
    x2 = min(w, center_x + rad_x)
    markers[y1:y2, x1:x2] = 2

    segmentation = watershed(elevation_map, markers)
    foreground = ndi.binary_fill_holes(segmentation - 1)
    return (foreground.astype(np.uint8) * 255)


def make_counts_dict() -> dict:
    return {
        "Anterior (Front)": {"Normal": 0, "Abnormal": 0},
        "Left Oblique (45°)": {"Normal": 0, "Abnormal": 0},
        "Right Oblique (45°)": {"Normal": 0, "Abnormal": 0},
        "Left Lateral (90°)": {"Normal": 0, "Abnormal": 0},
        "Right Lateral (90°)": {"Normal": 0, "Abnormal": 0},
        "Unknown": {"Normal": 0, "Abnormal": 0},
        "Skipped": 0,
    }


def map_label(category_name: str) -> str:
    category = category_name.strip().lower()
    if category == "benign":
        return "Normal"
    if category == "malignant":
        return "Abnormal"
    return "Abnormal"


def canonical_view_name(file_stem: str) -> str:
    known = {
        "Anterior (Front)",
        "Left Oblique (45°)",
        "Right Oblique (45°)",
        "Left Lateral (90°)",
        "Right Lateral (90°)",
    }
    return file_stem if file_stem in known else "Unknown"


def plot_distribution(counts: dict, output_root: Path, show_plot: bool) -> None:
    views = [
        "Anterior (Front)",
        "Left Oblique (45°)",
        "Right Oblique (45°)",
        "Left Lateral (90°)",
        "Right Lateral (90°)",
    ]
    normal_vals = [counts[v]["Normal"] for v in views]
    abnormal_vals = [counts[v]["Abnormal"] for v in views]

    x = np.arange(len(views))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, normal_vals, width, label="Normal", color="green", alpha=0.7)
    rects2 = ax.bar(x + width / 2, abnormal_vals, width, label="Abnormal", color="red", alpha=0.7)

    ax.set_ylabel("Number of Images")
    ax.set_title("Organized-by-Patient Distribution by View and Class")
    ax.set_xticks(x)
    ax.set_xticklabels(views, rotation=20, ha="right")
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()

    plot_path = output_root / "dataset_distribution.png"
    plt.savefig(plot_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close(fig)
    print(f"Distribution plot saved to: {plot_path}")


def process_organized_dataset(
    source_root: Path,
    output_root: Path,
    max_images: int,
    show_plot: bool,
    save_radiometric_masked: bool,
) -> None:
    counts = make_counts_dict()

    image_paths = sorted(
        list(source_root.glob("Patient_*/*/*.tif"))
        + list(source_root.glob("Patient_*/*/*.tiff"))
        + list(source_root.glob("Patient_*/*/*.png"))
    )

    print(f"Found {len(image_paths)} candidate images in {source_root}")

    processed = 0
    for i, img_path in enumerate(image_paths, start=1):
        try:
            if max_images > 0 and processed >= max_images:
                break

            patient_id = img_path.parents[1].name
            category = img_path.parents[0].name
            label_name = map_label(category)
            view_name = canonical_view_name(img_path.stem)

            raw = load_grayscale_image(img_path)
            norm_img = normalize_to_uint8(raw)
            mask = watershed_foreground_mask(norm_img)
            rel = img_path.relative_to(source_root)
            out_mask_path = (output_root / rel.parent / f"{img_path.stem}__mask.png")
            out_mask_npy_path = (output_root / rel.parent / f"{img_path.stem}__mask.npy")
            out_radiometric_npy_path = (output_root / rel.parent / f"{img_path.stem}__masked_radiometric.npy")

            out_mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_mask_path), mask)
            np.save(str(out_mask_npy_path), (mask > 0).astype(np.uint8))

            if save_radiometric_masked:
                # Preserve raw thermal values and only zero-out background.
                raw_masked = np.where(mask > 0, raw, 0)
                np.save(str(out_radiometric_npy_path), raw_masked)

            if view_name not in counts:
                counts["Unknown"][label_name] += 1
            else:
                counts[view_name][label_name] += 1

            processed += 1
            if processed % 50 == 0:
                print(f"Processed {processed}/{len(image_paths)}...", end="\r")

        except Exception:
            counts["Skipped"] += 1

    print("\n\n--- Processing Complete ---")
    print(f"Processed images: {processed}")
    print(f"Skipped files: {counts['Skipped']}")
    print(f"Output root: {output_root}")

    plot_distribution(counts, output_root, show_plot=show_plot)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watershed background removal for organized-by-patient thermograms")
    parser.add_argument(
        "--source-root",
        type=str,
        default="data/organized_by_patient",
        help="Input organized dataset root",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/organized_by_patient_watershed",
        help="Output root for generated masks and optional radiometric masked arrays",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum images to process (0 means all)",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the generated distribution plot window",
    )
    parser.add_argument(
        "--save-radiometric-masked",
        action="store_true",
        help="Save masked raw thermal arrays as .npy without 8-bit normalization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)

    if not source_root.is_absolute():
        source_root = script_dir / source_root
    if not output_root.is_absolute():
        output_root = script_dir / output_root

    if not source_root.exists():
        raise FileNotFoundError(f"Source folder not found: {source_root}")

    process_organized_dataset(
        source_root=source_root,
        output_root=output_root,
        max_images=args.max_images,
        show_plot=args.show_plot,
        save_radiometric_masked=args.save_radiometric_masked,
    )


if __name__ == "__main__":
    main()
