from pathlib import Path
import csv
import os

import numpy as np
from datasets import load_dataset


DATASET_ID = "SemilleroCV/DMR-IR"
WORKSPACE_ROOT = Path(__file__).resolve().parent
CACHE_DIR = WORKSPACE_ROOT / "data" / "raw"
EXPORT_DIR = WORKSPACE_ROOT / "data" / "local" / "DMR-IR"


def safe_text(value):
	return str(value).strip().replace(" ", "_").replace("/", "-")


def decode_class_value(feature, value):
	if value is None:
		return ""
	try:
		return feature.int2str(value)
	except Exception:
		return ""


def export_split(split_name, split_ds):
	features = split_ds.features
	scalar_columns = [
		c for c in split_ds.column_names if c not in {"image", "segmentation_mask", "text_embedding"}
	]
	class_columns = [c for c in scalar_columns if hasattr(features[c], "int2str")]

	split_dir = EXPORT_DIR / split_name
	images_dir = split_dir / "images"
	masks_dir = split_dir / "masks"
	embeddings_dir = split_dir / "embeddings"
	split_dir.mkdir(parents=True, exist_ok=True)
	images_dir.mkdir(parents=True, exist_ok=True)
	masks_dir.mkdir(parents=True, exist_ok=True)
	embeddings_dir.mkdir(parents=True, exist_ok=True)

	csv_path = split_dir / "metadata.csv"
	temp_csv_path = split_dir / "metadata.tmp.csv"
	fieldnames = (
		["index", "image_path", "mask_path", "embedding_path", "label_name"]
		+ scalar_columns
		+ [f"{c}_name" for c in class_columns]
	)

	with temp_csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()

		total = len(split_ds)
		for idx, ex in enumerate(split_ds):
			patient_id = safe_text(ex.get("patient_id", "unknown"))
			record = safe_text(ex.get("record", idx))
			label = ex.get("label")
			label_name = decode_class_value(features["label"], label) if "label" in features else ""
			label_folder = safe_text(label_name if label_name else label if label is not None else "unknown")

			base_name = f"{patient_id}_{record}_{idx:05d}"
			image_rel = Path(split_name) / "images" / label_folder / f"{base_name}.tiff"
			mask_rel = Path(split_name) / "masks" / label_folder / f"{base_name}.npy"
			embedding_rel = Path(split_name) / "embeddings" / label_folder / f"{base_name}.npy"

			image_path = EXPORT_DIR / image_rel
			mask_path = EXPORT_DIR / mask_rel
			embedding_path = EXPORT_DIR / embedding_rel
			image_path.parent.mkdir(parents=True, exist_ok=True)
			mask_path.parent.mkdir(parents=True, exist_ok=True)
			embedding_path.parent.mkdir(parents=True, exist_ok=True)

			if not image_path.exists():
				ex["image"].save(image_path)
			if not mask_path.exists():
				np.save(mask_path, np.asarray(ex["segmentation_mask"], dtype=np.uint8))
			if not embedding_path.exists():
				np.save(embedding_path, np.asarray(ex["text_embedding"], dtype=np.float32))

			row = {
				"index": idx,
				"image_path": image_rel.as_posix(),
				"mask_path": mask_rel.as_posix(),
				"embedding_path": embedding_rel.as_posix(),
				"label_name": label_name,
			}
			for col in scalar_columns:
				row[col] = ex.get(col)
			for col in class_columns:
				row[f"{col}_name"] = decode_class_value(features[col], ex.get(col))

			writer.writerow(row)

			if (idx + 1) % 100 == 0 or (idx + 1) == total:
				print(f"[{split_name}] exported {idx + 1}/{total}")

	os.replace(temp_csv_path, csv_path)


def main():
	print(f"Loading dataset {DATASET_ID} with cache at: {CACHE_DIR}")
	dataset = load_dataset(DATASET_ID, cache_dir=str(CACHE_DIR))
	print(dataset)

	EXPORT_DIR.mkdir(parents=True, exist_ok=True)
	for split_name, split_ds in dataset.items():
		print(f"\nExporting split: {split_name}")
		export_split(split_name, split_ds)

	print(f"\nAll files exported to: {EXPORT_DIR}")


if __name__ == "__main__":
	main()