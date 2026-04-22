"""
Reorganize DMR-IR Dataset by Patient ID and View
Creates folder structure: output_folder/Patient_ID/ViewType.tiff
"""

import os
from pathlib import Path
import pandas as pd
import shutil
import argparse
from tqdm import tqdm


class DatasetOrganizer:
    def __init__(self, source_root, output_root):
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.metadata_dict = {}
        self.stats = {
            'total_files': 0,
            'copied_files': 0,
            'skipped_files': 0,
            'errors': 0
        }
        
        # View name normalization (friendly names for folder structure)
        self.view_mapping = {
            "Frontal": "Anterior (Front)",
            "Left 45°": "Left Oblique (45°)",
            "Right 45°": "Right Oblique (45°)",
            "Left 90°": "Left Lateral (90°)",
            "Right 90°": "Right Lateral (90°)"
        }
    
    def load_metadata(self):
        """Load all metadata from CSV files."""
        print("Loading metadata...")
        for split in ["train", "test", "validation"]:
            meta_path = self.source_root / split / "metadata.csv"
            if meta_path.exists():
                print(f"  Reading {split} metadata...")
                df = pd.read_csv(meta_path)
                for _, row in df.iterrows():
                    img_relative_path = row['image_path']
                    self.metadata_dict[img_relative_path] = row
        
        print(f"Loaded metadata for {len(self.metadata_dict)} images\n")
    
    def organize_dataset(self):
        """Reorganize dataset by patient ID and view."""
        print("Organizing dataset...")
        
        # Create output root if it doesn't exist
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Process each split
        for split in ["train", "test", "validation"]:
            split_path = self.source_root / split / "images"
            if not split_path.exists():
                continue
            
            print(f"\nProcessing {split} split...")
            
            # Process each category
            for category in ["benign", "malignant"]:
                category_path = split_path / category
                if not category_path.exists():
                    continue
                
                # Get all TIFF files
                tiff_files = sorted(list(category_path.glob("*.tif")) + list(category_path.glob("*.tiff")))
                
                if not tiff_files:
                    continue
                
                print(f"  Processing {category} images ({len(tiff_files)} files)...")
                
                for img_path in tqdm(tiff_files, desc=f"  {category}", leave=False):
                    try:
                        self.stats['total_files'] += 1
                        
                        # Get metadata
                        relative_path = img_path.relative_to(self.source_root)
                        img_relative_str = str(relative_path).replace("\\", "/")
                        
                        if img_relative_str not in self.metadata_dict:
                            self.stats['skipped_files'] += 1
                            continue
                        
                        row = self.metadata_dict[img_relative_str]
                        patient_id = str(row['patient_id']).strip()
                        view_name = str(row['view_name']).strip()
                        
                        # Skip unknown values
                        if patient_id == "Unknown" or view_name == "Unknown" or pd.isna(patient_id) or pd.isna(view_name):
                            self.stats['skipped_files'] += 1
                            continue
                        
                        # Get friendly view name
                        friendly_view = self.view_mapping.get(view_name, view_name)
                        
                        # Create patient folder structure
                        patient_folder = self.output_root / f"Patient_{patient_id}" / category
                        patient_folder.mkdir(parents=True, exist_ok=True)
                        
                        # Destination file
                        dest_file = patient_folder / f"{friendly_view}.tiff"
                        
                        # Copy file
                        shutil.copy2(img_path, dest_file)
                        self.stats['copied_files'] += 1
                        
                    except Exception as e:
                        print(f"    Error processing {img_path.name}: {e}")
                        self.stats['errors'] += 1
        
        print("\n" + "="*60)
        print("ORGANIZATION COMPLETE")
        print("="*60)
        print(f"Total files processed: {self.stats['total_files']}")
        print(f"Files copied: {self.stats['copied_files']}")
        print(f"Files skipped: {self.stats['skipped_files']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Output location: {self.output_root}")
        print("="*60)
    
    def print_structure(self):
        """Print folder structure overview."""
        print("\nGenerated Folder Structure:")
        print("="*60)
        
        patient_dirs = sorted([d for d in self.output_root.iterdir() if d.is_dir()])
        
        for patient_dir in patient_dirs[:10]:  # Show first 10 patients as example
            print(f"\n{patient_dir.name}/")
            for category_dir in sorted(patient_dir.iterdir()):
                if category_dir.is_dir():
                    print(f"  {category_dir.name}/")
                    for img_file in sorted(category_dir.iterdir()):
                        if img_file.is_file():
                            print(f"    - {img_file.name}")
        
        if len(patient_dirs) > 10:
            print(f"\n... and {len(patient_dirs) - 10} more patients")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize DMR-IR dataset by patient ID and view"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/local/DMR-IR",
        help="Source dataset root directory (default: data/local/DMR-IR)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/organized_by_patient",
        help="Output root directory (default: data/organized_by_patient)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview of generated structure without copying files"
    )
    
    args = parser.parse_args()
    
    # Make paths absolute if they're relative
    script_dir = Path(__file__).parent
    source_root = Path(args.source) if Path(args.source).is_absolute() else script_dir / args.source
    output_root = Path(args.output) if Path(args.output).is_absolute() else script_dir / args.output
    
    # Verify source exists
    if not source_root.exists():
        print(f"Error: Source directory not found: {source_root}")
        return
    
    # Create organizer
    organizer = DatasetOrganizer(source_root, output_root)
    
    # Load metadata
    organizer.load_metadata()
    
    if not args.preview:
        # Reorganize dataset
        organizer.organize_dataset()
        organizer.print_structure()
    else:
        print("Preview mode: No files will be copied")
        print(f"Would organize from: {source_root}")
        print(f"Would output to: {output_root}")


if __name__ == "__main__":
    main()
