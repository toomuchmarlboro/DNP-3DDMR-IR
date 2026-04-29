import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import numpy as np
import pandas as pd
from collections import defaultdict


class PatientDataViewer:
    def __init__(self):
        self.data_root = Path(__file__).parent / "data" / "local" / "DMR-IR"
        self.metadata_dict = {}
        self.patients_data = defaultdict(lambda: defaultdict(list))  # patient_id -> category -> [(split, view_name, img_path, mask_path, image_shape), ...]
        self.sorted_patient_ids = []
        self.current_patient_idx = 0
        self.current_category_idx = 0
        self.current_view_idx = 0
        self.current_sample_idx = 0
        self.current_category_options = []
        self.current_view_options = []
        self.current_samples = []
        self.show_mode = "overlay"  # "image", "mask", "both", or "overlay"
        
        # Load data
        self._load_metadata()
        self._organize_by_patient()
        
        if not self.sorted_patient_ids:
            print("No patient data found.")
            return
        
        print(f"Found {len(self.sorted_patient_ids)} patients with {sum(len(self.patients_data[p]) for p in self.sorted_patient_ids)} categories")
        
        # Create figure
        self.fig = plt.figure(figsize=(16, 10))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.15)
        
        # Create subplots for image and mask
        self.ax_img = plt.subplot(1, 2, 1)
        self.ax_mask = plt.subplot(1, 2, 2)
        
        # Navigation buttons
        # Patient navigation
        ax_pat_prev = plt.axes([0.05, 0.08, 0.06, 0.04])
        ax_pat_next = plt.axes([0.89, 0.08, 0.06, 0.04])
        
        # Category navigation
        ax_cat_prev = plt.axes([0.05, 0.03, 0.06, 0.04])
        ax_cat_next = plt.axes([0.89, 0.03, 0.06, 0.04])
        
        # View navigation
        ax_view_prev = plt.axes([0.35, 0.05, 0.05, 0.04])
        ax_view_next = plt.axes([0.6, 0.05, 0.05, 0.04])
        ax_sample_prev = plt.axes([0.43, 0.09, 0.05, 0.035])
        ax_sample_next = plt.axes([0.52, 0.09, 0.05, 0.035])
        
        # Display mode buttons
        ax_show_img = plt.axes([0.35, 0.01, 0.05, 0.03])
        ax_show_mask = plt.axes([0.42, 0.01, 0.05, 0.03])
        ax_show_both = plt.axes([0.49, 0.01, 0.05, 0.03])
        ax_show_overlay = plt.axes([0.56, 0.01, 0.06, 0.03])
        
        self.btn_pat_prev = Button(ax_pat_prev, '< Patient', color='lightblue')
        self.btn_pat_next = Button(ax_pat_next, 'Patient >', color='lightblue')
        self.btn_cat_prev = Button(ax_cat_prev, '< Category', color='lightgreen')
        self.btn_cat_next = Button(ax_cat_next, 'Category >', color='lightgreen')
        self.btn_view_prev = Button(ax_view_prev, '< View', color='lightyellow')
        self.btn_view_next = Button(ax_view_next, 'View >', color='lightyellow')
        self.btn_sample_prev = Button(ax_sample_prev, '< Sample', color='wheat')
        self.btn_sample_next = Button(ax_sample_next, 'Sample >', color='wheat')
        
        self.btn_show_img = Button(ax_show_img, 'Image', color='lightcoral')
        self.btn_show_mask = Button(ax_show_mask, 'Mask', color='lightcoral')
        self.btn_show_both = Button(ax_show_both, 'Both', color='lightcoral')
        self.btn_show_overlay = Button(ax_show_overlay, 'Overlay', color='lightcoral')
        
        self.btn_pat_prev.on_clicked(self._on_patient_prev)
        self.btn_pat_next.on_clicked(self._on_patient_next)
        self.btn_cat_prev.on_clicked(self._on_category_prev)
        self.btn_cat_next.on_clicked(self._on_category_next)
        self.btn_view_prev.on_clicked(self._on_view_prev)
        self.btn_view_next.on_clicked(self._on_view_next)
        self.btn_sample_prev.on_clicked(self._on_sample_prev)
        self.btn_sample_next.on_clicked(self._on_sample_next)
        
        self.btn_show_img.on_clicked(lambda e: self._set_show_mode("image"))
        self.btn_show_mask.on_clicked(lambda e: self._set_show_mode("mask"))
        self.btn_show_both.on_clicked(lambda e: self._set_show_mode("both"))
        self.btn_show_overlay.on_clicked(lambda e: self._set_show_mode("overlay"))
        
        # Display first patient
        self._update_category_options()
        self._display_current_data()
        
        plt.show()
    
    def _load_metadata(self):
        """Load metadata from all CSV files."""
        print("Loading metadata...")
        for split in ["train", "test", "validation"]:
            meta_path = self.data_root / split / "metadata.csv"
            if meta_path.exists():
                df = pd.read_csv(meta_path)
                for _, row in df.iterrows():
                    img_relative_path = row['image_path']
                    self.metadata_dict[img_relative_path] = row
    
    def _organize_by_patient(self):
        """Organize images and masks by patient ID, category, and view."""
        print("Organizing data by patient...")
        
        for split in ["train", "test", "validation"]:
            split_path = self.data_root / split
            images_path = split_path / "images"
            masks_path = split_path / "masks"
            
            if not images_path.exists():
                continue
            
            # Process each category
            for category in ["benign", "malignant"]:
                cat_images = images_path / category
                cat_masks = masks_path / category if masks_path.exists() else None
                
                if not cat_images.exists():
                    continue
                
                # Get all image files
                img_files = sorted(list(cat_images.glob("*.tif")) + list(cat_images.glob("*.tiff")))
                
                for img_path in img_files:
                    # Get metadata
                    relative_path = img_path.relative_to(self.data_root)
                    img_relative_str = str(relative_path).replace("\\", "/")
                    
                    if img_relative_str not in self.metadata_dict:
                        continue
                    
                    row = self.metadata_dict[img_relative_str]
                    patient_id = str(row['patient_id']).strip()
                    view_name = str(row['view_name']).strip()
                    
                    # Resolve the exact mask path from metadata instead of inferring by stem.
                    mask_path = None
                    mask_relative = str(row.get('mask_path', '')).strip()
                    if mask_relative and mask_relative != 'nan':
                        candidate_mask = self.data_root / mask_relative
                        if candidate_mask.exists():
                            mask_path = candidate_mask
                    
                    image_shape = None
                    try:
                        image_shape = tuple(Image.open(img_path).size[::-1])
                    except Exception:
                        image_shape = None

                    # Store data
                    self.patients_data[patient_id][category].append((split, view_name, img_path, mask_path, image_shape))
        
        # Get sorted patient IDs
        self.sorted_patient_ids = sorted(self.patients_data.keys())
    
    def _update_category_options(self):
        """Update available categories for current patient."""
        if not self.sorted_patient_ids:
            self.current_category_options = []
            return
        
        current_patient = self.sorted_patient_ids[self.current_patient_idx]
        self.current_category_options = sorted(self.patients_data[current_patient].keys())
        
        # Reset category index if needed
        if self.current_category_idx >= len(self.current_category_options):
            self.current_category_idx = 0
        
        self._update_view_options()
    
    def _update_view_options(self):
        """Update available views for current patient and category."""
        if not self.sorted_patient_ids or not self.current_category_options:
            self.current_view_options = []
            return
        
        current_patient = self.sorted_patient_ids[self.current_patient_idx]
        current_category = self.current_category_options[self.current_category_idx]
        
        # Get unique views for this patient-category combination
        items = self.patients_data[current_patient][current_category]
        views_dict = {}  # view_name -> list of items
        for split, view_name, img_path, mask_path, image_shape in items:
            if view_name not in views_dict:
                views_dict[view_name] = []
            views_dict[view_name].append((split, view_name, img_path, mask_path, image_shape))
        
        self.current_view_options = sorted(views_dict.keys())
        
        # Reset view index if needed
        if self.current_view_idx >= len(self.current_view_options):
            self.current_view_idx = 0

        self._update_samples()

    def _update_samples(self):
        """Update the exact image rows for the current patient/category/view."""
        self.current_samples = []
        if not self.sorted_patient_ids or not self.current_category_options or not self.current_view_options:
            return

        current_patient = self.sorted_patient_ids[self.current_patient_idx]
        current_category = self.current_category_options[self.current_category_idx]
        current_view = self.current_view_options[self.current_view_idx]

        items = self.patients_data[current_patient][current_category]
        self.current_samples = [item for item in items if item[1] == current_view]

        if self.current_sample_idx >= len(self.current_samples):
            self.current_sample_idx = 0
    
    def _get_current_data(self):
        """Get current image and mask data."""
        if not self.sorted_patient_ids or not self.current_category_options or not self.current_view_options:
            return None, None, None
        
        current_patient = self.sorted_patient_ids[self.current_patient_idx]
        current_category = self.current_category_options[self.current_category_idx]
        current_view = self.current_view_options[self.current_view_idx]

        if not self.current_samples:
            self._update_samples()

        if not self.current_samples:
            return None, None, None

        split, view_name, img_path, mask_path, image_shape = self.current_samples[self.current_sample_idx]
        return img_path, mask_path, (current_patient, current_category, current_view, split, image_shape)
    
    def _display_current_data(self):
        """Display current image and mask."""
        img_path, mask_path, info = self._get_current_data()
        
        if img_path is None:
            self._clear_display("No data available")
            return
        
        patient_id, category, view_name, split, image_shape = info
        
        try:
            # Load and display image
            if self.show_mode in ["image", "both", "overlay"]:
                self._display_image(img_path, patient_id, category, view_name, split)
            else:
                self.ax_img.clear()
                self.ax_img.axis('off')
            
            # Load and display mask
            if self.show_mode in ["mask", "both"]:
                self._display_mask(mask_path, patient_id, category, view_name, split)
            elif self.show_mode == "overlay":
                self._display_mask(mask_path, patient_id, category, view_name, split, overlay_on_image=True)
            else:
                self.ax_mask.clear()
                self.ax_mask.axis('off')
            
            self.fig.canvas.draw()
            
        except Exception as e:
            self._clear_display(f"Error loading data:\n{str(e)}")
    
    def _display_image(self, img_path, patient_id, category, view_name, split):
        """Display image on left subplot."""
        img = Image.open(img_path)
        img_array = np.array(img)
        
        self.ax_img.clear()
        
        if len(img_array.shape) == 2:
            self.ax_img.imshow(img_array, cmap='hot')
        else:
            self.ax_img.imshow(img_array)
        
        title = f"Image - {img_path.name}\n"
        title += f"Patient: {patient_id} | View: {view_name}\n"
        title += f"Category: {category} | Split: {split}\n"
        title += f"Shape: {img_array.shape}"
        if self.current_samples:
            title += f" | Sample: {self.current_sample_idx + 1}/{len(self.current_samples)}"
        
        self.ax_img.set_title(title, fontsize=9, fontweight='bold')
        self.ax_img.axis('off')
    
    def _normalize_mask_array(self, mask_array):
        """Convert common stored mask shapes into a displayable array."""
        mask_array = np.asarray(mask_array)

        if mask_array.ndim == 3 and mask_array.shape[0] == 1:
            mask_array = mask_array[0]
        elif mask_array.ndim == 3 and mask_array.shape[-1] == 1:
            mask_array = mask_array[..., 0]
        elif mask_array.ndim > 3:
            mask_array = np.squeeze(mask_array)
            if mask_array.ndim == 3 and mask_array.shape[0] == 1:
                mask_array = mask_array[0]

        return np.asarray(mask_array)

    def _load_mask_array(self, mask_path):
        """Load and normalize a mask from disk."""
        if mask_path is None:
            return None

        mask_array = np.load(mask_path)
        return self._normalize_mask_array(mask_array)

    def _display_mask(self, mask_path, patient_id, category, view_name, split, overlay_on_image=False):
        """Display mask on right subplot."""
        self.ax_mask.clear()
        
        if mask_path is None:
            self.ax_mask.text(0.5, 0.5, "No mask available", 
                            ha='center', va='center', transform=self.ax_mask.transAxes,
                            fontsize=12)
            self.ax_mask.set_title("Mask - Not Available", fontsize=9, fontweight='bold')
            self.ax_mask.axis('off')
            return
        
        try:
            mask_data = self._load_mask_array(mask_path)
            if mask_data is None:
                raise ValueError("Mask path is missing")

            img_path, _, _ = self._get_current_data()
            img_array = np.array(Image.open(img_path))

            original_mask_shape = mask_data.shape
            if overlay_on_image and mask_data.ndim == 2 and img_array.ndim >= 2:
                target_height, target_width = img_array.shape[:2]
                if mask_data.shape != (target_height, target_width):
                    resized = Image.fromarray(mask_data.astype(np.float32) if np.issubdtype(mask_data.dtype, np.floating) else mask_data)
                    mask_data = np.array(resized.resize((target_width, target_height), resample=Image.NEAREST))
            
            if overlay_on_image:
                if img_array.ndim == 2:
                    self.ax_img.imshow(img_array, cmap='hot')
                else:
                    self.ax_img.imshow(img_array)

                overlay_mask = mask_data > 0
                self.ax_img.imshow(overlay_mask, cmap='Reds', alpha=0.35, interpolation='nearest')
                self.ax_img.set_title(f"Overlay - {patient_id} | {category} | {view_name}", fontsize=9, fontweight='bold')
                self.ax_img.axis('off')

            if mask_data.ndim == 2:
                self.ax_mask.imshow(mask_data, cmap='gray', interpolation='nearest')
            elif mask_data.ndim == 3 and mask_data.shape[-1] in [3, 4]:
                self.ax_mask.imshow(mask_data)
            else:
                raise ValueError(f"Unsupported mask shape after normalization: {mask_data.shape}")
            
            title = f"Mask - {mask_path.name}\n"
            title += f"Patient: {patient_id} | View: {view_name}\n"
            title += f"Shape: {mask_data.shape} | dtype: {mask_data.dtype}"
            if original_mask_shape != mask_data.shape:
                title += f" | resized from {original_mask_shape}"
            if overlay_on_image:
                title += "\nOverlay mode"
            
            self.ax_mask.set_title(title, fontsize=9, fontweight='bold')
            self.ax_mask.axis('off')
        except Exception as e:
            self.ax_mask.text(0.5, 0.5, f"Error loading mask:\n{str(e)}", 
                            ha='center', va='center', transform=self.ax_mask.transAxes,
                            fontsize=10)
            self.ax_mask.set_title("Mask - Error", fontsize=9, fontweight='bold')
            self.ax_mask.axis('off')
    
    def _clear_display(self, message):
        """Clear both displays with a message."""
        for ax in [self.ax_img, self.ax_mask]:
            ax.clear()
            ax.text(0.5, 0.5, message, ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
        self.fig.canvas.draw()
    
    def _set_show_mode(self, mode):
        """Set what to display: image, mask, or both."""
        self.show_mode = mode
        self._display_current_data()
    
    def _on_patient_prev(self, event):
        """Navigate to previous patient."""
        self.current_patient_idx = (self.current_patient_idx - 1) % len(self.sorted_patient_ids)
        self.current_category_idx = 0
        self.current_view_idx = 0
        self._update_category_options()
        self._display_current_data()
    
    def _on_patient_next(self, event):
        """Navigate to next patient."""
        self.current_patient_idx = (self.current_patient_idx + 1) % len(self.sorted_patient_ids)
        self.current_category_idx = 0
        self.current_view_idx = 0
        self._update_category_options()
        self._display_current_data()
    
    def _on_category_prev(self, event):
        """Navigate to previous category."""
        if self.current_category_options:
            self.current_category_idx = (self.current_category_idx - 1) % len(self.current_category_options)
            self.current_view_idx = 0
            self._update_view_options()
            self._display_current_data()
    
    def _on_category_next(self, event):
        """Navigate to next category."""
        if self.current_category_options:
            self.current_category_idx = (self.current_category_idx + 1) % len(self.current_category_options)
            self.current_view_idx = 0
            self._update_view_options()
            self._display_current_data()
    
    def _on_view_prev(self, event):
        """Navigate to previous view."""
        if self.current_view_options:
            self.current_view_idx = (self.current_view_idx - 1) % len(self.current_view_options)
            self.current_sample_idx = 0
            self._update_samples()
            self._display_current_data()
    
    def _on_view_next(self, event):
        """Navigate to next view."""
        if self.current_view_options:
            self.current_view_idx = (self.current_view_idx + 1) % len(self.current_view_options)
            self.current_sample_idx = 0
            self._update_samples()
            self._display_current_data()

    def _on_sample_prev(self, event):
        """Navigate to previous sample for the current patient/view."""
        if self.current_samples:
            self.current_sample_idx = (self.current_sample_idx - 1) % len(self.current_samples)
            self._display_current_data()

    def _on_sample_next(self, event):
        """Navigate to next sample for the current patient/view."""
        if self.current_samples:
            self.current_sample_idx = (self.current_sample_idx + 1) % len(self.current_samples)
            self._display_current_data()


if __name__ == "__main__":
    try:
        viewer = PatientDataViewer()
    except Exception as e:
        print(f"Error starting viewer: {e}")
