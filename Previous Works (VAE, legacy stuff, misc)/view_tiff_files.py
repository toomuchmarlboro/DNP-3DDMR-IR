"""
Interactive TIFF Viewer for DMR-IR Dataset
Displays thermal images with filtering by view angle and category.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import numpy as np
import pandas as pd


class TIFFViewer:
    def __init__(self):
        self.data_root = Path(__file__).parent / "data" / "local" / "DMR-IR"
        self.metadata_dict = {}  # Maps image_path to metadata row
        self.image_data = []    # List of (image_path, split, category, patient_id, view_name)
        self.current_index = 0
        
        # Available views
        self.views = ["All Views", "Frontal", "Left 45°", "Right 45°", "Left 90°", "Right 90°"]
        self.current_view_filter = "All Views"
        self.current_category_filter = "All"
        self.categories = ["All", "benign", "malignant"]
        self.filtered_indices = []
        
        # Load metadata and images
        self._load_metadata()
        self._load_image_list()
        self._apply_filters()
        
        if not self.filtered_indices:
            print("No TIFF files found matching criteria.")
            return
        
        print(f"Found {len(self.image_data)} total TIFF files")
        print(f"Displaying {len(self.filtered_indices)} files")
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(14, 9))
        self.ax_img = plt.subplot(111)
        plt.subplots_adjust(bottom=0.2)
        
        # Add navigation buttons
        ax_prev = plt.axes([0.2, 0.05, 0.08, 0.075])
        ax_next = plt.axes([0.72, 0.05, 0.08, 0.075])
        ax_view_prev = plt.axes([0.2, 0.12, 0.08, 0.04])
        ax_view_next = plt.axes([0.72, 0.12, 0.08, 0.04])
        ax_cat_prev = plt.axes([0.2, 0.17, 0.08, 0.04])
        ax_cat_next = plt.axes([0.72, 0.17, 0.08, 0.04])
        
        self.btn_prev = Button(ax_prev, 'Previous Image')
        self.btn_next = Button(ax_next, 'Next Image')
        self.btn_view_prev = Button(ax_view_prev, '< View')
        self.btn_view_next = Button(ax_view_next, 'View >')
        self.btn_cat_prev = Button(ax_cat_prev, '< Category')
        self.btn_cat_next = Button(ax_cat_next, 'Category >')
        
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_view_prev.on_clicked(self._on_view_prev)
        self.btn_view_next.on_clicked(self._on_view_next)
        self.btn_cat_prev.on_clicked(self._on_cat_prev)
        self.btn_cat_next.on_clicked(self._on_cat_next)
        
        # Display first image
        self._display_current_image()
        
        plt.show()
    
    def _load_metadata(self):
        """Load metadata from CSV files."""
        for split in ["train", "test", "validation"]:
            meta_path = self.data_root / split / "metadata.csv"
            if meta_path.exists():
                df = pd.read_csv(meta_path)
                for _, row in df.iterrows():
                    img_relative_path = row['image_path']
                    self.metadata_dict[img_relative_path] = row
    
    def _load_image_list(self):
        """Collect TIFF files with metadata information."""
        for split in ["train", "test", "validation"]:
            split_path = self.data_root / split / "images"
            if split_path.exists():
                for category in ["benign", "malignant"]:
                    category_path = split_path / category
                    if category_path.exists():
                        tiff_files = sorted(list(category_path.glob("*.tif")) + list(category_path.glob("*.tiff")))
                        for img_path in tiff_files:
                            relative_path = img_path.relative_to(self.data_root)
                            img_relative_str = str(relative_path).replace("\\", "/")
                            
                            # Get metadata
                            if img_relative_str in self.metadata_dict:
                                row = self.metadata_dict[img_relative_str]
                                patient_id = str(row['patient_id'])
                                view_name = str(row['view_name'])
                            else:
                                patient_id = "Unknown"
                                view_name = "Unknown"
                            
                            self.image_data.append((img_path, split, category, patient_id, view_name))
    
    def _apply_filters(self):
        """Filter image list based on current filters."""
        self.filtered_indices = []
        for i, (_, split, category, patient_id, view_name) in enumerate(self.image_data):
            if self.current_category_filter != "All" and category != self.current_category_filter:
                continue
            if self.current_view_filter != "All Views" and view_name != self.current_view_filter:
                continue
            self.filtered_indices.append(i)
        
        # Reset index if it's out of bounds
        if self.current_index >= len(self.filtered_indices):
            self.current_index = 0
    
    def _display_current_image(self):
        """Display the current image with metadata."""
        if not self.filtered_indices:
            self.ax_img.clear()
            self.ax_img.text(0.5, 0.5, "No images match the current filters", 
                           ha='center', va='center', transform=self.ax_img.transAxes)
            self.ax_img.axis('off')
            self.fig.canvas.draw()
            return
        
        idx = self.filtered_indices[self.current_index]
        img_path, split, category, patient_id, view_name = self.image_data[idx]
        
        try:
            # Load and display image
            img = Image.open(img_path)
            img_array = np.array(img)
            
            self.ax_img.clear()
            
            # Display with appropriate colormap for thermal images
            if len(img_array.shape) == 2:
                self.ax_img.imshow(img_array, cmap='hot')
            else:
                self.ax_img.imshow(img_array)
            
            # Set title with information
            title = f"[{self.current_index + 1}/{len(self.filtered_indices)}] {img_path.name}\n"
            title += f"Patient: {patient_id} | View: {view_name} | Category: {category}\n"
            title += f"Split: {split} | Shape: {img_array.shape}\n"
            title += f"Filter: {self.current_view_filter} | Category Filter: {self.current_category_filter}"
            
            self.ax_img.set_title(title, fontsize=10)
            self.ax_img.axis('off')
            
            self.fig.canvas.draw()
            
        except Exception as e:
            self.ax_img.clear()
            self.ax_img.text(0.5, 0.5, f"Error loading image:\n{str(e)}", 
                           ha='center', va='center', transform=self.ax_img.transAxes)
            self.ax_img.axis('off')
            self.fig.canvas.draw()
    
    def _on_prev(self, event):
        """Navigate to previous image."""
        self.current_index = (self.current_index - 1) % len(self.filtered_indices)
        self._display_current_image()
    
    def _on_next(self, event):
        """Navigate to next image."""
        self.current_index = (self.current_index + 1) % len(self.filtered_indices)
        self._display_current_image()
    
    def _on_view_prev(self, event):
        """Previous view filter."""
        idx = self.views.index(self.current_view_filter)
        self.current_view_filter = self.views[(idx - 1) % len(self.views)]
        self.current_index = 0
        self._apply_filters()
        self._display_current_image()
    
    def _on_view_next(self, event):
        """Next view filter."""
        idx = self.views.index(self.current_view_filter)
        self.current_view_filter = self.views[(idx + 1) % len(self.views)]
        self.current_index = 0
        self._apply_filters()
        self._display_current_image()
    
    def _on_cat_prev(self, event):
        """Previous category filter."""
        idx = self.categories.index(self.current_category_filter)
        self.current_category_filter = self.categories[(idx - 1) % len(self.categories)]
        self.current_index = 0
        self._apply_filters()
        self._display_current_image()
    
    def _on_cat_next(self, event):
        """Next category filter."""
        idx = self.categories.index(self.current_category_filter)
        self.current_category_filter = self.categories[(idx + 1) % len(self.categories)]
        self.current_index = 0
        self._apply_filters()
        self._display_current_image()


if __name__ == "__main__":
    try:
        viewer = TIFFViewer()
    except Exception as e:
        print(f"Error starting viewer: {e}")


if __name__ == "__main__":
    try:
        viewer = TIFFViewer()
    except Exception as e:
        print(f"Error starting viewer: {e}")
