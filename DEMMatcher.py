from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMessageBox
from qgis.PyQt.QtCore import QCoreApplication
from qgis.utils import iface
import rasterio
import numpy as np
import os

class DEMMatcher:
    def __init__(self, iface):
        """
        Initializes the DEM Matcher plugin.
        :param iface: QGIS interface instance.
        """
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.action = QAction("DEM Matcher", iface.mainWindow())
        self.action.triggered.connect(self.run)
        iface.addPluginToMenu("DEM Tools", self.action)
    
    def unload(self):
        """
        Unloads the plugin from the QGIS menu.
        """
        self.iface.removePluginMenu("DEM Tools", self.action)
    
    def run(self):
        """
        Runs the DEM Matcher plugin by opening file dialogs to select input DEMs and output path.
        """
        file_dialog = QFileDialog()
        dem1_path, _ = file_dialog.getOpenFileName(None, "Select First DEM", "", "GeoTIFF (*.tif)")
        if not dem1_path:
            return
        
        dem2_path, _ = file_dialog.getOpenFileName(None, "Select Second DEM", "", "GeoTIFF (*.tif)")
        if not dem2_path:
            return
        
        output_path, _ = file_dialog.getSaveFileName(None, "Save Merged DEM", "", "GeoTIFF (*.tif)")
        if not output_path:
            return
        
        try:
            dem1, meta1 = self.load_dem(dem1_path)
            dem2, meta2 = self.load_dem(dem2_path)
            
            best_offset = self.find_best_overlap(dem1, dem2)
            
            merged_dem = self.merge_dems(dem1, dem2, best_offset)
            self.save_merged_dem(merged_dem, meta1, output_path)
            
            QMessageBox.information(None, "Success", f"Merged DEM saved at: {output_path}")
            iface.addRasterLayer(output_path, "Merged DEM")
        except Exception as e:
            QMessageBox.critical(None, "Error", str(e))
    
    def load_dem(self, dem_path):
        """
        Loads a DEM as a NumPy array and extracts its metadata.
        :param dem_path: Path to the DEM file.
        :return: Tuple containing the DEM array and its metadata.
        """
        with rasterio.open(dem_path) as src:
            data = src.read(1)  # Load the first band
            meta = src.meta
        return data, meta
    
    def find_best_overlap(self, dem1, dem2):
        """
        Finds the best alignment between two DEMs by shifting one over the other
        and maximizing the overlapping pixels.
        :param dem1: First DEM array.
        :param dem2: Second DEM array.
        :return: Tuple (dx, dy) representing the best shift.
        """
        max_overlap = 0
        best_offset = (0, 0)
        max_shift_x = min(50, dem1.shape[1] // 2)
        max_shift_y = min(50, dem1.shape[0] // 2)

        for dx in range(-max_shift_x, max_shift_x + 1):
            for dy in range(-max_shift_y, max_shift_y + 1):
                overlap = self.compute_overlap(dem1, dem2, dx, dy)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_offset = (dx, dy)
        return best_offset
    
    def compute_overlap(self, dem1, dem2, dx, dy):
        """
        Computes the overlap between two DEMs when shifting one over the other.
        :param dem1: First DEM array.
        :param dem2: Second DEM array.
        :param dx: Shift in x direction.
        :param dy: Shift in y direction.
        :return: Number of overlapping valid pixels.
        """
        h, w = dem1.shape
        x_start = max(0, dx)
        x_end = min(w, w + dx)
        y_start = max(0, dy)
        y_end = min(h, h + dy)
        
        if x_end > x_start and y_end > y_start:
            dem1_crop = dem1[y_start:y_end, x_start:x_end]
            dem2_crop = dem2[y_start - dy:y_end - dy, x_start - dx:x_end - dx]
            valid_pixels = (dem1_crop > 0) & (dem2_crop > 0)
            return np.sum(valid_pixels)
        return 0
    
    def merge_dems(self, dem1, dem2, offset):
        """
        Merges two DEMs by aligning them using the best found shift.
        :param dem1: First DEM array.
        :param dem2: Second DEM array.
        :param offset: Tuple (dx, dy) representing the best alignment shift.
        :return: Merged DEM as a NumPy array.
        """
        dx, dy = offset
        h1, w1 = dem1.shape
        h2, w2 = dem2.shape

        # Determine new dimensions based on shift
        new_h = max(h1, h2 + abs(dy))
        new_w = max(w1, w2 + abs(dx))

        # Initialize merged array
        merged = np.zeros((new_h, new_w), dtype=np.float32)

        # Place DEM1 in merged array
        merged[:h1, :w1] = dem1

        # Determine proper placement of DEM2
        if dy >= 0 and dx >= 0:
            merged[dy:dy+h2, dx:dx+w2] = np.where(dem2 > 0, dem2, merged[dy:dy+h2, dx:dx+w2])
        elif dy < 0 and dx < 0:
            merged[:h2+dy, :w2+dx] = np.where(dem2 > 0, dem2, merged[:h2+dy, :w2+dx])
        elif dy >= 0 and dx < 0:
            merged[dy:dy+h2, :w2+dx] = np.where(dem2 > 0, dem2, merged[dy:dy+h2, :w2+dx])
        elif dy < 0 and dx >= 0:
            merged[:h2+dy, dx:dx+w2] = np.where(dem2 > 0, dem2, merged[:h2+dy, dx:dx+w2])

        return merged

    
    def save_merged_dem(self, merged, meta, output_path):
        """
        Saves the merged DEM as a GeoTIFF file.
        :param merged: Merged DEM array.
        :param meta: Metadata from the original DEM.
        :param output_path: Path to save the merged DEM.
        """
        meta.update({"height": merged.shape[0], "width": merged.shape[1]})
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(merged, 1)

def classFactory(iface):
    """
    Required function for QGIS to initialize the plugin.
    :param iface: QGIS interface instance.
    """
    return DEMMatcher(iface)
