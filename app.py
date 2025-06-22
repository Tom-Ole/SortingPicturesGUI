
import sys
import os
import shutil
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QListWidget, QListWidgetItem, QHBoxLayout, QComboBox, QMessageBox, QAbstractItemView
)
from PyQt6.QtGui import QPixmap, QIcon, QMovie
from PyQt6.QtCore import (Qt, QSize, QThreadPool, QRunnable, pyqtSignal, QObject)
from PyQt6 import QtGui
from PIL import Image, ImageFilter, ImageFile

from dataclasses import dataclass
import numpy as np
from typing import List, Tuple

import json
import io

from imageClassifier import ImageClassifierAi

SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
SORTING_CRITERIAS = [
    "dominant_color",      # index 0
    "brightness",          # index 1 
    "aspect_ratio",        # index 2
    "file_size",           # index 3
    "resolution",          # index 4
    "sharpness",           # index 5
    "color_variance"       # index 6
]

PREV_WIDTH = 300
PREV_HEIGHT = 300

# Sorting algos: Bubble Sort, Merge Sort, Quick Sort, Radix Sort (Brightness, color), Custom
# Sorting criteria: Dominant Color, Brightness, Aspect-Ratio, File Size, Resolution, Sharpness, Color Variance, [AI-Based Sorting]

def LOG(msg, level="INFO"):
    if True:
        if level == "STRONG":
            print(f"============================")
            print(f"[{level}] {msg}")
            print(f"============================")
        elif level == "INFO":
            print(f"[{level}] {msg}")
        elif level == "ERROR":
            print(f"[{level}] {msg}")
        else:
            print(f"[LOG] {msg}")

def compute_dominant_color(img: ImageFile) -> Tuple[int, int, int]:
    LOG("Computing dominant color...")
    img = img.convert("RGB").resize((100, 100)) 
    colors = img.getcolors(maxcolors=10000)  
    if not colors:
        return (0, 0, 0)  
    dominant_color = max(colors, key=lambda x: x[0])[1]  
    return dominant_color

def compute_brightness(img: ImageFile) -> float:
    LOG("Computing brightness...")
    img = img.convert("L")
    stat = img.getextrema()
    return sum(stat) / 2 if stat else 0

def compute_aspect_ratio(img: ImageFile) -> float:
    LOG("Computing aspect ratio...")
    width, height = img.size
    return width / height if height != 0 else 0

def compute_file_size(image_path: str) -> int:
    LOG("Computing file size...")
    return os.path.getsize(image_path)

def compute_resolution(img: ImageFile) -> Tuple[int, int]:
    LOG("Computing resolution...")
    return img.size  # Returns (width, height)

def compute_sharpness(img: ImageFile) -> float:
    LOG("Computing sharpness...")
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    stat = edges.getextrema()
    # Mean edge intensity as sharpness proxy
    sharpness = sum(stat) / 2 if stat else 0
    return sharpness

def compute_color_variance(img: ImageFile) -> Tuple[float, float, float]:
    LOG("Computing color variance...")
    # Resize to speed up variance calculation
    img = img.resize((100, 100)).convert("RGB")
    pixels = np.array(img)  # shape (100, 100, 3)

    # Calculate variance per channel (R, G, B)
    r_var = np.var(pixels[:, :, 0])
    g_var = np.var(pixels[:, :, 1])
    b_var = np.var(pixels[:, :, 2])

    return (r_var, g_var, b_var)

def compute_sorting_algos(image_path: str) -> 'ImageLabel':
    LOG(f"Computing sorting algorithms for {image_path}...")
    img = Image.open(image_path)
    resized_img = img.resize((100, 100))

    return ImageLabel(
        dominant_color=compute_dominant_color(resized_img),
        brightness=compute_brightness(img),
        aspect_ratio=compute_aspect_ratio(img),
        file_size=compute_file_size(image_path),
        resolution=compute_resolution(img),
        sharpness=compute_sharpness(img),
        color_variance=compute_color_variance(img)
    )

@dataclass
class ImageLabel:
    dominant_color: Tuple[int, int, int]
    brightness: float
    aspect_ratio: float
    file_size: int
    resolution: Tuple[int, int]
    sharpness: float
    color_variance: Tuple[float, float, float]
    tags: List[str] = None  # Optional tags for AI classification

class WorkerSignals(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(str, ImageLabel)


class ImageLoadWorker(QRunnable):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.signals = WorkerSignals()

    def run(self):
        label = compute_sorting_algos(self.path)
        self.signals.result.emit(self.path, label)
        self.signals.finished.emit()

class ImageSorterApp(QWidget):
    def __init__(self):
        super().__init__()

        self.sorting_criteria_level = 0
        self.image_classifer = ImageClassifierAi(model_path="./models/model.pth")

        self.image_folder_location = ""
        
        self.setWindowTitle("Image Sorter")
        self.resize(1280, 720)

        self.image_list: List[Tuple[str, ImageLabel]] = []  # (path, label)
        self.selected_items = set()
        self.threadpool = QThreadPool()

        layout = QVBoxLayout()
        controls = QHBoxLayout()

        self.load_button = QPushButton("Select Folder")
        self.load_button.clicked.connect(self.select_folder)
        controls.addWidget(self.load_button)

        self.sort_criteria_combo = QComboBox()
        self.sort_criteria_combo.addItems([
            "Dominant Color",  # index 0
            "Brightness",      # index 1
            "Aspect Ratio",    # index 2
            "File Size",       # index 3
            "Resolution",      # index 4
            "Sharpness",       # index 5
            "Color Variance"   # index 6
        ])
        controls.addWidget(self.sort_criteria_combo)
        self.sort_criteria_combo.setCurrentIndex(0)
        self.sort_criteria_combo.currentIndexChanged.connect(self.set_sorting_criteria)
        self.sort_criteria_combo.setToolTip("Select sorting criteria")
        self.sort_criteria_combo.setWhatsThis(
            "Select the criteria by which to sort the images. "
            "Options include Dominant Color, Brightness, Aspect Ratio, "
            "File Size, Resolution, Sharpness, and Color Variance."
        )
        self.sort_criteria_combo.setStatusTip(
            "Choose sorting criteria for the images. "
            "This will determine how the images are ordered in the list."
        )


        self.sort_button = QPushButton("Sort Images")
        self.sort_button.clicked.connect(self.sort_images)
        controls.addWidget(self.sort_button)

        self.create_image_tags_button = QPushButton("Create Image Tags")
        self.create_image_tags_button.clicked.connect(self.create_image_tags)
        controls.addWidget(self.create_image_tags_button)

        self.save_image_tags_button = QPushButton("Save Image Labels")
        self.save_image_tags_button.clicked.connect(self.save_image_labels)
        controls.addWidget(self.save_image_tags_button)

        self.export_button = QPushButton("Export Images")
        self.export_button.clicked.connect(self.export_selected)
        controls.addWidget(self.export_button)

        layout.addLayout(controls)

        self.spinner_label = QLabel()
        self.spinner = QMovie("spinner.gif")
        self.spinner_label.setMovie(self.spinner)
        self.spinner_label.setVisible(False)
        self.spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinner_label.setScaledContents(True)
        self.spinner_label.setFixedSize(QSize(100, 100))
        layout.addWidget(self.spinner_label)

        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(PREV_WIDTH, PREV_HEIGHT))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        layout.addWidget(self.list_widget)

        self.setLayout(layout)

    def disable_ui(self, disable: bool):
        self.load_button.setDisabled(disable)
        self.sort_button.setDisabled(disable)
        self.export_button.setDisabled(disable)
        self.sort_criteria_combo.setDisabled(disable)

    def set_sorting_criteria(self, index):
        # LOG(f"Setting sorting criteria to index {index}")
        if index < 0 or index >= len(self.sort_criteria_combo):
            QMessageBox.warning(self, "Invalid Selection", "Please select a valid sorting criteria.")
            return
        
        self.sorting_criteria_level = index

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        
        self.image_folder_location = folder
        
        # check if "image_labels.json" exists in the folder
        labels_file = os.path.join(folder, "image_labels.json")
        if os.path.exists(labels_file):
            # load the labels from the file
            with io.open(labels_file, 'r', encoding="utf8") as f:
                labels_data = json.load(f)
                self.image_list.clear()
                self.list_widget.clear()

                images = labels_data.get("images", {})

                for image_name, labels  in images.items():
                    full_path = os.path.join(folder, image_name)
                    label = ImageLabel(
                        dominant_color=tuple(labels.get("dominant_color")),
                        brightness=labels.get("brightness"),
                        aspect_ratio=labels.get("aspect_ratio"),
                        file_size=labels.get("file_size"),
                        resolution=tuple(labels.get("resolution")),
                        sharpness=labels.get("sharpness"),
                        color_variance=tuple(labels.get("color_variance")),
                        tags=labels.get("tags")
                    )
                    self.image_list.append((full_path, label))
                LOG(f"Loaded {len(self.image_list)} images from {labels_file}")
                f.close()
            
            for path, label in self.image_list:
                pixmap = QPixmap(path)
                icon = self.create_icon(pixmap)
                item = QListWidgetItem(icon, os.path.basename(path))
                item.setData(Qt.ItemDataRole.UserRole, path)
                self.list_widget.addItem(item)
                        

                
            return

        

        self.image_list.clear()
        self.list_widget.clear()
        self.spinner_label.setVisible(True)
        self.spinner.start()
        self.disable_ui(True)
        self.pending_tasks = 0

        def on_result(path, label):
            self.image_list.append((path, label))
            pixmap = QPixmap(path)
            icon = self.create_icon(pixmap)
            item = QListWidgetItem(icon, os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.list_widget.addItem(item)

        def on_finished():
            self.pending_tasks -= 1
            if self.pending_tasks == 0:
                self.spinner.stop()
                self.spinner_label.setVisible(False)
                self.disable_ui(False)
                LOG("All images loaded", "STRONG")

        files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED_FORMATS)
        ]

         # Multithreaded execution
        for full_path in files:
            worker = ImageLoadWorker(full_path)
            worker.signals.result.connect(on_result)
            worker.signals.finished.connect(on_finished)
            self.threadpool.start(worker)
            self.pending_tasks += 1
    

    
    def create_icon(self, pixmap: QPixmap) -> QIcon:
        canvas = QPixmap(PREV_WIDTH, PREV_HEIGHT)
        canvas.fill(Qt.GlobalColor.transparent)

        # Scale image keeping aspect ratio
        scaled = pixmap.scaled(
            PREV_WIDTH, PREV_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        painter = QPixmap(canvas)
        painter.fill(Qt.GlobalColor.transparent)
        painter_painter = QtGui.QPainter(painter)
        x = (PREV_WIDTH - scaled.width()) // 2
        y = (PREV_HEIGHT - scaled.height()) // 2
        painter_painter.drawPixmap(x, y, scaled)
        painter_painter.end()

        return QIcon(painter)


    def sort_images(self):
        sorting_criteria = SORTING_CRITERIAS[self.sorting_criteria_level] 
        LOG("criteria_index: " + str(self.sorting_criteria_level) + " => " + sorting_criteria, "STRONG")


        
        def get_sort_key(image_label: ImageLabel):
            if sorting_criteria == "dominant_color":
                # Convert RGB tuple to luminance for sortings
                r, g, b = image_label.dominant_color
                return 0.299 * r + 0.587 * g + 0.114 * b
            elif sorting_criteria == "resolution":
                width, height = image_label.resolution
                return width * height  # Total pixel count
            elif sorting_criteria == "color_variance":
                return sum(image_label.color_variance)  # Aggregate variance
            else:
                return getattr(image_label, sorting_criteria)


        self.image_list.sort(key=lambda x: get_sort_key(x[1]))
        self.list_widget.clear()

        for path, _ in self.image_list:
            pixmap = QPixmap(path)
            icon = self.create_icon(pixmap)
            item = QListWidgetItem(icon, os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.list_widget.addItem(item)
    
    def export_selected(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select images to export.")
            return

        target_folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not target_folder:
            return
        for item in selected_items:
            src_path = item.data(Qt.ItemDataRole.UserRole)
            dst_path = os.path.join(target_folder, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)

        QMessageBox.information(self, "Export Complete", f"Exported {len(selected_items)} images.")

    def create_image_tags(self):
        if not self.image_list:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        self.spinner_label.setVisible(True)
        self.spinner.start()
        self.disable_ui(True)

        # linear execution
        for path, label in self.image_list:
            try:
                img = Image.open(path)
                tags = self.image_classifer.classify_image(img)
                # save tags to label
                label.tags = tags
                LOG(f"Image: {os.path.basename(path)} - Tags: {tags}")
            except Exception as e:
                LOG(f"Error processing image {path}: {e}", "ERROR")
        
        self.spinner.stop()
        self.spinner_label.setVisible(False)
        self.disable_ui(False)


        LOG(f"Length of image list: {self.image_classifer.load_model()} {self.image_classifer.classify_image("NOPE")}")
        
    def save_image_labels(self):
        if not self.image_list:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        # save the file to the path: self.image_folder_location
        save_path = self.image_folder_location
        if not save_path:
            QMessageBox.warning(self, "No Folder", "Please select a folder first.")
            return
        labels_file = os.path.join(save_path, "image_labels.json")

        '''
        Format: 
        "path": "path/to/images",
        "images": [
            "path": {
                "dominant_color": (r, g, b),
                "brightness": float,
                "aspect_ratio": float,
                "file_size": int,
                "resolution": (width, height),
                "sharpness": float,
                "color_variance": (r_var, g_var, b_var),
                "tags": [list of tags]
            },
            "path2": {
                ...
            }
        ]
        '''

        with io.open(labels_file, 'w', encoding="utf8") as f:
            labels_data = {
                "path": self.image_folder_location,
                "images": {
                    os.path.basename(path): {
                        "dominant_color": label.dominant_color,
                        "brightness": label.brightness,
                        "aspect_ratio": label.aspect_ratio,
                        "file_size": label.file_size,
                        "resolution": label.resolution,
                        "sharpness": label.sharpness,
                        "color_variance": label.color_variance,
                        "tags": label.tags or []
                    }
                    for path, label in self.image_list
                }
            }
            json.dump(labels_data, f, indent=4, ensure_ascii=False, sort_keys=True)
    
        QMessageBox.information(self, "Save Complete", f"Image labels saved to {labels_file}")


    
        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageSorterApp()
    window.show()
    sys.exit(app.exec())