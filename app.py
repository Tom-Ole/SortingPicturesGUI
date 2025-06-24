
import sys
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image, ImageFilter

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QListWidget, QListWidgetItem, QHBoxLayout, QComboBox, 
    QMessageBox, QAbstractItemView, QProgressBar, QSplitter,
    QGroupBox, QCheckBox, QSpinBox, QSlider
)
from PyQt6.QtGui import QPixmap, QIcon, QMovie, QPainter, QImage
from PyQt6.QtCore import (Qt, QSize, QThreadPool, QRunnable, pyqtSignal, 
                          QObject, QTimer, QMutex, pyqtSlot)

from imageClassifier import ImageClassifierAi
from logger import Logger


SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
SORTING_CRITERIA = [
    "dominant_color",
    "brightness", 
    "aspect_ratio",
    "file_size",
    "resolution",
    "sharpness",
    "color_variance"
]

SORTING_DISPLAY_NAMES = [
    "Dominant Color",
    "Brightness",
    "Aspect Ratio", 
    "File Size",
    "Resolution",
    "Sharpness",
    "Color Variance"
]

# UI Constants
PREVIEW_WIDTH = 200
PREVIEW_HEIGHT = 200
THUMBNAIL_SIZE = 150
MAX_WORKERS = min(8, os.cpu_count() or 4)


# Sorting algos: Bubble Sort, Merge Sort, Quick Sort, Radix Sort (Brightness, color), Custom
# Sorting criteria: Dominant Color, Brightness, Aspect-Ratio, File Size, Resolution, Sharpness, Color Variance, [AI-Based Sorting]

logger = Logger()


@dataclass
class ImageMetrics:
    """Enhanced image metrics with validation"""
    dominant_color: Tuple[int, int, int] = (0, 0, 0)
    brightness: float = 0.0
    aspect_ratio: float = 1.0
    file_size: int = 0
    resolution: Tuple[int, int] = (0, 0)
    sharpness: float = 0.0
    color_variance: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageMetrics':
        """Create instance from dictionary"""
        return cls(**data)



class ImageProcessor:
    """Separated image processing logic"""

    @staticmethod
    def compute_dominant_color(img: Image.Image) -> Tuple[int, int, int]:
        """Compute dominant color using quantization for better performance"""
        try:
            # Reduce to smaller size and quantize colors
            img_small = img.convert("RGB").resize((50, 50))
            img_quantized = img_small.quantize(colors=8)
            palette = img_quantized.getpalette()
            
            if not palette:
                return (0, 0, 0)
            
            # Get color counts
            colors = img_quantized.getcolors(maxcolors=8)
            if not colors:
                return (0, 0, 0)
            
            # Find most frequent color
            most_frequent = max(colors, key=lambda x: x[0])
            color_index = most_frequent[1]
            
            # Extract RGB from palette
            r = palette[color_index * 3]
            g = palette[color_index * 3 + 1] 
            b = palette[color_index * 3 + 2]
            
            return (r, g, b)
        except Exception as e:
            logger.error(f"Error computing dominant color: {e}")
            return (0, 0, 0)
    
    @staticmethod
    def compute_brightness(img: Image.Image) -> float:
        """Compute average brightness"""
        try:
            grayscale = img.convert("L").resize((100, 100))
            pixels = list(grayscale.getdata())
            return sum(pixels) / len(pixels) if pixels else 0.0
        except Exception as e:
            logger.error(f"Error computing brightness: {e}")
            return 0.0
    
    @staticmethod
    def compute_aspect_ratio(img: Image.Image) -> float:
        """Compute width/height ratio"""
        width, height = img.size
        return width / height if height != 0 else 1.0
    
    @staticmethod
    def compute_file_size(image_path: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(image_path)
        except Exception as e:
            logger.error(f"Error getting file size for {image_path}: {e}")
            return 0
    
    @staticmethod
    def compute_resolution(img: Image.Image) -> Tuple[int, int]:
        """Get image resolution"""
        return img.size
    
    @staticmethod
    def compute_sharpness(img: Image.Image) -> float:
        """Compute sharpness using Laplacian variance"""
        try:
            # Resize for performance
            img_small = img.convert("L").resize((200, 200))
            edges = img_small.filter(ImageFilter.FIND_EDGES)
            pixels = list(edges.getdata())
            
            if not pixels:
                return 0.0
            
            # Calculate variance as sharpness measure
            mean_val = sum(pixels) / len(pixels)
            variance = sum((p - mean_val) ** 2 for p in pixels) / len(pixels)
            return variance
        except Exception as e:
            logger.error(f"Error computing sharpness: {e}")
            return 0.0
    
    @staticmethod
    def compute_color_variance(img: Image.Image) -> Tuple[float, float, float]:
        """Compute color variance per channel"""
        try:
            img_small = img.resize((100, 100)).convert("RGB")
            pixels = np.array(img_small)
            
            if pixels.size == 0:
                return (0.0, 0.0, 0.0)
            
            r_var = float(np.var(pixels[:, :, 0]))
            g_var = float(np.var(pixels[:, :, 1]))
            b_var = float(np.var(pixels[:, :, 2]))
            
            return (r_var, g_var, b_var)
        except Exception as e:
            logger.error(f"Error computing color variance: {e}")
            return (0.0, 0.0, 0.0)
    
    @classmethod
    def analyze_image(cls, image_path: str) -> ImageMetrics:
        """Analyze image and return all metrics"""
        try:
            with Image.open(image_path) as img:
                return ImageMetrics(
                    dominant_color=cls.compute_dominant_color(img),
                    brightness=cls.compute_brightness(img),
                    aspect_ratio=cls.compute_aspect_ratio(img),
                    file_size=cls.compute_file_size(image_path),
                    resolution=cls.compute_resolution(img),
                    sharpness=cls.compute_sharpness(img),
                    color_variance=cls.compute_color_variance(img)
                )
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return ImageMetrics()
        
class WorkerSignals(QObject):
    """Enhanced worker signals with progress tracking"""
    finished = pyqtSignal()                 # signal emitted when the worker is done
    result = pyqtSignal(str, ImageMetrics)  # path, metrics
    progress = pyqtSignal(int)              # progress percentage
    error = pyqtSignal(str, str)            # path, error_message

    sorted_result = pyqtSignal(list)  # sorted list of images
    thumb_ready = pyqtSignal(str, QIcon) 


class ImageAnalysisWorker(QRunnable):
    """Worker for analyzing images in a separate thread"""

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.signals = WorkerSignals()
        self.is_cancelled = False

    @pyqtSlot()
    def run(self):
        try:
            if self.is_cancelled:
                return
            
            metrics = ImageProcessor.analyze_image(self.path)
            thumb = QImage(self.path)
            thumb  = thumb.scaled(PREVIEW_WIDTH, PREVIEW_HEIGHT,
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
            self.signals.result.emit(self.path, metrics)
            self.signals.thumb_ready.emit(self.path, QIcon(QPixmap.fromImage(thumb)))

        except Exception as e:
            error_msg = f"Failed to process {self.path}: {str(e)}"
            logger.error(error_msg)
            self.signals.error.emit(self.path, str(e))
        finally:
            self.signals.finished.emit()

    def cancel(self):
        self.is_cancelled = True

class SortingWorker(QRunnable):
    """Worker for sorting images in background thread"""
    
    def __init__(self, image_list: List[Tuple[str, ImageMetrics]], criteria_index: int, reverse: bool):
        super().__init__()
        self.image_list = image_list.copy()  # Make a copy to avoid thread issues
        self.criteria_index = criteria_index
        self.reverse = reverse
        self.signals = WorkerSignals()
        self.is_cancelled = False
    
    @pyqtSlot()
    def run(self):
        try:
            if self.is_cancelled:
                return
            
            criteria = SORTING_CRITERIA[self.criteria_index]
            
            def get_sort_key(item: Tuple[str, ImageMetrics]) -> Any:
                _, metrics = item
                
                if criteria == "dominant_color":
                    # Sort by luminance of dominant color
                    r, g, b = metrics.dominant_color
                    return 0.299 * r + 0.587 * g + 0.114 * b
                elif criteria == "resolution":
                    width, height = metrics.resolution
                    return width * height
                elif criteria == "color_variance":
                    return sum(metrics.color_variance)
                else:
                    return getattr(metrics, criteria, 0)
            
            # Perform the sort
            sorted_list = sorted(self.image_list, key=get_sort_key, reverse=self.reverse)
            
            if not self.is_cancelled:
                self.signals.sorted_result.emit(sorted_list)
                
        except Exception as e:
            logger.error(f"Error in sorting worker: {e}")
            self.signals.error.emit("sorting", str(e))
    
    def cancel(self):
        self.is_cancelled = True

class ThumbnailLoaderWorker(QRunnable):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            pixmap = QPixmap(self.path)
            if pixmap.isNull():
                pixmap = QPixmap(PREVIEW_WIDTH, PREVIEW_HEIGHT)
                pixmap.fill(Qt.GlobalColor.lightGray)

            icon = QIcon(pixmap.scaled(
                PREVIEW_WIDTH, PREVIEW_HEIGHT,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

            self.signals.thumb_ready.emit(self.path, icon)
        except Exception as e:
            logger.error(f"Thumbnail loading failed for {self.path}: {e}")


class ThumbnailCache:
    """Simple thumbnail cache to improve performance"""

    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, QIcon] = {}
        self.max_size = max_size
        self.access_order: List[str] = []

    def get(self, path: str) -> Optional[QIcon]:
        if path in self.cache:
            self.access_order.remove(path)
            self.access_order.append(path)
            return self.cache[path]
        return None
    
    def put(self, path: str, icon: QIcon):
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[path] = icon
        self.access_order.append(path)

    def clear(self):
        self.cache.clear()
        self.access_order.clear()

class ImageSorterApp(QWidget):

    def __init__(self):
        super().__init__()

        self.image_list: List[Tuple[str, ImageMetrics]] = []
        self.image_folder_location = ""
        self.sorting_criteria_index = 0
        self.reverse_sort = False

        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(MAX_WORKERS)
        self.workers: List[ImageAnalysisWorker] = []
        self.pending_tasks = 0
        self.mutex = QMutex()

        self.item_lookup: Dict[str, QListWidgetItem] = {}

        self.thumbnail_cache = ThumbnailCache()

        self.image_classifier = ImageClassifierAi("path/to/your/model")

        self.setup_ui()
        self.setup_logger()

    def setup_ui(self):
        """Initialize the UI components"""
        self.setWindowTitle("Image Sorter")
        self.setMinimumSize(800, 600)
        self.resize(1400, 800)

        main_layout = QHBoxLayout()

        left_panel = self.create_control_panel()

        right_panel = self.create_image_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 1100])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def create_control_panel(self) -> QWidget:
        """Create the left control panel with buttons and settings"""
        panel = QWidget()
        panel.setMinimumWidth(320)
        layout = QVBoxLayout()

        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()

        self.load_button = QPushButton("Select Folder")
        self.load_button.clicked.connect(self.select_folder)
        file_layout.addWidget(self.load_button)

        self.save_labels_button = QPushButton("Save Labels")
        self.save_labels_button.clicked.connect(self.save_image_labels)        
        file_layout.addWidget(self.save_labels_button)

        self.export_button = QPushButton("Export selected images")
        self.export_button.clicked.connect(self.export_selected)
        file_layout.addWidget(self.export_button)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Sorting group
        sort_group = QGroupBox("Sorting Options")
        sort_layout = QVBoxLayout()

        self.sort_criteria_combo = QComboBox()
        self.sort_criteria_combo.addItems(SORTING_DISPLAY_NAMES)
        self.sort_criteria_combo.currentIndexChanged.connect(self.set_sorting_criteria)
        sort_layout.addWidget(QLabel("Sort by:"))
        sort_layout.addWidget(self.sort_criteria_combo)
        
        self.reverse_checkbox = QCheckBox("Reverse order")
        self.reverse_checkbox.toggled.connect(self.set_reverse_sort)
        sort_layout.addWidget(self.reverse_checkbox)
        
        self.sort_button = QPushButton("Sort Images")
        self.sort_button.clicked.connect(self.sort_images)
        sort_layout.addWidget(self.sort_button)
        
        sort_group.setLayout(sort_layout)
        layout.addWidget(sort_group)

        # AI group
        ai_group = QGroupBox("AI Operations")
        ai_layout = QVBoxLayout()
        
        self.create_tags_button = QPushButton("🏷️ Generate Tags")
        self.create_tags_button.clicked.connect(self.create_image_tags)
        ai_layout.addWidget(self.create_tags_button)
        
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)

        # Progress tracking
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status and spinner
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.spinner_label = QLabel()
        if os.path.exists("spinner.gif"):
            self.spinner = QMovie("spinner.gif")
            self.spinner_label.setMovie(self.spinner)
        self.spinner_label.setVisible(False)
        self.spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.spinner_label)

        # Stats display
        self.stats_label = QLabel("No images loaded")
        layout.addWidget(self.stats_label)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_image_panel(self) -> QWidget:
        """Create the right image display panel"""
        panel = QWidget()
        layout = QVBoxLayout()

        # Image list
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(PREVIEW_WIDTH, PREVIEW_HEIGHT))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.list_widget.itemSelectionChanged.connect(self.update_selection_stats)
        layout.addWidget(self.list_widget)
        
        panel.setLayout(layout)
        return panel
    
    def setup_logger(self):
        """Setup logger (TODO: maybe GUI output)"""
        global logger
        logger = Logger()

    def update_ui_state(self, processing: bool):
        """Update UI elements based on processing state"""

        self.load_button.setDisabled(processing)
        self.sort_button.setDisabled(processing or not self.image_list)
        self.export_button.setDisabled(processing or not self.image_list)
        self.save_labels_button.setDisabled(processing or not self.image_list)
        self.create_tags_button.setDisabled(processing or not self.image_list)

        if processing:
            self.spinner_label.setVisible(True)
            if hasattr(self, "spinner"):
                self.spinner.start()
            self.progress_bar.setVisible(True)
        else:
            self.spinner_label.setVisible(False)
            if hasattr(self, "spinner"):
                self.spinner.stop()
            self.progress_bar.setVisible(False)
    
    def set_sorting_criteria(self, index: int):
        """Set the sorting criteria"""
        if 0 <= index < len(SORTING_CRITERIA):
            self.sorting_criteria_index = index
            logger.info(f"Sorting criteria set to: {SORTING_CRITERIA[index]}")

    def set_reverse_sort(self, checked: bool):
        """Set reverse sorting order"""
        self.reverse_sort = checked
        logger.info(f"Reverse sorting set to: {checked}")

    def update_stats(self):
        """Update statistics display"""
        if not self.image_list:
            self.stats_label.setText("No images loaded")
            return
        
        total = len(self.image_list)
        selected = len(self.list_widget.selectedItems())
        self.stats_label.setText(f"Total: {total} | Selected: {selected}")

    def update_selection_stats(self):
        """Update selection statistics"""
        self.update_stats()

    def select_folder(self):
        """Select and load images from folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        
        self.image_folder_location = folder
        self.load_images_from_folder(folder)

    def load_images_from_folder(self, folder: str):
        """Load images from folder with caching support"""
        labels_file = Path(folder) / "image_labels.json"

        if labels_file.exists():
            if  self.load_cached_labels(labels_file):
                logger.info(f"Loaded cached labels from {labels_file}")
                return
            
        self.analyze_folder_images(folder)

    def load_cached_labels(self, labels_file: Path) -> bool:
        """Load prev computed image labels"""
        try:
            with open(labels_file, 'r', encoding="utf-8") as f:
                data = json.load(f)

            self.image_list.clear()
            self.list_widget.clear()
            self.thumbnail_cache.clear()

            images_data = data.get("images", {})
            folder = Path(data.get("path", self.image_folder_location))

            for image_name, metrics_dict in images_data.items():
                full_path = str(folder / image_name)

                if not os.path.exists(full_path):
                    logger.error(f"Cached image {full_path} does not exist, skipping")
                    continue

                metrics = ImageMetrics.from_dict(metrics_dict)
                self.image_list.append((full_path, metrics))

            self.populate_image_list()
            logger.info(f"Loaded {len(self.image_list)} images from cache")
            self.update_stats()
            return True
        except Exception as e:
            logger.error(f"Error loading cached labels: {e}")
            return False
    
    def analyze_folder_images(self, folder: str):
        """Analyze all images in folder"""

        image_files = []

        for ext in SUPPORTED_FORMATS:
            image_files.extend(Path(folder).glob(f"*{ext}"))
            image_files.extend(Path(folder).glob(f"{ext.upper()}"))

        if not image_files:
            QMessageBox.information(self, "No Images Found", "No supported images files found in the selected folder.")
            return
        
        self.image_list.clear()
        self.list_widget.clear()
        self.thumbnail_cache.clear()
        self.cancel_workers()


        self.pending_tasks = len(image_files)
        self.progress_bar.setMaximum(self.pending_tasks)
        self.progress_bar.setValue(0)
        self.update_ui_state(True)
        self.status_label.setText(f"Analyzing {self.pending_tasks} images...")

        for image_path in image_files:
            worker = ImageAnalysisWorker(str(image_path))
            worker.signals.result.connect(self.on_image_analyzed)
            worker.signals.finished.connect(self.on_worker_finished)
            worker.signals.error.connect(self.on_worker_error)

            worker.signals.thumb_ready.connect(self.on_thumb_ready)
            
            self.workers.append(worker)
            self.threadpool.start(worker)

    def cancel_workers(self):
        """Cancel all running workers"""
        for worker in self.workers:
            worker.cancel()
        self.workers.clear()

    def on_image_analyzed(self, path: str, metrics: ImageMetrics):
        """Handle completed image analysis"""
        self.mutex.lock()
        try:
            self.image_list.append((path, metrics))
            self.add_image_to_list(path)
        finally:
            self.mutex.unlock()
    
    def on_worker_finished(self):
        """Handle worker completion"""
        self.pending_tasks -= 1
        self.progress_bar.setValue(self.progress_bar.maximum() - self.pending_tasks)
        
        if self.pending_tasks <= 0:
            self.update_ui_state(False)
            self.status_label.setText("Analysis complete")
            self.update_stats()
            logger.strong(f"Successfully analyzed {len(self.image_list)} images")
            self.workers.clear()
    
    def on_worker_error(self, path: str, error: str):
        """Handle worker error"""
        logger.error(f"Error processing {path}: {error}")

    def add_image_to_list(self, path: str):
        # placeholder (light-grey square)
        placeholder = QPixmap(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        placeholder.fill(Qt.GlobalColor.lightGray)

        icon = self.get_thumbnail_icon(path) or QIcon(placeholder)
        item = QListWidgetItem(icon, os.path.basename(path))
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)

        self.list_widget.addItem(item)
        self.item_lookup[path] = item 

    def populate_image_list(self):
        """Populate the list widget with all images"""
        for path, _ in self.image_list:
            self.add_image_to_list(path)

    def get_thumbnail_icon(self, path: str) -> QIcon:
        """Get thumbnail icon with caching"""
        # Check cache first
        cached_icon = self.thumbnail_cache.get(path)
        if cached_icon:
            return cached_icon
        
        # Create new thumbnail
        try:
            pixmap = QPixmap(path)
            if pixmap.isNull():
                # Create placeholder icon
                pixmap = QPixmap(PREVIEW_WIDTH, PREVIEW_HEIGHT)
                pixmap.fill(Qt.GlobalColor.lightGray)
            
            icon = self.create_thumbnail_icon(pixmap)
            self.thumbnail_cache.put(path, icon)
            return icon
            
        except Exception as e:
            logger.error(f"Failed to create thumbnail for {path}: {e}")
            # Return placeholder
            pixmap = QPixmap(PREVIEW_WIDTH, PREVIEW_HEIGHT)
            pixmap.fill(Qt.GlobalColor.red)
            return QIcon(pixmap)

    def create_thumbnail_icon(self, pixmap: QPixmap) -> QIcon:
        """Create a properly sized thumbnail icon"""
        # Create canvas
        canvas = QPixmap(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        canvas.fill(Qt.GlobalColor.transparent)
        
        # Scale image maintaining aspect ratio
        scaled = pixmap.scaled(
            PREVIEW_WIDTH, PREVIEW_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Center the image on canvas
        painter = QPainter(canvas)
        x = (PREVIEW_WIDTH - scaled.width()) // 2
        y = (PREVIEW_HEIGHT - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
        
        return QIcon(canvas)
    
    def sort_images(self):
        """Sort images based on selected criteria"""
        if not self.image_list:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        criteria = SORTING_CRITERIA[self.sorting_criteria_index]
        logger.info(f"Sorting by {criteria} (reverse: {self.reverse_sort})")
        
        self.update_ui_state(True)
        self.status_label.setText("Sorting images...")

        worker = SortingWorker(
            self.image_list, 
            self.sorting_criteria_index, 
            self.reverse_sort
        )

        worker.signals.sorted_result.connect(self.on_sorting_finished)
        worker.signals.error.connect(self.on_sorting_error)

        self.workers.append(worker)
        self.threadpool.start(worker)

    def on_thumb_ready(self, path: str, icon: QIcon):
        """Handle thumbnail ready signal"""
        if item := self.item_lookup.get(path):
            item.setIcon(icon)

    def on_sorting_finished(self, sorted_list):
        # 1. update the *data* first
        self.image_list = sorted_list

        # 2. rebuild the view quickly with cheap placeholders
        self.item_lookup.clear()              
        self.list_widget.setUpdatesEnabled(False)
        self.list_widget.clear()

        for path, _ in sorted_list:
            # Use cached thumbnail or a placeholder here:
            icon = self.get_thumbnail_icon(path)
            item = QListWidgetItem(icon, os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.list_widget.addItem(item)
            self.item_lookup[path] = item
        self.list_widget.setUpdatesEnabled(True)

        # 3. tell the user we’re done
        self.status_label.setText("Sorting complete")
        self.update_ui_state(False)           

        self.load_thumbnails_async([path for path, _ in sorted_list])

    def load_thumbnails_async(self, paths: List[str]):
        # Cancel previous thumbnail workers if needed
        self.cancel_workers()

        for path in paths:
            worker = ThumbnailLoaderWorker(path)  # You need to create this worker
            worker.signals.thumb_ready.connect(self.on_thumb_ready)
            self.workers.append(worker)
            self.threadpool.start(worker)

    def on_sorting_error(self, path: str, error: str):
        """Handle sorting error"""
        logger.error(f"Error during sorting: {error}")
        QMessageBox.critical(self, "Sorting Error", f"Failed to sort images:\n{error}")
        self.update_ui_state(False)
        self.status_label.setText("Sorting failed")

    def export_selected(self):
        """Export selected images to a folder"""
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select images to export.")
            return
        
        target_folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not target_folder:
            return
        
        success_count = 0
        error_count = 0
        
        for item in selected_items:
            try:
                src_path = item.data(Qt.ItemDataRole.UserRole)
                dst_path = os.path.join(target_folder, os.path.basename(src_path))
                
                # Handle duplicate names
                counter = 1
                base_name, ext = os.path.splitext(dst_path)
                while os.path.exists(dst_path):
                    dst_path = f"{base_name}_{counter}{ext}"
                    counter += 1
                
                shutil.copy2(src_path, dst_path)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to export {src_path}: {e}")
                error_count += 1
        
        message = f"Export complete!\nSuccessful: {success_count}"
        if error_count > 0:
            message += f"\nFailed: {error_count}"
        
        QMessageBox.information(self, "Export Complete", message)



    def save_image_labels(self):
        """Save computed image labels to JSON file"""
        if not self.image_list:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        if not self.image_folder_location:
            QMessageBox.warning(self, "No Folder", "Please select a folder first.")
            return
        
        labels_file = os.path.join(self.image_folder_location, "image_labels.json")
        
        try:
            labels_data = {
                "path": self.image_folder_location,
                "version": "2.0",
                "created_at": str(QTimer().currentTime()),
                "total_images": len(self.image_list),
                "images": {
                    os.path.basename(path): metrics.to_dict()
                    for path, metrics in self.image_list
                }
            }
            
            with open(labels_file, 'w', encoding='utf-8') as f:
                json.dump(labels_data, f, indent=2, ensure_ascii=False, sort_keys=True)
            
            QMessageBox.information(self, "Save Complete", f"Labels saved to:\n{labels_file}")
            logger.info(f"Saved labels for {len(self.image_list)} images")
            
        except Exception as e:
            logger.error(f"Failed to save labels: {e}")
            QMessageBox.critical(self, "Save Error", f"Failed to save labels:\n{str(e)}")


    def create_image_tags(self):
        """Generate AI tags for images (placeholder for when AI is available)"""
        if not self.image_list:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        # Placeholder implementation - uncomment and modify when AI classifier is available
        self.update_ui_state(True)
        self.status_label.setText("Generating AI tags...")
        
        success_count = 0
        error_count = 0
        
        for i, (path, metrics) in enumerate(self.image_list):
            try:
                img = Image.open(path)
                tags = self.image_classifier.classify_image(img)
                metrics.tags = tags
                success_count += 1
                logger.info(f"Generated tags for {os.path.basename(path)}: {tags}")
                
            except Exception as e:
                logger.error(f"Failed to generate tags for {path}: {e}")
                error_count += 1
            
            # Update progress
            progress = int((i + 1) / len(self.image_list) * 100)
            self.progress_bar.setValue(progress)
        
        self.update_ui_state(False)
        self.status_label.setText("Tag generation complete")
        
        message = f"Tag generation complete!\nSuccessful: {success_count}"
        if error_count > 0:
            message += f"\nFailed: {error_count}"
        
        QMessageBox.information(self, "Tags Generated", message)
        

    def closeEvent(self, event):
        """Handle application close event"""
        # Cancel any running workers
        self.cancel_workers()
        
        # Wait for threadpool to finish
        self.threadpool.waitForDone(3000)  # Wait max 3 seconds
        
        event.accept()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key.Key_Delete:
            # Remove selected items (optional feature)
            self.remove_selected_images()
        elif event.key() == Qt.Key.Key_F5:
            # Refresh current folder
            if self.image_folder_location:
                self.load_images_from_folder(self.image_folder_location)
        else:
            super().keyPressEvent(event)

    def remove_selected_images(self):
        """Remove selected images from the list (not from disk)"""
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return
        
        reply = QMessageBox.question(
            self,
            "Remove Images",
            f"Remove {len(selected_items)} selected images from the list?\n"
            "(This will not delete the files from disk)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Get paths of selected items
            selected_paths = set()
            for item in selected_items:
                path = item.data(Qt.ItemDataRole.UserRole)
                selected_paths.add(path)
            
            # Remove from image list
            self.image_list = [
                (path, metrics) for path, metrics in self.image_list 
                if path not in selected_paths
            ]
            
            # Remove from list widget
            for item in selected_items:
                row = self.list_widget.row(item)
                self.list_widget.takeItem(row)
            
            # Update cache
            for path in selected_paths:
                if path in self.thumbnail_cache.cache:
                    del self.thumbnail_cache.cache[path]
            
            self.update_stats()
            logger.info(f"Removed {len(selected_paths)} images from list")


def main():
    app = QApplication(sys.argv)
    
    app.setApplicationName("Advanced Image Sorter")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("ImageSorter")
    
    try:
        window = ImageSorterApp()
        window.show()
        
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        QMessageBox.critical(None, "Startup Error", f"Failed to start application:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
