import sys
import os
import shutil
from typing import List, Tuple, Dict, Set
import json
from dataclasses import dataclass, asdict
from datetime import datetime

from PIL import Image
from PIL.ExifTags import TAGS

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QListWidget, QListWidgetItem, QHBoxLayout, QComboBox, 
    QMessageBox, QAbstractItemView, QProgressBar, QSplitter,
    QGroupBox, QCheckBox, QSpinBox, QTextEdit, QTabWidget,
    QLineEdit, QListView
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import (Qt, QSize, QThreadPool, QRunnable, QTimer, QMutex)


from logger import Logger
from imageClassifier import ImageClassifierAi
from imageMetrics import ImageMetrics
from worker import ImageAnalysisWorker, SmartSortingWorker
from cache import EnhancedThumbnailCache
from imageProcessor import _pil_to_qicon

logger = Logger()

SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.svg')
SORTING_CRITERIA = [
    "dominant_color", "brightness", "aspect_ratio", "file_size", 
    "resolution", "sharpness", "color_variance", "created_date", 
    "modified_date", "filename"
]

SORTING_DISPLAY_NAMES = [
    "Dominant Color", "Brightness", "Aspect Ratio", "File Size",
    "Resolution", "Sharpness", "Color Variance", "Created Date",
    "Modified Date", "Filename"
]

SORTING_ALGORITHMS = ["Quick Sort", "Merge Sort", "Heap Sort", "Radix Sort"]

# UI Constants
PREVIEW_WIDTH = 200
PREVIEW_HEIGHT = 200
THUMBNAIL_SIZE = 250
MAX_WORKERS = min(12, (os.cpu_count() or 4) * 2)
CACHE_SIZE = 300
BATCH_SIZE = 50


class ImageSorterApp(QWidget):
    """Enhanced main application with better architecture"""

    def __init__(self):
        super().__init__()

        # Core data
        self.image_list: List[Tuple[str, ImageMetrics]] = []
        self.filtered_list: List[Tuple[str, ImageMetrics]] = []
        self.image_folder_location = ""
        self.sorting_criteria_index = 0
        self.reverse_sort = False
        self.current_algorithm = "Quick Sort"

        # Threading
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(MAX_WORKERS)
        self.workers: List[QRunnable] = []
        self.pending_batches = 0
        self.completed_batches = 0
        self.mutex = QMutex()

        # UI state
        self.item_lookup: Dict[str, QListWidgetItem] = {}
        self.selected_tags: Set[str] = set()
        
        # Enhanced caching
        self.thumbnail_cache = EnhancedThumbnailCache(max_size=CACHE_SIZE)
        
        # Mock AI classifier
        self.image_classifier = ImageClassifierAi("path/to/model")

        self.setup_ui()
        self.setup_shortcuts()
        
        # Auto-save timer
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_labels)
        self.auto_save_timer.start(300000)  # 5 minutes

    def setup_ui(self):
        """Initialize enhanced UI"""
        self.setWindowTitle("Advanced Image Sorter v2.0")
        self.setMinimumSize(1000, 700)
        self.resize(1600, 900)

        main_layout = QHBoxLayout()

        # Create tabbed interface
        left_panel = self.create_tabbed_control_panel()
        right_panel = self.create_enhanced_image_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 1200])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def create_tabbed_control_panel(self) -> QWidget:
        """Create tabbed control panel"""
        tab_widget = QTabWidget()
        tab_widget.setMinimumWidth(400)

        # File operations tab
        file_tab = QWidget()
        file_layout = QVBoxLayout()
        
        # Load section
        load_group = QGroupBox("Load Images")
        load_layout = QVBoxLayout()
        
        self.load_button = QPushButton("📁 Select Folder")
        self.load_button.clicked.connect(self.select_folder)
        load_layout.addWidget(self.load_button)
        
        self.recursive_checkbox = QCheckBox("Include subfolders")
        load_layout.addWidget(self.recursive_checkbox)
        
        load_group.setLayout(load_layout)
        file_layout.addWidget(load_group)

        # Export section
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout()
        
        self.export_button = QPushButton("📤 Export Selected")
        self.export_button.clicked.connect(self.export_selected)
        export_layout.addWidget(self.export_button)
        
        self.copy_structure_checkbox = QCheckBox("Preserve folder structure")
        export_layout.addWidget(self.copy_structure_checkbox)
        
        export_group.setLayout(export_layout)
        file_layout.addWidget(export_group)

        # Save section
        save_group = QGroupBox("Save Data")
        save_layout = QVBoxLayout()
        
        self.save_labels_button = QPushButton("💾 Save Analysis")
        self.save_labels_button.clicked.connect(self.save_image_labels)
        save_layout.addWidget(self.save_labels_button)
        
        self.auto_save_label = QLabel("Auto-save: Every 5 minutes")
        save_layout.addWidget(self.auto_save_label)
        
        save_group.setLayout(save_layout)
        file_layout.addWidget(save_group)
        
        file_layout.addStretch()
        file_tab.setLayout(file_layout)

        # Sorting tab
        sort_tab = QWidget()
        sort_layout = QVBoxLayout()
        
        # Algorithm selection
        algo_group = QGroupBox("Sorting Algorithm")
        algo_layout = QVBoxLayout()
        
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(SORTING_ALGORITHMS)
        self.algorithm_combo.currentTextChanged.connect(self.set_sorting_algorithm)
        algo_layout.addWidget(self.algorithm_combo)
        
        algo_group.setLayout(algo_layout)
        sort_layout.addWidget(algo_group)
        
        # Criteria selection
        criteria_group = QGroupBox("Sort Criteria")
        criteria_layout = QVBoxLayout()
        
        self.sort_criteria_combo = QComboBox()
        self.sort_criteria_combo.addItems(SORTING_DISPLAY_NAMES)
        self.sort_criteria_combo.currentIndexChanged.connect(self.set_sorting_criteria)
        criteria_layout.addWidget(self.sort_criteria_combo)
        
        self.reverse_checkbox = QCheckBox("Reverse order")
        self.reverse_checkbox.toggled.connect(self.set_reverse_sort)
        criteria_layout.addWidget(self.reverse_checkbox)
        
        self.sort_button = QPushButton("🔄 Sort Images")
        self.sort_button.clicked.connect(self.sort_images)
        criteria_layout.addWidget(self.sort_button)
        
        criteria_group.setLayout(criteria_layout)
        sort_layout.addWidget(criteria_group)
        
        sort_layout.addStretch()
        sort_tab.setLayout(sort_layout)

        # Filter tab
        filter_tab = QWidget()
        filter_layout = QVBoxLayout()
        
        # Search
        search_group = QGroupBox("Search & Filter")
        search_layout = QVBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search filenames...")
        self.search_input.textChanged.connect(self.filter_images)
        search_layout.addWidget(self.search_input)
        
        # Size filter
        size_filter_layout = QHBoxLayout()
        size_filter_layout.addWidget(QLabel("Min file size (KB):"))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setMaximum(999999)
        self.min_size_spin.valueChanged.connect(self.filter_images)
        size_filter_layout.addWidget(self.min_size_spin)
        search_layout.addLayout(size_filter_layout)
        
        # Resolution filter
        res_filter_layout = QHBoxLayout()
        res_filter_layout.addWidget(QLabel("Min resolution:"))
        self.min_width_spin = QSpinBox()
        self.min_width_spin.setMaximum(99999)
        self.min_width_spin.valueChanged.connect(self.filter_images)
        res_filter_layout.addWidget(self.min_width_spin)
        res_filter_layout.addWidget(QLabel("x"))
        self.min_height_spin = QSpinBox()
        self.min_height_spin.setMaximum(99999)
        self.min_height_spin.valueChanged.connect(self.filter_images)
        res_filter_layout.addWidget(self.min_height_spin)
        search_layout.addLayout(res_filter_layout)
        
        search_group.setLayout(search_layout)
        filter_layout.addWidget(search_group)
        
        filter_layout.addStretch()
        filter_tab.setLayout(filter_layout)

        # AI tab
        ai_tab = QWidget()
        ai_layout = QVBoxLayout()
        
        ai_group = QGroupBox("AI Analysis")
        ai_group_layout = QVBoxLayout()
        
        self.create_tags_button = QPushButton("🏷️ Generate Tags")
        self.create_tags_button.clicked.connect(self.create_image_tags)
        ai_group_layout.addWidget(self.create_tags_button)
        
        self.batch_size_layout = QHBoxLayout()
        self.batch_size_layout.addWidget(QLabel("Batch size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 100)
        self.batch_size_spin.setValue(BATCH_SIZE)
        self.batch_size_layout.addWidget(self.batch_size_spin)
        ai_group_layout.addLayout(self.batch_size_layout)
        
        ai_group.setLayout(ai_group_layout)
        ai_layout.addWidget(ai_group)
        
        ai_layout.addStretch()
        ai_tab.setLayout(ai_layout)

        # Add tabs
        tab_widget.addTab(file_tab, "Files")
        tab_widget.addTab(sort_tab, "Sorting")
        tab_widget.addTab(filter_tab, "Filtering")
        tab_widget.addTab(ai_tab, "AI")

        # Status panel
        status_widget = QWidget()
        status_layout = QVBoxLayout()
        
        # Progress tracking
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.total_images_label = QLabel("Total images: 0")
        self.filtered_images_label = QLabel("Filtered: 0")
        self.cache_status_label = QLabel("Cache: 0/200")
        
        stats_layout.addWidget(self.total_images_label)
        stats_layout.addWidget(self.filtered_images_label)
        stats_layout.addWidget(self.cache_status_label)
        
        stats_group.setLayout(stats_layout)
        status_layout.addWidget(stats_group)
        
        status_widget.setLayout(status_layout)

        # Combine everything
        main_panel = QWidget()
        main_panel_layout = QVBoxLayout()
        main_panel_layout.addWidget(tab_widget)
        main_panel_layout.addWidget(status_widget)
        main_panel.setLayout(main_panel_layout)

        return main_panel

    def create_enhanced_image_panel(self) -> QWidget:
        """Create enhanced image display panel"""
        panel = QWidget()
        layout = QVBoxLayout()

        # Image list
        self.image_list_widget = QListWidget()
        self.image_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.image_list_widget.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))

        self.image_list_widget.setViewMode(QListView.ViewMode.IconMode)
        self.image_list_widget.setGridSize(QSize(THUMBNAIL_SIZE + 12, THUMBNAIL_SIZE + 28))
        self.image_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)

        self.image_list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.image_list_widget.itemDoubleClicked.connect(self.open_image_preview)
        
        layout.addWidget(self.image_list_widget)

        # Image details
        details_group = QGroupBox("Image Details")
        details_layout = QVBoxLayout()
        
        details_group.setMaximumHeight(250)

        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(225)
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        panel.setLayout(layout)
        return panel

    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Can be implemented later
        pass

    def select_folder(self):
        """Select folder containing images"""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_folder_location = folder
            self.load_images_from_folder(folder)

    def load_images_from_folder(self, folder_path: str):
        """Load images from folder with progress tracking"""
        try:
            # Clear existing data
            self.image_list.clear()
            self.filtered_list.clear()
            self.image_list_widget.clear()
            self.item_lookup.clear()
            self.thumbnail_cache.clear()

            # Find image files
            image_paths = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(SUPPORTED_FORMATS):
                        image_paths.append(os.path.join(root, file))
                
                if not self.recursive_checkbox.isChecked():
                    break  # Only scan root directory

            if not image_paths:
                QMessageBox.information(self, "No Images", "No supported image files found.")
                return

            # Start processing
            self.status_label.setText(f"Loading {len(image_paths)} images...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(image_paths))
            self.progress_bar.setValue(0)

            # Process in batches
            batch_size = self.batch_size_spin.value()
            self.pending_batches = (len(image_paths) + batch_size - 1) // batch_size
            self.completed_batches = 0

            # Complete the load_images_from_folder method (continuation from line 450+)
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                worker = ImageAnalysisWorker(batch, PREVIEW_WIDTH, PREVIEW_HEIGHT, batch_id=(i // batch_size))
                
                # Connect signals
                worker.signals.result.connect(self.on_image_analyzed)
                worker.signals.progress.connect(self.update_progress)
                worker.signals.error.connect(self.on_analysis_error)
                worker.signals.thumb_ready.connect(self.on_thumbnail_ready)
                worker.signals.batch_complete.connect(self.on_batch_complete)
                worker.signals.finished.connect(self.on_worker_finished)
                
                self.workers.append(worker)
                self.threadpool.start(worker)

        except Exception as e:
            logger.error(f"Error loading images: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load images: {e}")
            self.progress_bar.setVisible(False)

    def on_image_analyzed(self, path: str, metrics: ImageMetrics):
        """Handle analyzed image result"""
        self.image_list.append((path, metrics))
        
        # Create list item
        item = QListWidgetItem()
        item.setText(os.path.basename(path))
        item.setData(Qt.ItemDataRole.UserRole, path)
        
        # Add to widget
        self.image_list_widget.addItem(item)
        self.item_lookup[path] = item
        
        # Update UI
        self.update_statistics()

    def on_thumbnail_ready(self, path: str, icon: QIcon):
        """Handle thumbnail creation"""
        self.thumbnail_cache.put(path, icon)
        
        if path in self.item_lookup:
            self.item_lookup[path].setIcon(icon)

    def on_batch_complete(self, batch_id: int):
        """Handle batch completion"""
        self.completed_batches += 1
        completion_pct = (self.completed_batches / self.pending_batches) * 100
        self.status_label.setText(f"Processed {self.completed_batches}/{self.pending_batches} batches ({completion_pct:.0f}%)")

    def on_worker_finished(self):
        """Handle worker completion"""
        if self.completed_batches >= self.pending_batches:
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"Loaded {len(self.image_list)} images")
            self.filtered_list = self.image_list.copy()
            self.update_statistics()

    def update_progress(self, value: int, message: str):
        """Update progress bar"""
        self.progress_bar.setValue(self.progress_bar.value() + 1)
        self.status_label.setText(message)

    def on_analysis_error(self, path: str, error: str):
        """Handle analysis errors"""
        logger.error(f"Analysis failed for {path}: {error}")

    def set_sorting_criteria(self, index: int):
        """Set sorting criteria"""
        self.sorting_criteria_index = index

    def set_reverse_sort(self, reverse: bool):
        """Set reverse sorting"""
        self.reverse_sort = reverse

    def set_sorting_algorithm(self, algorithm: str):
        """Set sorting algorithm"""
        self.current_algorithm = algorithm

    def sort_images(self):
        """Sort images using selected criteria and algorithm"""
        if not self.filtered_list:
            QMessageBox.information(self, "No Images", "No images to sort.")
            return

        self.status_label.setText("Sorting images...")
        
        worker = SmartSortingWorker(
            self.filtered_list, 
            self.sorting_criteria_index, 
            self.reverse_sort, 
            self.current_algorithm
        )
        
        worker.signals.sorted_result.connect(self.on_sorting_complete)
        worker.signals.error.connect(self.on_sorting_error)
        
        self.threadpool.start(worker)

    def on_sorting_complete(self, sorted_list: List[Tuple[str, ImageMetrics]]):
        """Handle sorting completion"""
        self.filtered_list = sorted_list
        self.refresh_image_list()
        self.status_label.setText("Sorting complete")

    def on_sorting_error(self, error_type: str, error_msg: str):
        """Handle sorting errors"""
        QMessageBox.critical(self, "Sorting Error", f"Failed to sort images: {error_msg}")
        self.status_label.setText("Sorting failed")

    def refresh_image_list(self):
        """Refresh the image list widget"""
        self.image_list_widget.clear()
        self.item_lookup.clear()
        
        for path, metrics in self.filtered_list:
            item = QListWidgetItem()
            item.setText(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            
            # Set thumbnail from cache or create new one from path
            cached_icon = self.thumbnail_cache.get(path)
            if cached_icon:
                item.setIcon(cached_icon)
            else:
                try:
                    with Image.open(path) as img:
                        img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
                        icon = _pil_to_qicon(img)
                        self.thumbnail_cache.put(path, icon)
                        item.setIcon(icon)
                except Exception as e:
                    logger.error(f"Failed to create thumbnail for {path}: {e}")
            
            self.image_list_widget.addItem(item)
            self.item_lookup[path] = item

    def filter_images(self):
        """Filter images based on search criteria"""
        search_text = self.search_input.text().lower()
        min_size = self.min_size_spin.value() * 1024  # Convert KB to bytes
        min_width = self.min_width_spin.value()
        min_height = self.min_height_spin.value()
        
        self.filtered_list = []
        
        for path, metrics in self.image_list:
            # Filename filter
            if search_text and search_text not in os.path.basename(path).lower():
                continue
            
            # Size filter
            if metrics.file_size < min_size:
                continue
            
            # Resolution filter
            width, height = metrics.resolution
            if width < min_width or height < min_height:
                continue
            
            self.filtered_list.append((path, metrics))
        
        self.refresh_image_list()
        self.update_statistics()

    def on_selection_changed(self):
        """Handle selection change in image list"""
        selected_items = self.image_list_widget.selectedItems()
        
        if len(selected_items) == 1:
            item = selected_items[0]
            path = item.data(Qt.ItemDataRole.UserRole)
            
            # Find metrics for this path
            for img_path, metrics in self.filtered_list:
                if img_path == path:
                    self.display_image_details(path, metrics)
                    break
        elif len(selected_items) > 1:
            self.details_text.setText(f"Selected {len(selected_items)} images")
        else:
            self.details_text.clear()

    def display_image_details(self, path: str, metrics: ImageMetrics):
        """Display detailed image information"""
        details = f"File: {os.path.basename(path)}\n"
        details += f"Path: {path}\n"
        details += f"Size: {metrics.file_size / 1024:.1f} KB\n"
        details += f"Resolution: {metrics.resolution[0]} x {metrics.resolution[1]}\n"
        details += f"Aspect Ratio: {metrics.aspect_ratio:.2f}\n"
        details += f"Brightness: {metrics.brightness:.1f}\n"
        details += f"Contrast: {metrics.contrast:.1f}\n"
        details += f"Sharpness: {metrics.sharpness:.1f}\n"
        details += f"Dominant Color: RGB{metrics.dominant_color}\n"
        
        if metrics.created_date:
            details += f"Created: {metrics.created_date}\n"
        if metrics.modified_date:
            details += f"Modified: {metrics.modified_date}\n"
        
        if metrics.tags:
            details += f"Tags: {', '.join(metrics.tags)}\n"
        
        self.details_text.setText(details)

    def open_image_preview(self, item: QListWidgetItem):
        """Open image in system default viewer"""
        path = item.data(Qt.ItemDataRole.UserRole)
        try:
            if sys.platform.startswith('darwin'):  # macOS
                os.system(f'open "{path}"')
            elif sys.platform.startswith('win'):   # Windows
                os.startfile(path)
            else:  # Linux
                os.system(f'xdg-open "{path}"')
        except Exception as e:
            logger.error(f"Failed to open image: {e}")

    def create_image_tags(self):
        """Generate AI tags for images"""
        if not self.filtered_list:
            QMessageBox.information(self, "No Images", "No images to tag.")
            return
        
        # Mock implementation - in real app would use actual AI
        for path, metrics in self.filtered_list:
            if not metrics.tags:
                mock_tags = self.image_classifier.classify_image(path)
                metrics.tags = mock_tags
        
        self.status_label.setText("AI tagging complete")
        QMessageBox.information(self, "Tagging Complete", "AI tags generated for all images.")

    def export_selected(self):
        """Export selected images to new folder"""
        selected_items = self.image_list_widget.selectedItems()
        
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Please select images to export.")
            return
        
        output_folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not output_folder:
            return
        
        try:
            for item in selected_items:
                path = item.data(Qt.ItemDataRole.UserRole)
                filename = os.path.basename(path)
                destination = os.path.join(output_folder, filename)
                
                # Handle duplicates
                counter = 1
                base_name, ext = os.path.splitext(filename)
                while os.path.exists(destination):
                    new_name = f"{base_name}_{counter}{ext}"
                    destination = os.path.join(output_folder, new_name)
                    counter += 1
                
                shutil.copy2(path, destination)
            
            QMessageBox.information(self, "Export Complete", 
                                  f"Exported {len(selected_items)} images to {output_folder}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export images: {e}")

    def save_image_labels(self):
        """Save image analysis data to JSON"""
        if not self.image_list:
            QMessageBox.information(self, "No Data", "No image data to save.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis Data", "image_analysis.json", "JSON Files (*.json)"
        )
        
        if filename:
            try:
                data = {
                    "folder": self.image_folder_location,
                    "timestamp": datetime.now().isoformat(),
                    "images": {path: metrics.to_dict() for path, metrics in self.image_list}
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "Save Complete", f"Analysis data saved to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save data: {e}")

    def auto_save_labels(self):
        """Auto-save functionality"""
        if self.image_list and self.image_folder_location:
            try:
                auto_save_path = os.path.join(self.image_folder_location, ".image_analysis_autosave.json")
                data = {
                    "folder": self.image_folder_location,
                    "timestamp": datetime.now().isoformat(),
                    "images": {path: metrics.to_dict() for path, metrics in self.image_list}
                }
                
                with open(auto_save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                logger.info("Auto-save completed")
                
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")

    def update_statistics(self):
        """Update statistics display"""
        self.total_images_label.setText(f"Total images: {len(self.image_list)}")
        self.filtered_images_label.setText(f"Filtered: {len(self.filtered_list)}")
        self.cache_status_label.setText(self.thumbnail_cache.get_memory_usage())

    def closeEvent(self, event):
        """Handle application close"""
        # Cancel all workers
        for worker in self.workers:
            if hasattr(worker, 'cancel'):
                worker.cancel()
        
        # Wait for threads to finish
        self.threadpool.waitForDone(3000)  # 3 second timeout
        
        # Auto-save before closing
        self.auto_save_labels()
        
        event.accept()


def main():
    """Main application entry point"""
    logger.info("Starting Advanced Image Sorter v2.5")
    app = QApplication(sys.argv)
    app.setApplicationName("Advanced Image Sorter")
    app.setApplicationVersion("2.5")
    app.setStyle("windows11")
    
    # List available styles
    # import PyQt6.QtWidgets as PyQt6QtWidgets
    # print(PyQt6QtWidgets.QStyleFactory.keys())  
    
    # Set application icon if available
    try:
        app.setWindowIcon(QIcon("icon.png"))
    except:
        pass
    
    window = ImageSorterApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()