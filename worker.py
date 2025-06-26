from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon
from PIL import Image
from typing import List, Tuple, Any
import os
import time

from imageMetrics import ImageMetrics
from imageProcessor import ImageProcessor, _pil_to_qicon


class WorkerSignals(QObject):
    """Enhanced worker signals with better progress tracking"""
    finished = pyqtSignal()
    result = pyqtSignal(str, ImageMetrics)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str, str)
    sorted_result = pyqtSignal(list)
    thumb_ready = pyqtSignal(str, QIcon)
    batch_complete = pyqtSignal(int)

class ImageAnalysisWorker(QRunnable):
    """Optimized worker with better error handling and cancellation"""

    def __init__(self, paths: List[str], PREVIEW_WIDTH: int, PREVIEW_HEIGHT: int, batch_id: int = 0,):
        super().__init__()
        self.paths = paths
        self.batch_id = batch_id
        self.signals = WorkerSignals()
        self.is_cancelled = False
        self.PREVIEW_WIDTH = PREVIEW_WIDTH
        self.PREVIEW_HEIGHT = PREVIEW_HEIGHT

    @pyqtSlot()
    def run(self):
        try:
            total = len(self.paths)
            for i, path in enumerate(self.paths):
                if self.is_cancelled:
                    return
                
                try:
                    metrics = ImageProcessor.analyze_image(path)
                    self.signals.result.emit(path, metrics)
                    
                    self._create_thumbnail(path)
                    
                    progress = int((i + 1) / total * 100)
                    self.signals.progress.emit(progress, f"Processing {os.path.basename(path)}")
                    
                except Exception as e:
                    self.signals.error.emit(path, str(e))
                    
            self.signals.batch_complete.emit(self.batch_id)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
        finally:
            self.signals.finished.emit()


    def _create_thumbnail(self, path: str):
        """Create thumbnail with better quality"""
        try:
            with Image.open(path) as img:
                img.thumbnail((self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT), Image.Resampling.LANCZOS)
                
                icon = _pil_to_qicon(img)
                
                self.signals.thumb_ready.emit(path, icon)
                
        except Exception as e:
            logger.error(f"Thumbnail creation failed for {path}: {e}")

    def cancel(self):
        self.is_cancelled = True

class SmartSortingWorker(QRunnable):

    global SORTING_CRITERIA
    global logger

    """Enhanced sorting with multiple algorithms"""
    
    def __init__(self, image_list: List[Tuple[str, ImageMetrics]], 
                 criteria_index: int, reverse: bool, algorithm: str = "Quick Sort"):
        super().__init__()
        self.image_list = image_list.copy()
        self.criteria_index = criteria_index
        self.reverse = reverse
        self.algorithm = algorithm
        self.signals = WorkerSignals()
        self.is_cancelled = False
    
    @pyqtSlot()
    def run(self):
        try:
            if self.is_cancelled:
                return
            
            start_time = time.time()
            criteria = SORTING_CRITERIA[self.criteria_index]
            
            # Use Python's built-in sorting (Timsort)
            sorted_list = sorted(self.image_list, 
                               key=lambda x: self._get_sort_key(x, criteria), 
                               reverse=self.reverse)
            
            end_time = time.time()
            logger.info(f"{self.algorithm} completed in {end_time - start_time:.2f} seconds")
            
            if not self.is_cancelled:
                self.signals.sorted_result.emit(sorted_list)
                
        except Exception as e:
            logger.error(f"Error in sorting worker: {e}")
            self.signals.error.emit("sorting", str(e))
    
    def _get_sort_key(self, item: Tuple[str, ImageMetrics], criteria: str) -> Any:
        """Get sort key for given criteria"""
        path, metrics = item
        
        if criteria == "dominant_color":
            r, g, b = metrics.dominant_color
            return 0.299 * r + 0.587 * g + 0.114 * b
        elif criteria == "resolution":
            width, height = metrics.resolution
            return width * height
        elif criteria == "color_variance":
            return sum(metrics.color_variance)
        elif criteria == "filename":
            return os.path.basename(path).lower()
        elif criteria in ["created_date", "modified_date"]:
            date_str = getattr(metrics, criteria)
            return date_str if date_str else "1970-01-01T00:00:00"
        else:
            return getattr(metrics, criteria, 0)
    
    def cancel(self):
        self.is_cancelled = True