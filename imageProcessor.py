from datetime import datetime
import os
from PIL import Image, ImageStat, ImageFilter
from PIL.ExifTags import TAGS
from typing import Tuple, List, Dict, Any, Optional
from imageMetrics import ImageMetrics
from PyQt6.QtGui import QIcon, QPixmap, QImage

from logger import Logger

class ImageProcessor:
    """Enhanced image processing with better algorithms and error handling"""
    global logger 
    logger = Logger()

    @staticmethod
    def compute_dominant_color(img: Image.Image) -> Tuple[int, int, int]:
        """Compute dominant color using simple quantization"""
        try:
            img_small = img.convert("RGB").resize((50, 50))
            img_quantized = img_small.quantize(colors=8)
            palette = img_quantized.getpalette()
            
            if not palette:
                return (128, 128, 128)
            
            colors = img_quantized.getcolors(maxcolors=8)
            if not colors:
                return (128, 128, 128)
            
            most_frequent = max(colors, key=lambda x: x[0])
            color_index = most_frequent[1]
            
            r = palette[color_index * 3]
            g = palette[color_index * 3 + 1] 
            b = palette[color_index * 3 + 2]
            
            return (r, g, b)
        except Exception as e:
            logger.error(f"Error computing dominant color: {e}")
            return (128, 128, 128)
    
    @staticmethod
    def compute_brightness(img: Image.Image) -> float:
        """Compute average brightness using ImageStat"""
        try:
            stat = ImageStat.Stat(img.convert("L").resize((100, 100)))
            return stat.mean[0]
        except Exception as e:
            logger.error(f"Error computing brightness: {e}")
            return 128.0
    
    @staticmethod
    def compute_contrast(img: Image.Image) -> float:
        """Compute contrast using standard deviation"""
        try:
            stat = ImageStat.Stat(img.convert("L").resize((100, 100)))
            return stat.stddev[0]
        except Exception as e:
            logger.error(f"Error computing contrast: {e}")
            return 0.0
    
    @staticmethod
    def compute_aspect_ratio(img: Image.Image) -> float:
        """Compute width/height ratio"""
        width, height = img.size
        return width / height if height != 0 else 1.0
    
    @staticmethod
    def compute_file_info(image_path: str) -> Tuple[int, Optional[str], Optional[str]]:
        """Get file size and timestamps"""
        try:
            stat = os.stat(image_path)
            file_size = stat.st_size
            created_date = datetime.fromtimestamp(stat.st_ctime).isoformat()
            modified_date = datetime.fromtimestamp(stat.st_mtime).isoformat()
            return file_size, created_date, modified_date
        except Exception as e:
            logger.error(f"Error getting file info for {image_path}: {e}")
            return 0, None, None
    
    @staticmethod
    def compute_resolution(img: Image.Image) -> Tuple[int, int]:
        """Get image resolution"""
        return img.size
    
    @staticmethod
    def compute_sharpness(img: Image.Image) -> float:
        """Compute sharpness using Laplacian variance - optimized"""
        try:
            img_gray = img.convert("L").resize((200, 200))
            edges = img_gray.filter(ImageFilter.Kernel((3, 3), 
                [-1, -1, -1, -1, 8, -1, -1, -1, -1], 1, 0))
            stat = ImageStat.Stat(edges)
            return stat.var[0]
        except Exception as e:
            logger.error(f"Error computing sharpness: {e}")
            return 0.0
    
    @staticmethod
    def compute_color_variance(img: Image.Image) -> Tuple[float, float, float]:
        """Compute color variance per channel using PIL stats"""
        try:
            img_rgb = img.resize((100, 100)).convert("RGB")
            r, g, b = img_rgb.split()
            
            r_stat = ImageStat.Stat(r)
            g_stat = ImageStat.Stat(g)
            b_stat = ImageStat.Stat(b)
            
            return (r_stat.var[0], g_stat.var[0], b_stat.var[0])
        except Exception as e:
            logger.error(f"Error computing color variance: {e}")
            return (0.0, 0.0, 0.0)
    
    @staticmethod
    def extract_exif_data(img: Image.Image) -> Dict[str, Any]:
        """Extract EXIF data from image"""
        try:
            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)
            return exif_data
        except Exception as e:
            logger.error(f"Error extracting EXIF data: {e}")
            return {}
    
    @staticmethod
    def compute_histogram(img: Image.Image) -> Tuple[List[int], List[int], List[int]]:
        """Compute RGB histogram"""
        try:
            img_rgb = img.convert("RGB").resize((100, 100))
            r, g, b = img_rgb.split()
            return (r.histogram(), g.histogram(), b.histogram())
        except Exception as e:
            logger.error(f"Error computing histogram: {e}")
            return ([], [], [])
    
    @classmethod
    def analyze_image(cls, image_path: str) -> ImageMetrics:
        """Analyze image and return comprehensive metrics"""
        try:
            with Image.open(image_path) as img:
                file_size, created_date, modified_date = cls.compute_file_info(image_path)
                
                return ImageMetrics(
                    dominant_color=cls.compute_dominant_color(img),
                    brightness=cls.compute_brightness(img),
                    contrast=cls.compute_contrast(img),
                    aspect_ratio=cls.compute_aspect_ratio(img),
                    file_size=file_size,
                    resolution=cls.compute_resolution(img),
                    sharpness=cls.compute_sharpness(img),
                    color_variance=cls.compute_color_variance(img),
                    created_date=created_date,
                    modified_date=modified_date,
                    exif_data=cls.extract_exif_data(img),
                    histogram=cls.compute_histogram(img)
                )
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            return ImageMetrics()
        
def _pil_to_qicon(img: Image.Image) -> QIcon:
    """Return a deep-copied QIcon from a resized PIL image."""
    img = img.convert("RGB")
    w, h = img.size
    bytes_per_line = 3 * w
    qimg = QImage(
        img.tobytes(), w, h,
        bytes_per_line,
        QImage.Format.Format_RGB888
    ).copy()                                      
    pixmap = QPixmap.fromImage(qimg)              
    return QIcon(pixmap)