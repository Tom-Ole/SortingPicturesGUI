import os

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