from collections import deque
from typing import Dict, Optional
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import  QMutex


class EnhancedThumbnailCache:
    """LRU cache with memory management"""

    def __init__(self, max_size: int):
        self.cache: Dict[str, QIcon] = {}
        self.max_size = max_size
        self.access_order: deque = deque()
        self.mutex = QMutex()

    def get(self, path: str) -> Optional[QIcon]:
        if path in self.cache:
            self.access_order.remove(path)
            self.access_order.append(path)
            return self.cache[path]
        return None
    
    def put(self, path: str, icon: QIcon):
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.popleft()
            del self.cache[oldest]

        self.cache[path] = icon
        self.access_order.append(path)

    def clear(self):
        self.cache.clear()
        self.access_order.clear()
    
    def get_memory_usage(self) -> str:
        """Get estimated memory usage"""
        return f"{len(self.cache)}/{self.max_size} thumbnails cached"