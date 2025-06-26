from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any, Optional

@dataclass
class ImageMetrics:
    """Enhanced image metrics with validation and EXIF data"""
    dominant_color: Tuple[int, int, int] = (0, 0, 0)
    brightness: float = 0.0
    contrast: float = 0.0
    aspect_ratio: float = 1.0
    file_size: int = 0
    resolution: Tuple[int, int] = (0, 0)
    sharpness: float = 0.0
    color_variance: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    tags: List[str] = None
    exif_data: Dict[str, Any] = None
    histogram: Tuple[List[int], List[int], List[int]] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.exif_data is None:
            self.exif_data = {}
        if self.histogram is None:
            self.histogram = ([], [], [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageMetrics':
        """Create instance from dictionary"""
        return cls(**data)