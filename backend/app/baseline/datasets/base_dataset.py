"""
Base Dataset Class
==================
Abstract base class for ground truth dataset loaders.
Provides interface for loading annotations, images, and ground truth data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np


@dataclass
class BoundingBox:
    """Bounding box representation."""

    x1: float  # Top-left x
    y1: float  # Top-left y
    x2: float  # Bottom-right x
    y2: float  # Bottom-right y

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Convert to (x_center, y_center, width, height) format."""
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2) format."""
        return (self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BoundingBox":
        """Create from (x_center, y_center, width, height) format."""
        return cls(
            x1=x - w / 2,
            y1=y - h / 2,
            x2=x + w / 2,
            y2=y + h / 2,
        )

    @classmethod
    def from_coco(cls, x: float, y: float, w: float, h: float) -> "BoundingBox":
        """Create from COCO format (x_min, y_min, width, height)."""
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)


@dataclass
class GroundTruthAnnotation:
    """Single ground truth annotation."""

    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float = 1.0  # Ground truth is always 1.0
    is_difficult: bool = False  # Difficult/occluded objects
    is_truncated: bool = False  # Truncated objects
    track_id: Optional[int] = None  # For tracking datasets

    @property
    def area(self) -> float:
        return self.bbox.area


@dataclass
class ImageAnnotation:
    """All annotations for a single image."""

    image_id: str
    image_path: str
    width: int
    height: int
    annotations: list[GroundTruthAnnotation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_objects(self) -> int:
        return len(self.annotations)

    def get_annotations_by_class(self, class_id: int) -> list[GroundTruthAnnotation]:
        """Get all annotations for a specific class."""
        return [ann for ann in self.annotations if ann.class_id == class_id]

    def get_class_distribution(self) -> dict[str, int]:
        """Get count of each class in the image."""
        distribution = {}
        for ann in self.annotations:
            distribution[ann.class_name] = distribution.get(ann.class_name, 0) + 1
        return distribution


@dataclass
class DatasetStatistics:
    """Statistics for a dataset."""

    total_images: int = 0
    total_annotations: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    size_distribution: dict[str, int] = field(
        default_factory=lambda: {"small": 0, "medium": 0, "large": 0}
    )
    avg_annotations_per_image: float = 0.0
    min_annotations_per_image: int = 0
    max_annotations_per_image: int = 0

    def to_dict(self) -> dict:
        return {
            "total_images": self.total_images,
            "total_annotations": self.total_annotations,
            "class_distribution": self.class_distribution,
            "size_distribution": self.size_distribution,
            "avg_annotations_per_image": self.avg_annotations_per_image,
            "min_annotations_per_image": self.min_annotations_per_image,
            "max_annotations_per_image": self.max_annotations_per_image,
        }


class BaseDataset(ABC):
    """
    Abstract base class for ground truth datasets.

    All dataset loaders should inherit from this class and implement
    the abstract methods.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "val",
        vehicle_classes: Optional[dict[int, str]] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to dataset root directory
            split: Dataset split ('train', 'val', 'test')
            vehicle_classes: Mapping of class IDs to names (filter only these)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.vehicle_classes = vehicle_classes or {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }
        self.vehicle_class_ids = set(self.vehicle_classes.keys())

        # Size thresholds for object categorization (area in pixels^2)
        self.small_threshold = 32 * 32
        self.medium_threshold = 96 * 96

        # Cache
        self._annotations: Optional[list[ImageAnnotation]] = None
        self._statistics: Optional[DatasetStatistics] = None

    @abstractmethod
    def load_annotations(self) -> list[ImageAnnotation]:
        """
        Load all annotations from the dataset.

        Returns:
            List of ImageAnnotation objects
        """
        pass

    @abstractmethod
    def get_image(self, image_id: str) -> Optional[np.ndarray]:
        """
        Load image by ID.

        Args:
            image_id: Image identifier

        Returns:
            Image as numpy array (BGR format) or None if not found
        """
        pass

    @abstractmethod
    def get_image_path(self, image_id: str) -> Optional[str]:
        """
        Get the file path for an image.

        Args:
            image_id: Image identifier

        Returns:
            Path to image file or None if not found
        """
        pass

    @property
    def annotations(self) -> list[ImageAnnotation]:
        """Lazy-load and cache annotations."""
        if self._annotations is None:
            self._annotations = self.load_annotations()
        return self._annotations

    @property
    def statistics(self) -> DatasetStatistics:
        """Calculate and cache dataset statistics."""
        if self._statistics is None:
            self._statistics = self._calculate_statistics()
        return self._statistics

    def _calculate_statistics(self) -> DatasetStatistics:
        """Calculate dataset statistics."""
        stats = DatasetStatistics()
        stats.total_images = len(self.annotations)

        annotations_per_image = []
        for img_ann in self.annotations:
            annotations_per_image.append(img_ann.num_objects)

            for ann in img_ann.annotations:
                stats.total_annotations += 1

                # Class distribution
                stats.class_distribution[ann.class_name] = (
                    stats.class_distribution.get(ann.class_name, 0) + 1
                )

                # Size distribution
                area = ann.area
                if area < self.small_threshold:
                    stats.size_distribution["small"] += 1
                elif area < self.medium_threshold:
                    stats.size_distribution["medium"] += 1
                else:
                    stats.size_distribution["large"] += 1

        if annotations_per_image:
            stats.avg_annotations_per_image = np.mean(annotations_per_image)
            stats.min_annotations_per_image = min(annotations_per_image)
            stats.max_annotations_per_image = max(annotations_per_image)

        return stats

    def get_size_category(self, area: float) -> str:
        """Categorize object by area."""
        if area < self.small_threshold:
            return "small"
        elif area < self.medium_threshold:
            return "medium"
        return "large"

    def filter_by_class(self, class_ids: list[int]) -> list[ImageAnnotation]:
        """
        Filter annotations to only include specific classes.

        Args:
            class_ids: List of class IDs to keep

        Returns:
            Filtered list of ImageAnnotation objects
        """
        filtered = []
        class_id_set = set(class_ids)

        for img_ann in self.annotations:
            filtered_anns = [
                ann for ann in img_ann.annotations if ann.class_id in class_id_set
            ]
            if filtered_anns:
                new_img_ann = ImageAnnotation(
                    image_id=img_ann.image_id,
                    image_path=img_ann.image_path,
                    width=img_ann.width,
                    height=img_ann.height,
                    annotations=filtered_anns,
                    metadata=img_ann.metadata,
                )
                filtered.append(new_img_ann)

        return filtered

    def iter_images(
        self, batch_size: int = 1
    ) -> Generator[list[ImageAnnotation], None, None]:
        """
        Iterate over dataset in batches.

        Args:
            batch_size: Number of images per batch

        Yields:
            Batches of ImageAnnotation objects
        """
        batch = []
        for img_ann in self.annotations:
            batch.append(img_ann)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def sample(self, n: int, seed: Optional[int] = None) -> list[ImageAnnotation]:
        """
        Randomly sample n images from the dataset.

        Args:
            n: Number of images to sample
            seed: Random seed for reproducibility

        Returns:
            List of sampled ImageAnnotation objects
        """
        if seed is not None:
            np.random.seed(seed)

        indices = np.random.choice(
            len(self.annotations), size=min(n, len(self.annotations)), replace=False
        )
        return [self.annotations[i] for i in indices]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> ImageAnnotation:
        return self.annotations[idx]

    def __iter__(self):
        return iter(self.annotations)

    def print_statistics(self):
        """Print dataset statistics to console."""
        stats = self.statistics
        print(f"\n{'=' * 50}")
        print(f"Dataset Statistics: {self.__class__.__name__}")
        print(f"{'=' * 50}")
        print(f"Split: {self.split}")
        print(f"Total Images: {stats.total_images}")
        print(f"Total Annotations: {stats.total_annotations}")
        print(f"Avg Annotations/Image: {stats.avg_annotations_per_image:.2f}")
        print(
            f"Min/Max Annotations: {stats.min_annotations_per_image}/{stats.max_annotations_per_image}"
        )
        print(f"\nClass Distribution:")
        for class_name, count in sorted(stats.class_distribution.items()):
            percentage = (
                (count / stats.total_annotations) * 100
                if stats.total_annotations > 0
                else 0
            )
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        print(f"\nSize Distribution:")
        for size, count in stats.size_distribution.items():
            percentage = (
                (count / stats.total_annotations) * 100
                if stats.total_annotations > 0
                else 0
            )
            print(f"  {size}: {count} ({percentage:.1f}%)")
        print(f"{'=' * 50}\n")
