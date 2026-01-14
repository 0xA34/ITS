"""
Custom Dataset Loader
=====================
Loader for custom annotated datasets for vehicle detection baseline evaluation.

Supports multiple annotation formats:
    - COCO JSON format (recommended)
    - YOLO TXT format
    - Pascal VOC XML format

Directory Structure (COCO format):
    custom_dataset/
    ├── annotations.json
    └── images/
        └── *.jpg

Directory Structure (YOLO format):
    custom_dataset/
    ├── images/
    │   └── *.jpg
    └── labels/
        └── *.txt

Directory Structure (VOC format):
    custom_dataset/
    ├── Annotations/
    │   └── *.xml
    └── JPEGImages/
        └── *.jpg
"""

import json
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from app.baseline.datasets.base_dataset import (
    BaseDataset,
    BoundingBox,
    GroundTruthAnnotation,
    ImageAnnotation,
)

logger = logging.getLogger(__name__)


class CustomDataset(BaseDataset):
    """
    Custom dataset loader supporting multiple annotation formats.

    Usage:
        # COCO format
        dataset = CustomDataset(
            data_path="path/to/custom",
            annotation_format="coco"
        )

        # YOLO format
        dataset = CustomDataset(
            data_path="path/to/custom",
            annotation_format="yolo",
            class_names=["person", "bicycle", "car", "motorcycle", "bus", "truck"]
        )

        # VOC format
        dataset = CustomDataset(
            data_path="path/to/custom",
            annotation_format="voc"
        )

        for img_annotation in dataset:
            image = dataset.get_image(img_annotation.image_id)
            # Process image and annotations
    """

    SUPPORTED_FORMATS = ["coco", "yolo", "voc"]
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        data_path: str,
        split: str = "val",
        vehicle_classes: Optional[dict[int, str]] = None,
        annotation_format: str = "coco",
        annotation_file: Optional[str] = None,
        images_dir: Optional[str] = None,
        labels_dir: Optional[str] = None,
        class_names: Optional[list[str]] = None,
        filter_empty: bool = True,
    ):
        """
        Initialize custom dataset loader.

        Args:
            data_path: Path to dataset root
            split: Dataset split (used in naming)
            vehicle_classes: Custom vehicle class mapping (optional)
            annotation_format: Format of annotations ('coco', 'yolo', 'voc')
            annotation_file: Path to annotation file (for COCO format)
            images_dir: Path to images directory
            labels_dir: Path to labels directory (for YOLO format)
            class_names: List of class names (for YOLO format)
            filter_empty: Filter out images with no annotations
        """
        super().__init__(data_path, split, vehicle_classes)

        if annotation_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported annotation format: {annotation_format}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

        self.annotation_format = annotation_format
        self.filter_empty = filter_empty

        # Set default paths based on format
        if annotation_format == "coco":
            self.annotation_file = Path(
                annotation_file or self.data_path / "annotations.json"
            )
            self.images_dir = Path(images_dir or self.data_path / "images")
        elif annotation_format == "yolo":
            self.images_dir = Path(images_dir or self.data_path / "images")
            self.labels_dir = Path(labels_dir or self.data_path / "labels")
            self.class_names = class_names or list(self.vehicle_classes.values())
        elif annotation_format == "voc":
            self.images_dir = Path(images_dir or self.data_path / "JPEGImages")
            self.annotations_dir = Path(
                annotation_file or self.data_path / "Annotations"
            )

        # Image ID to path mapping
        self._image_id_to_path: dict[str, str] = {}

    def load_annotations(self) -> list[ImageAnnotation]:
        """Load annotations based on format."""
        if self.annotation_format == "coco":
            return self._load_coco_annotations()
        elif self.annotation_format == "yolo":
            return self._load_yolo_annotations()
        elif self.annotation_format == "voc":
            return self._load_voc_annotations()
        else:
            raise ValueError(f"Unknown format: {self.annotation_format}")

    def _load_coco_annotations(self) -> list[ImageAnnotation]:
        """Load COCO format annotations."""
        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_file}"
            )

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Build indices
        image_id_to_info = {img["id"]: img for img in coco_data.get("images", [])}
        image_id_to_anns = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in image_id_to_anns:
                image_id_to_anns[img_id] = []
            image_id_to_anns[img_id].append(ann)

        # Build category mapping
        categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

        # Process images
        annotations = []
        for img_id, img_info in image_id_to_info.items():
            file_name = img_info.get("file_name", "")
            image_path = str(self.images_dir / file_name)
            self._image_id_to_path[str(img_id)] = image_path

            # Get annotations
            coco_anns = image_id_to_anns.get(img_id, [])
            gt_annotations = []

            for ann in coco_anns:
                cat_id = ann.get("category_id")
                cat_name = categories.get(cat_id, "unknown")

                # Map to our class system
                class_id = None
                for vid, vname in self.vehicle_classes.items():
                    if vname.lower() == cat_name.lower():
                        class_id = vid
                        break

                if class_id is None:
                    continue

                bbox_data = ann.get("bbox", [0, 0, 0, 0])
                if len(bbox_data) != 4:
                    continue

                bbox = BoundingBox.from_coco(
                    x=bbox_data[0],
                    y=bbox_data[1],
                    w=bbox_data[2],
                    h=bbox_data[3],
                )

                gt_annotations.append(
                    GroundTruthAnnotation(
                        bbox=bbox,
                        class_id=class_id,
                        class_name=cat_name,
                        is_difficult=ann.get("iscrowd", 0) == 1,
                    )
                )

            if self.filter_empty and len(gt_annotations) == 0:
                continue

            annotations.append(
                ImageAnnotation(
                    image_id=str(img_id),
                    image_path=image_path,
                    width=img_info.get("width", 0),
                    height=img_info.get("height", 0),
                    annotations=gt_annotations,
                )
            )

        logger.info(f"Loaded {len(annotations)} images from COCO format")
        return annotations

    def _load_yolo_annotations(self) -> list[ImageAnnotation]:
        """Load YOLO format annotations."""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        annotations = []

        # Find all image files
        image_files = []
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
            image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))

        for img_path in sorted(image_files):
            image_id = img_path.stem
            self._image_id_to_path[image_id] = str(img_path)

            # Find corresponding label file
            label_path = self.labels_dir / f"{image_id}.txt"

            gt_annotations = []

            if label_path.exists():
                # Read image dimensions
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Failed to read image: {img_path}")
                    continue

                img_height, img_width = img.shape[:2]

                # Parse YOLO label file
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        class_idx = int(parts[0])
                        if class_idx >= len(self.class_names):
                            continue

                        class_name = self.class_names[class_idx]

                        # Map to our class system
                        class_id = None
                        for vid, vname in self.vehicle_classes.items():
                            if vname.lower() == class_name.lower():
                                class_id = vid
                                break

                        if class_id is None:
                            continue

                        # YOLO format: class x_center y_center width height (normalized)
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height

                        bbox = BoundingBox.from_xywh(x_center, y_center, width, height)

                        gt_annotations.append(
                            GroundTruthAnnotation(
                                bbox=bbox,
                                class_id=class_id,
                                class_name=class_name,
                            )
                        )

                if self.filter_empty and len(gt_annotations) == 0:
                    continue

                annotations.append(
                    ImageAnnotation(
                        image_id=image_id,
                        image_path=str(img_path),
                        width=img_width,
                        height=img_height,
                        annotations=gt_annotations,
                    )
                )

        logger.info(f"Loaded {len(annotations)} images from YOLO format")
        return annotations

    def _load_voc_annotations(self) -> list[ImageAnnotation]:
        """Load Pascal VOC format annotations."""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        if not self.annotations_dir.exists():
            raise FileNotFoundError(
                f"Annotations directory not found: {self.annotations_dir}"
            )

        annotations = []

        # Find all XML annotation files
        xml_files = list(self.annotations_dir.glob("*.xml"))

        for xml_path in sorted(xml_files):
            image_id = xml_path.stem

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except ET.ParseError as e:
                logger.warning(f"Failed to parse {xml_path}: {e}")
                continue

            # Get image info
            filename = root.findtext("filename", "")
            if not filename:
                filename = f"{image_id}.jpg"

            image_path = str(self.images_dir / filename)
            self._image_id_to_path[image_id] = image_path

            size = root.find("size")
            if size is not None:
                img_width = int(size.findtext("width", "0"))
                img_height = int(size.findtext("height", "0"))
            else:
                img_width = 0
                img_height = 0

            # Parse objects
            gt_annotations = []
            for obj in root.findall("object"):
                class_name = obj.findtext("name", "").lower()

                # Map to our class system
                class_id = None
                for vid, vname in self.vehicle_classes.items():
                    if vname.lower() == class_name:
                        class_id = vid
                        break

                if class_id is None:
                    continue

                # Check difficult flag
                is_difficult = obj.findtext("difficult", "0") == "1"
                is_truncated = obj.findtext("truncated", "0") == "1"

                # Parse bounding box
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                x1 = float(bndbox.findtext("xmin", "0"))
                y1 = float(bndbox.findtext("ymin", "0"))
                x2 = float(bndbox.findtext("xmax", "0"))
                y2 = float(bndbox.findtext("ymax", "0"))

                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

                gt_annotations.append(
                    GroundTruthAnnotation(
                        bbox=bbox,
                        class_id=class_id,
                        class_name=class_name,
                        is_difficult=is_difficult,
                        is_truncated=is_truncated,
                    )
                )

            if self.filter_empty and len(gt_annotations) == 0:
                continue

            annotations.append(
                ImageAnnotation(
                    image_id=image_id,
                    image_path=image_path,
                    width=img_width,
                    height=img_height,
                    annotations=gt_annotations,
                )
            )

        logger.info(f"Loaded {len(annotations)} images from VOC format")
        return annotations

    def get_image(self, image_id: str) -> Optional[np.ndarray]:
        """Load image by ID."""
        image_path = self.get_image_path(image_id)
        if image_path is None:
            return None

        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return None

        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None

        return image

    def get_image_path(self, image_id: str) -> Optional[str]:
        """Get image file path by ID."""
        # Load annotations to build path mapping if not done
        if not self._image_id_to_path:
            _ = self.annotations

        return self._image_id_to_path.get(image_id)


def create_empty_dataset(
    output_path: str,
    annotation_format: str = "coco",
    class_names: Optional[list[str]] = None,
):
    """
    Create an empty dataset structure for manual annotation.

    Args:
        output_path: Path to create dataset
        annotation_format: Format ('coco', 'yolo', 'voc')
        class_names: List of class names
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

    if annotation_format == "coco":
        # Create COCO structure
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)

        # Create empty annotation file
        coco_data = {
            "info": {
                "description": "Custom Vehicle Detection Dataset",
                "version": "1.0",
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i, "name": name} for i, name in enumerate(class_names)
            ],
        }

        with open(output_path / "annotations.json", "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)

    elif annotation_format == "yolo":
        # Create YOLO structure
        (output_path / "images").mkdir(exist_ok=True)
        (output_path / "labels").mkdir(exist_ok=True)

        # Create classes file
        with open(output_path / "classes.txt", "w") as f:
            f.write("\n".join(class_names))

    elif annotation_format == "voc":
        # Create VOC structure
        (output_path / "JPEGImages").mkdir(exist_ok=True)
        (output_path / "Annotations").mkdir(exist_ok=True)

    # Create README
    readme_content = f"""# Custom Vehicle Detection Dataset

## Format: {annotation_format.upper()}

## Classes:
{chr(10).join(f"- {i}: {name}" for i, name in enumerate(class_names))}

## Structure:
"""

    if annotation_format == "coco":
        readme_content += """
```
dataset/
├── annotations.json   # COCO format annotations
├── images/           # Image files (.jpg, .png)
└── README.md
```

## Annotation Format (COCO):
Each annotation in annotations.json should have:
- image_id: ID of the image
- category_id: Class ID
- bbox: [x, y, width, height] in pixels
- area: bbox area
- iscrowd: 0 or 1
"""
    elif annotation_format == "yolo":
        readme_content += """
```
dataset/
├── images/           # Image files (.jpg, .png)
├── labels/           # Label files (.txt)
├── classes.txt       # Class names
└── README.md
```

## Annotation Format (YOLO):
Each .txt file should contain lines with:
class_id x_center y_center width height

All values are normalized (0-1) relative to image dimensions.
"""
    elif annotation_format == "voc":
        readme_content += """
```
dataset/
├── JPEGImages/       # Image files (.jpg)
├── Annotations/      # XML annotation files
└── README.md
```

## Annotation Format (VOC XML):
See Pascal VOC format specification.
"""

    with open(output_path / "README.md", "w") as f:
        f.write(readme_content)

    logger.info(f"Created empty {annotation_format.upper()} dataset at {output_path}")
    return output_path


def convert_yolo_to_coco(
    yolo_path: str,
    output_path: str,
    class_names: list[str],
) -> str:
    """
    Convert YOLO format dataset to COCO format.

    Args:
        yolo_path: Path to YOLO dataset
        output_path: Path to save COCO dataset
        class_names: List of class names

    Returns:
        Path to output dataset
    """
    yolo_path = Path(yolo_path)
    output_path = Path(output_path)

    # Load YOLO dataset
    yolo_dataset = CustomDataset(
        data_path=str(yolo_path),
        annotation_format="yolo",
        class_names=class_names,
        filter_empty=False,
    )

    # Create COCO structure
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)

    coco_data = {
        "info": {"description": "Converted from YOLO format", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)],
    }

    ann_id = 1
    for img_ann in yolo_dataset.annotations:
        img_id = len(coco_data["images"]) + 1

        # Add image info
        coco_data["images"].append(
            {
                "id": img_id,
                "file_name": Path(img_ann.image_path).name,
                "width": img_ann.width,
                "height": img_ann.height,
            }
        )

        # Add annotations
        for gt_ann in img_ann.annotations:
            coco_data["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": gt_ann.class_id,
                    "bbox": [
                        gt_ann.bbox.x1,
                        gt_ann.bbox.y1,
                        gt_ann.bbox.width,
                        gt_ann.bbox.height,
                    ],
                    "area": gt_ann.area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        # Copy image
        import shutil

        src = Path(img_ann.image_path)
        dst = images_dir / src.name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # Save annotations
    with open(output_path / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    logger.info(f"Converted YOLO to COCO format at {output_path}")
    return str(output_path)
