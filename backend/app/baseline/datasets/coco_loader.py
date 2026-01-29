"""
COCO Dataset Loader
===================
Loader for COCO format datasets for vehicle detection baseline evaluation.

COCO Format:
    - annotations/instances_{split}.json: Annotation file
    - {split}2017/: Image directory

Vehicle Classes in COCO:
    - 0: person
    - 1: bicycle
    - 2: car
    - 3: motorcycle
    - 5: bus
    - 7: truck
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from app.baseline.datasets.base_dataset import (
    BaseDataset,
    BoundingBox,
    GroundTruthAnnotation,
    ImageAnnotation,
)

logger = logging.getLogger(__name__)


# COCO category ID to our vehicle class ID mapping
COCO_VEHICLE_CATEGORIES = {
    1: {"id": 0, "name": "person"},
    2: {"id": 1, "name": "bicycle"},
    3: {"id": 2, "name": "car"},
    4: {"id": 3, "name": "motorcycle"},
    6: {"id": 5, "name": "bus"},
    8: {"id": 7, "name": "truck"},
}

COCO_VEHICLE_CATEGORY_IDS = set(COCO_VEHICLE_CATEGORIES.keys())


class COCODataset(BaseDataset):
    """
    COCO format dataset loader.

    Expected directory structure:
        data_path/
        ├── annotations/
        │   ├── instances_train2017.json
        │   ├── instances_val2017.json
        │   └── instances_test2017.json
        ├── train2017/
        │   └── *.jpg
        ├── val2017/
        │   └── *.jpg
        └── test2017/
            └── *.jpg

    Usage:
        dataset = COCODataset(data_path="path/to/coco", split="val")
        for img_annotation in dataset:
            image = dataset.get_image(img_annotation.image_id)
            # Process image and annotations
    """

    def __init__(
        self,
        data_path: str,
        split: str = "val",
        vehicle_classes: Optional[dict[int, str]] = None,
        year: str = "2017",
        filter_empty: bool = True,
    ):
        """
        Initialize COCO dataset loader.

        Args:
            data_path: Path to COCO dataset root
            split: Dataset split ('train', 'val', 'test')
            vehicle_classes: Custom vehicle class mapping (optional)
            year: Dataset year (default: '2017')
            filter_empty: Filter out images with no vehicle annotations
        """
        super().__init__(data_path, split, vehicle_classes)
        self.year = year
        self.filter_empty = filter_empty

        # Paths
        self.annotation_file = (
            self.data_path / "annotations" / f"instances_{split}{year}.json"
        )
        self.images_dir = self.data_path / f"{split}{year}"

        # COCO data cache
        self._coco_data: Optional[dict] = None
        self._image_id_to_info: dict[int, dict] = {}
        self._image_id_to_anns: dict[int, list[dict]] = {}

    def _load_coco_json(self) -> dict:
        """Load COCO annotation JSON file."""
        if self._coco_data is not None:
            return self._coco_data

        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"COCO annotation file not found: {self.annotation_file}\n"
                f"Please download COCO dataset and place it in: {self.data_path}"
            )

        logger.info(f"Loading COCO annotations from {self.annotation_file}")
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            self._coco_data = json.load(f)

        # Build image index
        for img_info in self._coco_data.get("images", []):
            self._image_id_to_info[img_info["id"]] = img_info

        # Build annotation index
        for ann in self._coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self._image_id_to_anns:
                self._image_id_to_anns[img_id] = []
            self._image_id_to_anns[img_id].append(ann)

        logger.info(
            f"Loaded {len(self._image_id_to_info)} images, "
            f"{len(self._coco_data.get('annotations', []))} annotations"
        )

        return self._coco_data

    def _convert_coco_annotation(self, ann: dict) -> Optional[GroundTruthAnnotation]:
        """Convert COCO annotation to our format."""
        category_id = ann.get("category_id")

        # Filter non-vehicle categories
        if category_id not in COCO_VEHICLE_CATEGORIES:
            return None

        class_info = COCO_VEHICLE_CATEGORIES[category_id]

        # Filter by our vehicle classes
        if class_info["id"] not in self.vehicle_class_ids:
            return None

        # COCO bbox format: [x, y, width, height]
        bbox_data = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox_data) != 4 or bbox_data[2] <= 0 or bbox_data[3] <= 0:
            return None

        bbox = BoundingBox.from_coco(
            x=bbox_data[0],
            y=bbox_data[1],
            w=bbox_data[2],
            h=bbox_data[3],
        )

        return GroundTruthAnnotation(
            bbox=bbox,
            class_id=class_info["id"],
            class_name=class_info["name"],
            confidence=1.0,
            is_difficult=ann.get("iscrowd", 0) == 1,
            is_truncated=False,
        )

    def load_annotations(self) -> list[ImageAnnotation]:
        """Load all annotations from COCO dataset."""
        coco_data = self._load_coco_json()
        annotations = []

        for img_id, img_info in self._image_id_to_info.items():
            # Get all annotations for this image
            coco_anns = self._image_id_to_anns.get(img_id, [])

            # Convert to our format
            gt_annotations = []
            for ann in coco_anns:
                gt_ann = self._convert_coco_annotation(ann)
                if gt_ann is not None:
                    gt_annotations.append(gt_ann)

            # Filter empty images if requested
            if self.filter_empty and len(gt_annotations) == 0:
                continue

            # Build image path
            file_name = img_info.get("file_name", "")
            image_path = str(self.images_dir / file_name)

            img_annotation = ImageAnnotation(
                image_id=str(img_id),
                image_path=image_path,
                width=img_info.get("width", 0),
                height=img_info.get("height", 0),
                annotations=gt_annotations,
                metadata={
                    "coco_url": img_info.get("coco_url", ""),
                    "flickr_url": img_info.get("flickr_url", ""),
                    "date_captured": img_info.get("date_captured", ""),
                },
            )
            annotations.append(img_annotation)

        logger.info(
            f"Loaded {len(annotations)} images with vehicle annotations "
            f"({sum(len(a.annotations) for a in annotations)} total annotations)"
        )

        return annotations

    def get_image(self, image_id: str) -> Optional[np.ndarray]:
        """Load image by ID."""
        image_path = self.get_image_path(image_id)
        if image_path is None or not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return None

        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None

        return image

    def get_image_path(self, image_id: str) -> Optional[str]:
        """Get image file path by ID."""
        # Ensure COCO data is loaded
        self._load_coco_json()

        img_id = int(image_id)
        img_info = self._image_id_to_info.get(img_id)
        if img_info is None:
            return None

        file_name = img_info.get("file_name", "")
        return str(self.images_dir / file_name)

    def get_category_mapping(self) -> dict[int, dict[str, Any]]:
        """Get COCO category to our class mapping."""
        return COCO_VEHICLE_CATEGORIES.copy()

    def download_dataset(self, target_path: Optional[str] = None):
        """
        Print instructions for downloading COCO dataset.

        Args:
            target_path: Target path for download (optional)
        """
        path = target_path or str(self.data_path)
        print(f"""
{"=" * 60}
COCO Dataset Download Instructions
{"=" * 60}

1. Download annotations:
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

2. Download images:
   wget http://images.cocodataset.org/zips/val2017.zip
   wget http://images.cocodataset.org/zips/train2017.zip

3. Extract to: {path}
   unzip annotations_trainval2017.zip -d {path}
   unzip val2017.zip -d {path}
   unzip train2017.zip -d {path}

Expected structure:
   {path}/
   ├── annotations/
   │   ├── instances_train2017.json
   │   └── instances_val2017.json
   ├── train2017/
   │   └── *.jpg
   └── val2017/
       └── *.jpg

For vehicle detection baseline, we recommend using val2017 split
which contains ~5000 images.
{"=" * 60}
        """)


def create_subset(
    source_dataset: COCODataset,
    output_path: str,
    num_images: int = 500,
    seed: int = 42,
    copy_images: bool = True,
):
    """
    Create a smaller subset of COCO dataset for quick testing.

    Args:
        source_dataset: Source COCODataset instance
        output_path: Path to save subset
        num_images: Number of images in subset
        seed: Random seed
        copy_images: Whether to copy image files
    """
    import shutil

    np.random.seed(seed)
    output_path = Path(output_path)

    # Sample images
    sampled = source_dataset.sample(num_images, seed=seed)
    logger.info(f"Creating subset with {len(sampled)} images")

    # Create directory structure
    annotations_dir = output_path / "annotations"
    images_dir = output_path / f"{source_dataset.split}{source_dataset.year}"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Build new COCO format annotation
    new_coco = {
        "info": {
            "description": f"COCO Subset - {num_images} images for vehicle detection baseline",
            "version": "1.0",
            "year": source_dataset.year,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": coco_id, "name": info["name"]}
            for coco_id, info in COCO_VEHICLE_CATEGORIES.items()
        ],
    }

    # Load original COCO data
    source_dataset._load_coco_json()

    ann_id = 1
    for img_ann in sampled:
        img_id = int(img_ann.image_id)
        img_info = source_dataset._image_id_to_info.get(img_id, {})

        # Add image info
        new_coco["images"].append(
            {
                "id": img_id,
                "file_name": img_info.get("file_name", ""),
                "width": img_ann.width,
                "height": img_ann.height,
            }
        )

        # Add annotations
        for gt_ann in img_ann.annotations:
            # Find original COCO category ID
            coco_cat_id = None
            for coco_id, info in COCO_VEHICLE_CATEGORIES.items():
                if info["id"] == gt_ann.class_id:
                    coco_cat_id = coco_id
                    break

            if coco_cat_id is None:
                continue

            new_coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": coco_cat_id,
                    "bbox": [
                        gt_ann.bbox.x1,
                        gt_ann.bbox.y1,
                        gt_ann.bbox.width,
                        gt_ann.bbox.height,
                    ],
                    "area": gt_ann.area,
                    "iscrowd": 1 if gt_ann.is_difficult else 0,
                }
            )
            ann_id += 1

        # Copy image file
        if copy_images:
            src_path = Path(img_ann.image_path)
            dst_path = images_dir / src_path.name
            if src_path.exists() and not dst_path.exists():
                shutil.copy2(src_path, dst_path)

    # Save annotations
    ann_file = (
        annotations_dir / f"instances_{source_dataset.split}{source_dataset.year}.json"
    )
    with open(ann_file, "w", encoding="utf-8") as f:
        json.dump(new_coco, f, indent=2)

    logger.info(f"Subset saved to {output_path}")
    logger.info(f"  - Images: {len(new_coco['images'])}")
    logger.info(f"  - Annotations: {len(new_coco['annotations'])}")

    return output_path
