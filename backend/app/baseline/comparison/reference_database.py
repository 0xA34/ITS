"""
Reference Database Module
=========================
Database of reference methods and official benchmarks for comparison.

Contains metrics from:
    - Official YOLO benchmarks (Ultralytics)
    - Academic papers on object detection
    - Popular detection frameworks (Detectron2, MMDetection)
    - Vehicle-specific detection benchmarks

Usage:
    from app.baseline.comparison import ReferenceDatabase

    # Get all reference methods
    references = ReferenceDatabase.get_all()

    # Get specific category
    yolo_refs = ReferenceDatabase.get_by_category("yolo")

    # Get methods suitable for real-time
    realtime_refs = ReferenceDatabase.get_realtime_methods(min_fps=30)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelCategory(str, Enum):
    """Categories of detection models."""

    YOLO = "yolo"
    RCNN = "rcnn"
    SSD = "ssd"
    TRANSFORMER = "transformer"
    ANCHOR_FREE = "anchor_free"
    LIGHTWEIGHT = "lightweight"
    OTHER = "other"


class DatasetType(str, Enum):
    """Common datasets for benchmarking."""

    COCO = "coco"
    PASCAL_VOC = "pascal_voc"
    KITTI = "kitti"
    UA_DETRAC = "ua_detrac"
    BDD100K = "bdd100k"
    CUSTOM = "custom"


@dataclass
class ReferenceMethod:
    """Reference method with benchmark metrics."""

    name: str
    category: ModelCategory

    # Accuracy metrics
    map_50: float = 0.0  # mAP@0.5
    map_75: float = 0.0  # mAP@0.75
    map_50_95: float = 0.0  # mAP@[.5:.95]

    # Speed metrics
    fps: float = 0.0  # Frames per second
    latency_ms: float = 0.0  # Inference latency in ms

    # Model complexity
    params_millions: float = 0.0  # Number of parameters (M)
    flops_billions: float = 0.0  # FLOPs (B)
    model_size_mb: float = 0.0  # Model file size (MB)

    # Metadata
    source: str = ""  # Paper/official source
    year: int = 0  # Publication year
    paper_url: str = ""  # Link to paper
    dataset: DatasetType = DatasetType.COCO
    input_size: int = 640  # Input image size
    device: str = "V100"  # GPU used for benchmarking
    batch_size: int = 1

    # Per-class AP (optional)
    ap_per_class: dict[str, float] = field(default_factory=dict)

    # Additional notes
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "accuracy": {
                "mAP@0.5": self.map_50,
                "mAP@0.75": self.map_75,
                "mAP@[.5:.95]": self.map_50_95,
            },
            "speed": {
                "fps": self.fps,
                "latency_ms": self.latency_ms,
            },
            "complexity": {
                "params_M": self.params_millions,
                "flops_B": self.flops_billions,
                "size_MB": self.model_size_mb,
            },
            "metadata": {
                "source": self.source,
                "year": self.year,
                "paper_url": self.paper_url,
                "dataset": self.dataset.value,
                "input_size": self.input_size,
                "device": self.device,
            },
            "notes": self.notes,
        }

    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (mAP per million parameters)."""
        if self.params_millions > 0:
            return self.map_50_95 / self.params_millions
        return 0.0

    @property
    def speed_accuracy_score(self) -> float:
        """Calculate speed-accuracy trade-off score."""
        if self.fps > 0:
            return self.map_50_95 * (self.fps / 100)
        return 0.0


class ReferenceDatabase:
    """
    Database of reference detection methods.

    Contains official benchmarks from papers and frameworks for comparison
    with evaluated models.
    """

    # =========================================================================
    # YOLO Family
    # =========================================================================
    YOLO_METHODS = [
        # YOLOv8 (Ultralytics, 2023)
        ReferenceMethod(
            name="YOLOv8n",
            category=ModelCategory.YOLO,
            map_50=0.370,
            map_75=0.228,
            map_50_95=0.183,
            fps=195,
            latency_ms=5.1,
            params_millions=3.2,
            flops_billions=8.7,
            model_size_mb=6.3,
            source="Ultralytics Official",
            year=2023,
            paper_url="https://github.com/ultralytics/ultralytics",
            input_size=640,
            device="V100",
            notes="Nano model, best for edge devices",
        ),
        ReferenceMethod(
            name="YOLOv8s",
            category=ModelCategory.YOLO,
            map_50=0.449,
            map_75=0.282,
            map_50_95=0.228,
            fps=142,
            latency_ms=7.0,
            params_millions=11.2,
            flops_billions=28.6,
            model_size_mb=22.5,
            source="Ultralytics Official",
            year=2023,
            paper_url="https://github.com/ultralytics/ultralytics",
            input_size=640,
            device="V100",
            notes="Small model, good balance",
        ),
        ReferenceMethod(
            name="YOLOv8m",
            category=ModelCategory.YOLO,
            map_50=0.502,
            map_75=0.327,
            map_50_95=0.257,
            fps=80,
            latency_ms=12.5,
            params_millions=25.9,
            flops_billions=78.9,
            model_size_mb=52.0,
            source="Ultralytics Official",
            year=2023,
            paper_url="https://github.com/ultralytics/ultralytics",
            input_size=640,
            device="V100",
            notes="Medium model",
        ),
        ReferenceMethod(
            name="YOLOv8l",
            category=ModelCategory.YOLO,
            map_50=0.528,
            map_75=0.350,
            map_50_95=0.273,
            fps=50,
            latency_ms=20.0,
            params_millions=43.7,
            flops_billions=165.2,
            model_size_mb=87.7,
            source="Ultralytics Official",
            year=2023,
            paper_url="https://github.com/ultralytics/ultralytics",
            input_size=640,
            device="V100",
            notes="Large model",
        ),
        ReferenceMethod(
            name="YOLOv8x",
            category=ModelCategory.YOLO,
            map_50=0.540,
            map_75=0.362,
            map_50_95=0.285,
            fps=35,
            latency_ms=28.6,
            params_millions=68.2,
            flops_billions=257.8,
            model_size_mb=136.7,
            source="Ultralytics Official",
            year=2023,
            paper_url="https://github.com/ultralytics/ultralytics",
            input_size=640,
            device="V100",
            notes="Extra-large model, highest accuracy",
        ),
        # YOLOv5 (Ultralytics, 2020)
        ReferenceMethod(
            name="YOLOv5n",
            category=ModelCategory.YOLO,
            map_50=0.280,
            map_75=0.175,
            map_50_95=0.140,
            fps=160,
            latency_ms=6.25,
            params_millions=1.9,
            flops_billions=4.5,
            model_size_mb=3.9,
            source="Ultralytics Official",
            year=2020,
            paper_url="https://github.com/ultralytics/yolov5",
            input_size=640,
            device="V100",
        ),
        ReferenceMethod(
            name="YOLOv5s",
            category=ModelCategory.YOLO,
            map_50=0.374,
            map_75=0.235,
            map_50_95=0.188,
            fps=118,
            latency_ms=8.5,
            params_millions=7.2,
            flops_billions=16.5,
            model_size_mb=14.4,
            source="Ultralytics Official",
            year=2020,
            paper_url="https://github.com/ultralytics/yolov5",
            input_size=640,
            device="V100",
        ),
        ReferenceMethod(
            name="YOLOv5m",
            category=ModelCategory.YOLO,
            map_50=0.453,
            map_75=0.293,
            map_50_95=0.231,
            fps=75,
            latency_ms=13.3,
            params_millions=21.2,
            flops_billions=49.0,
            model_size_mb=42.5,
            source="Ultralytics Official",
            year=2020,
            paper_url="https://github.com/ultralytics/yolov5",
            input_size=640,
            device="V100",
        ),
        ReferenceMethod(
            name="YOLOv5l",
            category=ModelCategory.YOLO,
            map_50=0.490,
            map_75=0.323,
            map_50_95=0.252,
            fps=50,
            latency_ms=20.0,
            params_millions=46.5,
            flops_billions=109.1,
            model_size_mb=93.4,
            source="Ultralytics Official",
            year=2020,
            paper_url="https://github.com/ultralytics/yolov5",
            input_size=640,
            device="V100",
        ),
        # YOLOv7 (Wang et al., 2022)
        ReferenceMethod(
            name="YOLOv7-tiny",
            category=ModelCategory.YOLO,
            map_50=0.380,
            map_75=0.240,
            map_50_95=0.191,
            fps=150,
            latency_ms=6.7,
            params_millions=6.2,
            flops_billions=13.7,
            model_size_mb=12.3,
            source="YOLOv7 Paper",
            year=2022,
            paper_url="https://arxiv.org/abs/2207.02696",
            input_size=640,
            device="V100",
        ),
        ReferenceMethod(
            name="YOLOv7",
            category=ModelCategory.YOLO,
            map_50=0.514,
            map_75=0.340,
            map_50_95=0.269,
            fps=55,
            latency_ms=18.2,
            params_millions=36.9,
            flops_billions=104.7,
            model_size_mb=74.8,
            source="YOLOv7 Paper",
            year=2022,
            paper_url="https://arxiv.org/abs/2207.02696",
            input_size=640,
            device="V100",
        ),
        ReferenceMethod(
            name="YOLOv7-X",
            category=ModelCategory.YOLO,
            map_50=0.530,
            map_75=0.355,
            map_50_95=0.281,
            fps=38,
            latency_ms=26.3,
            params_millions=71.3,
            flops_billions=189.9,
            model_size_mb=142.1,
            source="YOLOv7 Paper",
            year=2022,
            paper_url="https://arxiv.org/abs/2207.02696",
            input_size=640,
            device="V100",
        ),
        # YOLO-NAS (Deci AI, 2023)
        ReferenceMethod(
            name="YOLO-NAS-S",
            category=ModelCategory.YOLO,
            map_50=0.472,
            map_75=0.305,
            map_50_95=0.245,
            fps=125,
            latency_ms=8.0,
            params_millions=12.2,
            flops_billions=32.0,
            model_size_mb=24.0,
            source="Deci AI Official",
            year=2023,
            paper_url="https://github.com/Deci-AI/super-gradients",
            input_size=640,
            device="V100",
            notes="Neural Architecture Search optimized",
        ),
        ReferenceMethod(
            name="YOLO-NAS-M",
            category=ModelCategory.YOLO,
            map_50=0.512,
            map_75=0.338,
            map_50_95=0.267,
            fps=85,
            latency_ms=11.8,
            params_millions=31.9,
            flops_billions=82.0,
            model_size_mb=62.0,
            source="Deci AI Official",
            year=2023,
            paper_url="https://github.com/Deci-AI/super-gradients",
            input_size=640,
            device="V100",
        ),
        ReferenceMethod(
            name="YOLO-NAS-L",
            category=ModelCategory.YOLO,
            map_50=0.525,
            map_75=0.352,
            map_50_95=0.279,
            fps=60,
            latency_ms=16.7,
            params_millions=44.5,
            flops_billions=116.0,
            model_size_mb=87.0,
            source="Deci AI Official",
            year=2023,
            paper_url="https://github.com/Deci-AI/super-gradients",
            input_size=640,
            device="V100",
        ),
    ]

    # =========================================================================
    # R-CNN Family
    # =========================================================================
    RCNN_METHODS = [
        ReferenceMethod(
            name="Faster R-CNN (R50-FPN)",
            category=ModelCategory.RCNN,
            map_50=0.580,
            map_75=0.405,
            map_50_95=0.374,
            fps=15,
            latency_ms=66.7,
            params_millions=41.1,
            flops_billions=180.0,
            model_size_mb=167.0,
            source="Detectron2",
            year=2015,
            paper_url="https://arxiv.org/abs/1506.01497",
            input_size=800,
            device="V100",
            notes="ResNet-50 backbone with FPN",
        ),
        ReferenceMethod(
            name="Faster R-CNN (R101-FPN)",
            category=ModelCategory.RCNN,
            map_50=0.598,
            map_75=0.423,
            map_50_95=0.393,
            fps=10,
            latency_ms=100.0,
            params_millions=60.1,
            flops_billions=246.0,
            model_size_mb=237.0,
            source="Detectron2",
            year=2015,
            paper_url="https://arxiv.org/abs/1506.01497",
            input_size=800,
            device="V100",
            notes="ResNet-101 backbone with FPN",
        ),
        ReferenceMethod(
            name="Cascade R-CNN (R50-FPN)",
            category=ModelCategory.RCNN,
            map_50=0.605,
            map_75=0.443,
            map_50_95=0.406,
            fps=12,
            latency_ms=83.3,
            params_millions=69.2,
            flops_billions=235.0,
            model_size_mb=275.0,
            source="Detectron2",
            year=2018,
            paper_url="https://arxiv.org/abs/1712.00726",
            input_size=800,
            device="V100",
            notes="Multi-stage cascade detection",
        ),
        ReferenceMethod(
            name="Mask R-CNN (R50-FPN)",
            category=ModelCategory.RCNN,
            map_50=0.587,
            map_75=0.415,
            map_50_95=0.382,
            fps=13,
            latency_ms=76.9,
            params_millions=44.2,
            flops_billions=260.0,
            model_size_mb=178.0,
            source="Detectron2",
            year=2017,
            paper_url="https://arxiv.org/abs/1703.06870",
            input_size=800,
            device="V100",
            notes="Instance segmentation + detection",
        ),
    ]

    # =========================================================================
    # SSD Family
    # =========================================================================
    SSD_METHODS = [
        ReferenceMethod(
            name="SSD300 (VGG16)",
            category=ModelCategory.SSD,
            map_50=0.461,
            map_75=0.285,
            map_50_95=0.231,
            fps=46,
            latency_ms=21.7,
            params_millions=24.0,
            flops_billions=31.4,
            model_size_mb=96.0,
            source="Original Paper",
            year=2016,
            paper_url="https://arxiv.org/abs/1512.02325",
            input_size=300,
            device="V100",
        ),
        ReferenceMethod(
            name="SSD512 (VGG16)",
            category=ModelCategory.SSD,
            map_50=0.492,
            map_75=0.312,
            map_50_95=0.252,
            fps=22,
            latency_ms=45.5,
            params_millions=24.0,
            flops_billions=90.3,
            model_size_mb=96.0,
            source="Original Paper",
            year=2016,
            paper_url="https://arxiv.org/abs/1512.02325",
            input_size=512,
            device="V100",
        ),
        ReferenceMethod(
            name="SSDLite (MobileNetV3)",
            category=ModelCategory.SSD,
            map_50=0.345,
            map_75=0.195,
            map_50_95=0.168,
            fps=120,
            latency_ms=8.3,
            params_millions=3.4,
            flops_billions=0.6,
            model_size_mb=12.0,
            source="TorchVision",
            year=2019,
            paper_url="https://arxiv.org/abs/1905.02244",
            input_size=320,
            device="V100",
            notes="Mobile optimized",
        ),
    ]

    # =========================================================================
    # Transformer-based
    # =========================================================================
    TRANSFORMER_METHODS = [
        ReferenceMethod(
            name="DETR (R50)",
            category=ModelCategory.TRANSFORMER,
            map_50=0.620,
            map_75=0.445,
            map_50_95=0.420,
            fps=15,
            latency_ms=66.7,
            params_millions=41.3,
            flops_billions=86.0,
            model_size_mb=166.0,
            source="Facebook AI Research",
            year=2020,
            paper_url="https://arxiv.org/abs/2005.12872",
            input_size=800,
            device="V100",
            notes="End-to-end transformer detector",
        ),
        ReferenceMethod(
            name="DETR-DC5 (R50)",
            category=ModelCategory.TRANSFORMER,
            map_50=0.641,
            map_75=0.468,
            map_50_95=0.435,
            fps=10,
            latency_ms=100.0,
            params_millions=41.3,
            flops_billions=187.0,
            model_size_mb=166.0,
            source="Facebook AI Research",
            year=2020,
            paper_url="https://arxiv.org/abs/2005.12872",
            input_size=800,
            device="V100",
        ),
        ReferenceMethod(
            name="Deformable DETR",
            category=ModelCategory.TRANSFORMER,
            map_50=0.658,
            map_75=0.485,
            map_50_95=0.462,
            fps=19,
            latency_ms=52.6,
            params_millions=40.0,
            flops_billions=173.0,
            model_size_mb=160.0,
            source="Deformable DETR Paper",
            year=2021,
            paper_url="https://arxiv.org/abs/2010.04159",
            input_size=800,
            device="V100",
            notes="Multi-scale deformable attention",
        ),
        ReferenceMethod(
            name="RT-DETR-L",
            category=ModelCategory.TRANSFORMER,
            map_50=0.538,
            map_75=0.365,
            map_50_95=0.289,
            fps=72,
            latency_ms=13.9,
            params_millions=32.0,
            flops_billions=103.0,
            model_size_mb=64.0,
            source="Baidu Research",
            year=2023,
            paper_url="https://arxiv.org/abs/2304.08069",
            input_size=640,
            device="V100",
            notes="Real-time DETR",
        ),
        ReferenceMethod(
            name="RT-DETR-X",
            category=ModelCategory.TRANSFORMER,
            map_50=0.548,
            map_75=0.378,
            map_50_95=0.299,
            fps=55,
            latency_ms=18.2,
            params_millions=65.0,
            flops_billions=220.0,
            model_size_mb=130.0,
            source="Baidu Research",
            year=2023,
            paper_url="https://arxiv.org/abs/2304.08069",
            input_size=640,
            device="V100",
            notes="Real-time DETR Extra-Large",
        ),
    ]

    # =========================================================================
    # Anchor-Free Detectors
    # =========================================================================
    ANCHOR_FREE_METHODS = [
        ReferenceMethod(
            name="FCOS (R50-FPN)",
            category=ModelCategory.ANCHOR_FREE,
            map_50=0.574,
            map_75=0.403,
            map_50_95=0.386,
            fps=23,
            latency_ms=43.5,
            params_millions=32.1,
            flops_billions=180.0,
            model_size_mb=128.0,
            source="FCOS Paper",
            year=2019,
            paper_url="https://arxiv.org/abs/1904.01355",
            input_size=800,
            device="V100",
            notes="Fully Convolutional One-Stage",
        ),
        ReferenceMethod(
            name="CenterNet (R50)",
            category=ModelCategory.ANCHOR_FREE,
            map_50=0.556,
            map_75=0.385,
            map_50_95=0.378,
            fps=35,
            latency_ms=28.6,
            params_millions=32.0,
            flops_billions=115.0,
            model_size_mb=128.0,
            source="CenterNet Paper",
            year=2019,
            paper_url="https://arxiv.org/abs/1904.07850",
            input_size=512,
            device="V100",
            notes="Objects as Points",
        ),
        ReferenceMethod(
            name="CenterNet (Hourglass-104)",
            category=ModelCategory.ANCHOR_FREE,
            map_50=0.615,
            map_75=0.443,
            map_50_95=0.421,
            fps=7,
            latency_ms=142.9,
            params_millions=191.0,
            flops_billions=950.0,
            model_size_mb=764.0,
            source="CenterNet Paper",
            year=2019,
            paper_url="https://arxiv.org/abs/1904.07850",
            input_size=512,
            device="V100",
        ),
    ]

    # =========================================================================
    # Lightweight/Mobile Models
    # =========================================================================
    LIGHTWEIGHT_METHODS = [
        ReferenceMethod(
            name="MobileNetV3-SSD",
            category=ModelCategory.LIGHTWEIGHT,
            map_50=0.345,
            map_75=0.195,
            map_50_95=0.168,
            fps=120,
            latency_ms=8.3,
            params_millions=3.4,
            flops_billions=0.6,
            model_size_mb=12.0,
            source="TorchVision",
            year=2019,
            paper_url="https://arxiv.org/abs/1905.02244",
            input_size=320,
            device="V100",
        ),
        ReferenceMethod(
            name="EfficientDet-D0",
            category=ModelCategory.LIGHTWEIGHT,
            map_50=0.525,
            map_75=0.348,
            map_50_95=0.335,
            fps=55,
            latency_ms=18.2,
            params_millions=3.9,
            flops_billions=2.5,
            model_size_mb=15.9,
            source="EfficientDet Paper",
            year=2020,
            paper_url="https://arxiv.org/abs/1911.09070",
            input_size=512,
            device="V100",
        ),
        ReferenceMethod(
            name="EfficientDet-D1",
            category=ModelCategory.LIGHTWEIGHT,
            map_50=0.570,
            map_75=0.390,
            map_50_95=0.391,
            fps=35,
            latency_ms=28.6,
            params_millions=6.6,
            flops_billions=6.1,
            model_size_mb=26.8,
            source="EfficientDet Paper",
            year=2020,
            paper_url="https://arxiv.org/abs/1911.09070",
            input_size=640,
            device="V100",
        ),
        ReferenceMethod(
            name="NanoDet-Plus-m",
            category=ModelCategory.LIGHTWEIGHT,
            map_50=0.402,
            map_75=0.248,
            map_50_95=0.204,
            fps=190,
            latency_ms=5.3,
            params_millions=0.95,
            flops_billions=0.72,
            model_size_mb=3.8,
            source="NanoDet GitHub",
            year=2021,
            paper_url="https://github.com/RangiLyu/nanodet",
            input_size=320,
            device="V100",
            notes="Ultra-lightweight for edge devices",
        ),
    ]

    @classmethod
    def get_all(cls) -> list[ReferenceMethod]:
        """Get all reference methods."""
        all_methods = []
        all_methods.extend(cls.YOLO_METHODS)
        all_methods.extend(cls.RCNN_METHODS)
        all_methods.extend(cls.SSD_METHODS)
        all_methods.extend(cls.TRANSFORMER_METHODS)
        all_methods.extend(cls.ANCHOR_FREE_METHODS)
        all_methods.extend(cls.LIGHTWEIGHT_METHODS)
        return all_methods

    @classmethod
    def get_by_category(cls, category: ModelCategory) -> list[ReferenceMethod]:
        """Get reference methods by category."""
        category_map = {
            ModelCategory.YOLO: cls.YOLO_METHODS,
            ModelCategory.RCNN: cls.RCNN_METHODS,
            ModelCategory.SSD: cls.SSD_METHODS,
            ModelCategory.TRANSFORMER: cls.TRANSFORMER_METHODS,
            ModelCategory.ANCHOR_FREE: cls.ANCHOR_FREE_METHODS,
            ModelCategory.LIGHTWEIGHT: cls.LIGHTWEIGHT_METHODS,
        }
        return category_map.get(category, [])

    @classmethod
    def get_by_name(cls, name: str) -> Optional[ReferenceMethod]:
        """Get reference method by name."""
        for method in cls.get_all():
            if method.name.lower() == name.lower():
                return method
        return None

    @classmethod
    def get_realtime_methods(cls, min_fps: float = 30) -> list[ReferenceMethod]:
        """Get methods that achieve real-time performance."""
        return [m for m in cls.get_all() if m.fps >= min_fps]

    @classmethod
    def get_high_accuracy_methods(cls, min_map: float = 0.5) -> list[ReferenceMethod]:
        """Get methods with high accuracy."""
        return [m for m in cls.get_all() if m.map_50 >= min_map]

    @classmethod
    def get_lightweight_methods(cls, max_params: float = 10.0) -> list[ReferenceMethod]:
        """Get lightweight methods (< max_params million parameters)."""
        return [m for m in cls.get_all() if m.params_millions <= max_params]

    @classmethod
    def get_methods_by_year(cls, min_year: int = 2020) -> list[ReferenceMethod]:
        """Get methods from a specific year onwards."""
        return [m for m in cls.get_all() if m.year >= min_year]

    @classmethod
    def get_pareto_optimal(cls) -> list[ReferenceMethod]:
        """
        Get Pareto-optimal methods (best accuracy-speed trade-off).

        A method is Pareto-optimal if no other method is both faster
        and more accurate.
        """
        all_methods = cls.get_all()
        pareto_methods = []

        for method in all_methods:
            is_dominated = False
            for other in all_methods:
                if other.name == method.name:
                    continue
                # Check if 'other' dominates 'method'
                if other.map_50 >= method.map_50 and other.fps > method.fps:
                    is_dominated = True
                    break
                if other.map_50 > method.map_50 and other.fps >= method.fps:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_methods.append(method)

        return pareto_methods

    @classmethod
    def to_dict(cls) -> dict[str, list[dict]]:
        """Export all references as dictionary."""
        return {
            "yolo": [m.to_dict() for m in cls.YOLO_METHODS],
            "rcnn": [m.to_dict() for m in cls.RCNN_METHODS],
            "ssd": [m.to_dict() for m in cls.SSD_METHODS],
            "transformer": [m.to_dict() for m in cls.TRANSFORMER_METHODS],
            "anchor_free": [m.to_dict() for m in cls.ANCHOR_FREE_METHODS],
            "lightweight": [m.to_dict() for m in cls.LIGHTWEIGHT_METHODS],
        }

    @classmethod
    def print_summary(cls):
        """Print summary of all reference methods."""
        print("\n" + "=" * 80)
        print("REFERENCE DATABASE SUMMARY")
        print("=" * 80)

        categories = [
            ("YOLO Family", cls.YOLO_METHODS),
            ("R-CNN Family", cls.RCNN_METHODS),
            ("SSD Family", cls.SSD_METHODS),
            ("Transformer-based", cls.TRANSFORMER_METHODS),
            ("Anchor-Free", cls.ANCHOR_FREE_METHODS),
            ("Lightweight/Mobile", cls.LIGHTWEIGHT_METHODS),
        ]

        for cat_name, methods in categories:
            print(f"\n{cat_name}: {len(methods)} methods")
            print("-" * 60)
            print(f"{'Name':<25} {'mAP@0.5':>10} {'FPS':>10} {'Params(M)':>12}")
            print("-" * 60)

            for m in sorted(methods, key=lambda x: x.map_50, reverse=True):
                print(
                    f"{m.name:<25} {m.map_50:>10.3f} {m.fps:>10.1f} {m.params_millions:>12.1f}"
                )

        print("\n" + "=" * 80)
        print(f"Total methods: {len(cls.get_all())}")
        print("=" * 80)
