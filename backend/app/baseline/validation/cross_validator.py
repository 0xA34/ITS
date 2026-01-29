"""
Cross Validator Module
======================
Run multiple evaluation rounds with different random seeds to ensure
statistical reliability of baseline results.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result from a single evaluation run."""

    run_id: int
    seed: int
    metrics: dict[str, float]
    per_class_ap: dict[str, float]
    speed: dict[str, float]
    num_images: int
    num_predictions: int
    evaluation_time: float
    timestamp: str


@dataclass
class CrossValidationResult:
    """Aggregated results from cross-validation runs."""

    model_name: str
    model_path: str
    dataset_name: str
    num_runs: int
    runs: list[RunResult] = field(default_factory=list)

    # Aggregated metrics (computed after all runs)
    mean_metrics: dict[str, float] = field(default_factory=dict)
    std_metrics: dict[str, float] = field(default_factory=dict)
    min_metrics: dict[str, float] = field(default_factory=dict)
    max_metrics: dict[str, float] = field(default_factory=dict)

    # Per-class aggregated
    mean_per_class_ap: dict[str, float] = field(default_factory=dict)
    std_per_class_ap: dict[str, float] = field(default_factory=dict)

    # Speed aggregated
    mean_speed: dict[str, float] = field(default_factory=dict)
    std_speed: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model": {
                "name": self.model_name,
                "path": self.model_path,
            },
            "dataset": self.dataset_name,
            "num_runs": self.num_runs,
            "aggregated": {
                "metrics": {
                    "mean": self.mean_metrics,
                    "std": self.std_metrics,
                    "min": self.min_metrics,
                    "max": self.max_metrics,
                },
                "per_class_ap": {
                    "mean": self.mean_per_class_ap,
                    "std": self.std_per_class_ap,
                },
                "speed": {
                    "mean": self.mean_speed,
                    "std": self.std_speed,
                },
            },
            "runs": [
                {
                    "run_id": r.run_id,
                    "seed": r.seed,
                    "metrics": r.metrics,
                    "per_class_ap": r.per_class_ap,
                    "speed": r.speed,
                    "num_images": r.num_images,
                    "num_predictions": r.num_predictions,
                    "evaluation_time": r.evaluation_time,
                    "timestamp": r.timestamp,
                }
                for r in self.runs
            ],
        }

    def save(self, output_path: str):
        """Save results to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved cross-validation results to {path}")


class CrossValidator:
    """
    Cross-validator for running multiple evaluation rounds.

    This class runs the baseline evaluation multiple times with different
    random seeds to measure the variance in results and ensure statistical
    reliability.
    """

    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        dataset_format: str = "coco",
        device: str = "auto",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: tuple[int, int] = (640, 640),
        max_images: Optional[int] = None,
    ):
        """
        Initialize CrossValidator.

        Args:
            model_path: Path to the model file (.pt)
            dataset_path: Path to the dataset directory
            dataset_format: Dataset format ('coco', 'yolo', 'voc')
            device: Device to run on ('cuda', 'cpu', 'auto')
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            image_size: Input image size (width, height)
            max_images: Maximum number of images to evaluate (None for all)
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.dataset_format = dataset_format
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.max_images = max_images

        # Extract model name from path
        self.model_name = Path(model_path).stem

    def run(
        self,
        num_runs: int = 5,
        seeds: Optional[list[int]] = None,
        sample_ratio: float = 1.0,
        verbose: bool = True,
    ) -> CrossValidationResult:
        """
        Run cross-validation with multiple evaluation rounds.

        Args:
            num_runs: Number of evaluation runs
            seeds: List of random seeds (if None, generates automatically)
            sample_ratio: Ratio of dataset to sample for each run (0.0-1.0)
            verbose: Print progress information

        Returns:
            CrossValidationResult with aggregated statistics
        """
        from datetime import datetime

        # Generate seeds if not provided
        if seeds is None:
            base_seed = int(time.time()) % 10000
            seeds = [base_seed + i * 1000 for i in range(num_runs)]

        if len(seeds) < num_runs:
            raise ValueError(f"Not enough seeds provided: {len(seeds)} < {num_runs}")

        result = CrossValidationResult(
            model_name=self.model_name,
            model_path=self.model_path,
            dataset_name=self.dataset_format,
            num_runs=num_runs,
        )

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Cross-Validation: {self.model_name}")
            print(f"{'=' * 60}")
            print(f"Number of runs: {num_runs}")
            print(f"Sample ratio: {sample_ratio:.1%}")
            print(f"Seeds: {seeds[:num_runs]}")
            print(f"{'=' * 60}\n")

        # Run evaluations
        for i in range(num_runs):
            seed = seeds[i]
            if verbose:
                print(f"\n[Run {i + 1}/{num_runs}] Seed: {seed}")
                print("-" * 40)

            run_result = self._single_run(
                run_id=i + 1,
                seed=seed,
                sample_ratio=sample_ratio,
                verbose=verbose,
            )
            result.runs.append(run_result)

            if verbose:
                print(f"  mAP@0.5: {run_result.metrics.get('mAP@0.5', 0):.4f}")
                print(
                    f"  mAP@0.5:0.95: {run_result.metrics.get('mAP@[.5:.95]', 0):.4f}"
                )
                print(f"  FPS: {run_result.speed.get('fps', 0):.2f}")
                print(f"  Time: {run_result.evaluation_time:.2f}s")

        # Aggregate results
        self._aggregate_results(result)

        if verbose:
            self._print_summary(result)

        return result

    def _single_run(
        self,
        run_id: int,
        seed: int,
        sample_ratio: float,
        verbose: bool,
    ) -> RunResult:
        """Execute a single evaluation run."""
        from datetime import datetime

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package is required. Install with: pip install ultralytics"
            )

        # Set random seed
        np.random.seed(seed)

        start_time = time.time()

        # Load model
        model = YOLO(self.model_path)

        # Load dataset
        dataset = self._load_dataset()

        # Sample dataset if needed
        if sample_ratio < 1.0:
            num_samples = max(1, int(len(dataset) * sample_ratio))
            indices = np.random.choice(len(dataset), size=num_samples, replace=False)
            sampled_annotations = [dataset[i] for i in indices]
        else:
            sampled_annotations = list(dataset)

        if self.max_images is not None:
            sampled_annotations = sampled_annotations[: self.max_images]

        # Run evaluation
        metrics, per_class_ap, speed, num_predictions = self._evaluate(
            model, sampled_annotations, verbose
        )

        evaluation_time = time.time() - start_time

        return RunResult(
            run_id=run_id,
            seed=seed,
            metrics=metrics,
            per_class_ap=per_class_ap,
            speed=speed,
            num_images=len(sampled_annotations),
            num_predictions=num_predictions,
            evaluation_time=evaluation_time,
            timestamp=datetime.now().isoformat(),
        )

    def _load_dataset(self):
        """Load the dataset based on format."""
        if self.dataset_format == "coco":
            from app.baseline.datasets import COCODataset

            return COCODataset(data_path=self.dataset_path)
        elif self.dataset_format == "custom":
            from app.baseline.datasets import CustomDataset

            return CustomDataset(data_path=self.dataset_path)
        else:
            raise ValueError(f"Unsupported dataset format: {self.dataset_format}")

    def _evaluate(
        self,
        model,
        annotations: list,
        verbose: bool,
    ) -> tuple[dict, dict, dict, int]:
        """
        Run evaluation on the given annotations.

        Returns:
            Tuple of (metrics, per_class_ap, speed, num_predictions)
        """
        from app.baseline.metrics import MAPCalculator
        from app.baseline.metrics.fps_benchmark import benchmark_model

        # Initialize MAP calculator
        vehicle_classes = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }
        map_calculator = MAPCalculator(class_names=vehicle_classes)

        total_predictions = 0

        # Process each image
        for img_ann in annotations:
            # Load image
            image = self._load_image(img_ann.image_path)
            if image is None:
                continue

            # Run inference
            results = model.predict(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size[0],
                verbose=False,
            )

            # Process detections
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls.item())
                        if cls_id in vehicle_classes:
                            bbox = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf.item())
                            map_calculator.add_prediction(
                                bbox=tuple(bbox),
                                class_id=cls_id,
                                confidence=conf,
                                image_id=img_ann.image_id,
                            )
                            total_predictions += 1

            # Add ground truths
            for gt in img_ann.annotations:
                if gt.class_id in vehicle_classes:
                    map_calculator.add_ground_truth(
                        bbox=gt.bbox.to_xyxy(),
                        class_id=gt.class_id,
                        image_id=img_ann.image_id,
                        is_difficult=gt.is_difficult,
                    )

        # Calculate metrics
        map_result = map_calculator.calculate_coco_map()

        # Calculate Precision/Recall/F1 at IoU=0.5
        from app.baseline.metrics.precision_recall import calculate_precision_recall

        pr_result = calculate_precision_recall(
            predictions=map_calculator.predictions,
            ground_truths=map_calculator.ground_truths,
            iou_threshold=0.5,
        )

        metrics = {
            "mAP@0.5": map_result.map_50,
            "mAP@0.75": map_result.map_75,
            "mAP@[.5:.95]": map_result.map_50_95,
            "precision": pr_result.precision,
            "recall": pr_result.recall,
            "f1_score": pr_result.f1_score,
        }

        per_class_ap = map_result.per_class_ap

        # Benchmark speed
        speed_result = benchmark_model(
            model_path=self.model_path,
            num_iterations=50,
            image_size=self.image_size,
            device=self.device if self.device != "auto" else None,
        )

        speed = {
            "fps": speed_result.get("fps", 0),
            "latency_ms": speed_result.get("latency_ms", 0),
            "latency_std_ms": speed_result.get("latency_std_ms", 0),
        }

        return metrics, per_class_ap, speed, total_predictions

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from path."""
        import cv2

        try:
            image = cv2.imread(image_path)
            return image
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None

    def _aggregate_results(self, result: CrossValidationResult):
        """Aggregate results from all runs."""
        if not result.runs:
            return

        # Collect all metrics
        all_metrics: dict[str, list[float]] = {}
        all_per_class_ap: dict[str, list[float]] = {}
        all_speed: dict[str, list[float]] = {}

        for run in result.runs:
            for key, value in run.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

            for key, value in run.per_class_ap.items():
                if key not in all_per_class_ap:
                    all_per_class_ap[key] = []
                all_per_class_ap[key].append(value)

            for key, value in run.speed.items():
                if key not in all_speed:
                    all_speed[key] = []
                all_speed[key].append(value)

        # Calculate statistics
        for key, values in all_metrics.items():
            result.mean_metrics[key] = float(np.mean(values))
            result.std_metrics[key] = float(np.std(values))
            result.min_metrics[key] = float(np.min(values))
            result.max_metrics[key] = float(np.max(values))

        for key, values in all_per_class_ap.items():
            result.mean_per_class_ap[key] = float(np.mean(values))
            result.std_per_class_ap[key] = float(np.std(values))

        for key, values in all_speed.items():
            result.mean_speed[key] = float(np.mean(values))
            result.std_speed[key] = float(np.std(values))

    def _print_summary(self, result: CrossValidationResult):
        """Print summary of cross-validation results."""
        print(f"\n{'=' * 60}")
        print(f"Cross-Validation Summary: {result.model_name}")
        print(f"{'=' * 60}")
        print(f"Number of runs: {result.num_runs}")
        print(f"\nAggregated Metrics (mean ± std):")
        print("-" * 40)

        for key in [
            "mAP@0.5",
            "mAP@0.75",
            "mAP@[.5:.95]",
            "precision",
            "recall",
            "f1_score",
        ]:
            if key in result.mean_metrics:
                mean = result.mean_metrics[key]
                std = result.std_metrics[key]
                print(f"  {key}: {mean:.4f} ± {std:.4f}")

        print(f"\nSpeed (mean ± std):")
        print("-" * 40)
        for key in ["fps", "latency_ms"]:
            if key in result.mean_speed:
                mean = result.mean_speed[key]
                std = result.std_speed[key]
                unit = "fps" if key == "fps" else "ms"
                print(f"  {key}: {mean:.2f} ± {std:.2f} {unit}")

        print(f"\nPer-class AP (mean ± std):")
        print("-" * 40)
        for class_name in sorted(result.mean_per_class_ap.keys()):
            mean = result.mean_per_class_ap[class_name]
            std = result.std_per_class_ap[class_name]
            print(f"  {class_name}: {mean:.4f} ± {std:.4f}")

        print(f"{'=' * 60}\n")


def run_cross_validation(
    model_path: str,
    dataset_path: str,
    num_runs: int = 5,
    output_path: Optional[str] = None,
    **kwargs,
) -> CrossValidationResult:
    """
    Convenience function to run cross-validation.

    Args:
        model_path: Path to model file
        dataset_path: Path to dataset
        num_runs: Number of validation runs
        output_path: Path to save results (optional)
        **kwargs: Additional arguments for CrossValidator

    Returns:
        CrossValidationResult
    """
    validator = CrossValidator(
        model_path=model_path,
        dataset_path=dataset_path,
        **kwargs,
    )

    result = validator.run(num_runs=num_runs)

    if output_path:
        result.save(output_path)

    return result
