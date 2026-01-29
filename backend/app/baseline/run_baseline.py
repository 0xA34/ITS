"""
Baseline Evaluation Runner
===========================
Main script to run baseline evaluation for the Vehicle Detection & Monitoring Survey.

This script provides:
    1. Ground truth dataset loading and management
    2. Model evaluation with multiple metrics (mAP, Precision, Recall, F1)
    3. FPS/Latency benchmarking
    4. Comparison with reference methods
    5. Report generation (Markdown, HTML, JSON)

Usage:
    # Run full baseline evaluation
    python -m app.baseline.run_baseline

    # Run with specific options
    python -m app.baseline.run_baseline --model yolov8n.pt --dataset coco --output results/

    # Run only benchmarking
    python -m app.baseline.run_baseline --benchmark-only

    # Generate comparison report only
    python -m app.baseline.run_baseline --report-only

Example:
    from app.baseline.run_baseline import BaselineEvaluator

    evaluator = BaselineEvaluator()
    results = evaluator.run_full_evaluation(
        model_path="yolov8n.pt",
        dataset_path="data/coco",
    )
    evaluator.generate_report(results, output_path="results/baseline_report")
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for baseline evaluation."""

    # Model settings
    model_path: str = "yolov8n.pt"
    model_name: Optional[str] = None
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "auto"  # 'cuda', 'cpu', or 'auto'

    # Dataset settings
    dataset_path: str = ""
    dataset_format: str = "coco"  # 'coco', 'yolo', 'voc', 'custom'
    dataset_split: str = "val"
    max_images: Optional[int] = None  # Limit number of images for quick testing

    # Evaluation settings
    iou_thresholds: list[float] = field(
        default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    evaluate_per_class: bool = True
    evaluate_per_size: bool = True

    # Benchmark settings
    benchmark_iterations: int = 100
    benchmark_warmup: int = 10
    image_size: tuple[int, int] = (640, 640)

    # Output settings
    output_dir: str = "results/baseline"
    report_formats: list[str] = field(
        default_factory=lambda: ["json", "markdown", "html"]
    )
    save_predictions: bool = False
    save_visualizations: bool = False


@dataclass
class EvaluationResults:
    """Results from baseline evaluation."""

    # Model info
    model_name: str = ""
    model_path: str = ""

    # Accuracy metrics
    map_50: float = 0.0
    map_75: float = 0.0
    map_50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Per-class metrics
    ap_per_class: dict[str, float] = field(default_factory=dict)
    precision_per_class: dict[str, float] = field(default_factory=dict)
    recall_per_class: dict[str, float] = field(default_factory=dict)

    # Per-size metrics
    map_small: float = 0.0
    map_medium: float = 0.0
    map_large: float = 0.0

    # Speed metrics
    fps: float = 0.0
    latency_ms: float = 0.0
    latency_std_ms: float = 0.0
    throughput: float = 0.0

    # Dataset info
    dataset_name: str = ""
    num_images: int = 0
    num_ground_truths: int = 0
    num_predictions: int = 0

    # Timing
    evaluation_time_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "model": {
                "name": self.model_name,
                "path": self.model_path,
            },
            "accuracy": {
                "mAP@0.5": self.map_50,
                "mAP@0.75": self.map_75,
                "mAP@[.5:.95]": self.map_50_95,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
            },
            "per_class": {
                "ap": self.ap_per_class,
                "precision": self.precision_per_class,
                "recall": self.recall_per_class,
            },
            "per_size": {
                "mAP_small": self.map_small,
                "mAP_medium": self.map_medium,
                "mAP_large": self.map_large,
            },
            "speed": {
                "fps": self.fps,
                "latency_ms": self.latency_ms,
                "latency_std_ms": self.latency_std_ms,
                "throughput": self.throughput,
            },
            "dataset": {
                "name": self.dataset_name,
                "num_images": self.num_images,
                "num_ground_truths": self.num_ground_truths,
                "num_predictions": self.num_predictions,
            },
            "meta": {
                "evaluation_time_seconds": self.evaluation_time_seconds,
                "timestamp": self.timestamp,
            },
        }


class BaselineEvaluator:
    """
    Main class for running baseline evaluation.

    Orchestrates:
        - Dataset loading
        - Model inference
        - Metric calculation
        - Benchmark execution
        - Report generation
    """

    # Vehicle classes for evaluation
    VEHICLE_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize baseline evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.model = None
        self.dataset = None
        self.results: list[EvaluationResults] = []

    def load_model(self, model_path: Optional[str] = None) -> Any:
        """
        Load detection model.

        Args:
            model_path: Path to model file (uses config if not provided)

        Returns:
            Loaded model
        """
        model_path = model_path or self.config.model_path

        logger.info(f"Loading model: {model_path}")

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)

            # Move to device
            device = self.config.device
            if device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"

            if device == "cuda":
                self.model.to("cuda")

            logger.info(f"Model loaded on device: {device}")
            return self.model

        except ImportError:
            raise ImportError(
                "ultralytics package is required. Install with: pip install ultralytics"
            )

    def load_dataset(
        self,
        dataset_path: Optional[str] = None,
        dataset_format: Optional[str] = None,
    ):
        """
        Load ground truth dataset.

        Args:
            dataset_path: Path to dataset
            dataset_format: Format of dataset ('coco', 'yolo', 'voc', 'custom')

        Returns:
            Loaded dataset
        """
        dataset_path = dataset_path or self.config.dataset_path
        dataset_format = dataset_format or self.config.dataset_format

        if not dataset_path:
            logger.warning("No dataset path provided, skipping dataset loading")
            return None

        logger.info(f"Loading dataset from: {dataset_path} (format: {dataset_format})")

        try:
            if dataset_format == "coco":
                from app.baseline.datasets import COCODataset

                self.dataset = COCODataset(
                    data_path=dataset_path,
                    split=self.config.dataset_split,
                    vehicle_classes=self.VEHICLE_CLASSES,
                )
            else:
                from app.baseline.datasets import CustomDataset

                self.dataset = CustomDataset(
                    data_path=dataset_path,
                    split=self.config.dataset_split,
                    annotation_format=dataset_format,
                    vehicle_classes=self.VEHICLE_CLASSES,
                )

            logger.info(f"Dataset loaded: {len(self.dataset)} images")
            self.dataset.print_statistics()

            return self.dataset

        except FileNotFoundError as e:
            logger.error(f"Dataset not found: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None

    def run_inference(self, image) -> list[dict]:
        """
        Run inference on a single image.

        Args:
            image: Image as numpy array or path

        Returns:
            List of detections
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self.model.predict(
            source=image,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=list(self.VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                detections.append(
                    {
                        "bbox": (
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        ),
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": self.VEHICLE_CLASSES.get(cls_id, "unknown"),
                    }
                )

        return detections

    def evaluate_accuracy(self) -> dict:
        """
        Evaluate model accuracy on the dataset.

        Returns:
            Dictionary with accuracy metrics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.dataset is None:
            logger.warning("No dataset loaded, skipping accuracy evaluation")
            return {}

        logger.info("Running accuracy evaluation...")
        start_time = time.time()

        from app.baseline.metrics.map_calculator import MAPCalculator
        from app.baseline.metrics.precision_recall import (
            Detection,
            GroundTruth,
            calculate_precision_recall,
        )

        # Collect all predictions and ground truths
        all_predictions: list[Detection] = []
        all_ground_truths: list[GroundTruth] = []

        num_images = len(self.dataset)
        if self.config.max_images:
            num_images = min(num_images, self.config.max_images)

        for i, img_annotation in enumerate(self.dataset.annotations[:num_images]):
            if (i + 1) % 100 == 0:
                logger.info(f"Processing image {i + 1}/{num_images}")

            # Load image and run inference
            image = self.dataset.get_image(img_annotation.image_id)
            if image is None:
                continue

            detections = self.run_inference(image)

            # Convert to metric format
            for det in detections:
                all_predictions.append(
                    Detection(
                        bbox=det["bbox"],
                        class_id=det["class_id"],
                        confidence=det["confidence"],
                        image_id=img_annotation.image_id,
                    )
                )

            # Add ground truths
            for gt_ann in img_annotation.annotations:
                all_ground_truths.append(
                    GroundTruth(
                        bbox=gt_ann.bbox.to_xyxy(),
                        class_id=gt_ann.class_id,
                        is_difficult=gt_ann.is_difficult,
                        image_id=img_annotation.image_id,
                    )
                )

        # Calculate mAP
        logger.info("Calculating mAP...")
        map_calculator = MAPCalculator(
            class_names=self.VEHICLE_CLASSES,
            iou_thresholds=self.config.iou_thresholds,
        )
        map_calculator.predictions = all_predictions
        map_calculator.ground_truths = all_ground_truths

        map_result = map_calculator.calculate_coco_map()

        # Calculate precision/recall
        pr_result = calculate_precision_recall(
            predictions=all_predictions,
            ground_truths=all_ground_truths,
            iou_threshold=0.5,
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Accuracy evaluation completed in {elapsed_time:.2f}s")

        return {
            "map_result": map_result,
            "pr_result": pr_result,
            "num_predictions": len(all_predictions),
            "num_ground_truths": len(all_ground_truths),
            "num_images": num_images,
            "evaluation_time": elapsed_time,
        }

    def run_benchmark(self) -> dict:
        """
        Run FPS/latency benchmark.

        Returns:
            Dictionary with benchmark results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info("Running FPS benchmark...")

        from app.baseline.metrics.fps_benchmark import FPSBenchmark

        benchmark = FPSBenchmark(
            model_path=self.config.model_path,
            image_size=self.config.image_size,
            device=self.config.device,
            model_name=self.config.model_name or Path(self.config.model_path).stem,
        )

        result = benchmark.run(
            num_iterations=self.config.benchmark_iterations,
            warmup=self.config.benchmark_warmup,
        )

        logger.info(
            f"Benchmark completed: {result.fps:.2f} FPS, {result.latency_ms:.2f}ms latency"
        )

        return {
            "benchmark_result": result,
            "fps": result.fps,
            "latency_ms": result.latency_ms,
            "latency_std_ms": result.latency_std_ms,
            "throughput": result.throughput,
        }

    def run_full_evaluation(
        self,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
    ) -> EvaluationResults:
        """
        Run full baseline evaluation including accuracy and speed.

        Args:
            model_path: Path to model file
            dataset_path: Path to dataset

        Returns:
            EvaluationResults with all metrics
        """
        logger.info("=" * 60)
        logger.info("STARTING BASELINE EVALUATION")
        logger.info("=" * 60)

        start_time = time.time()
        results = EvaluationResults(timestamp=datetime.now().isoformat())

        # Update config if paths provided
        if model_path:
            self.config.model_path = model_path
        if dataset_path:
            self.config.dataset_path = dataset_path

        # Load model
        self.load_model()
        results.model_path = self.config.model_path
        results.model_name = self.config.model_name or Path(self.config.model_path).stem

        # Load dataset
        self.load_dataset()
        if self.dataset:
            results.dataset_name = self.dataset.__class__.__name__

        # Run accuracy evaluation
        if self.dataset:
            accuracy_results = self.evaluate_accuracy()

            if "map_result" in accuracy_results:
                map_result = accuracy_results["map_result"]
                results.map_50 = map_result.map_50 or 0.0
                results.map_75 = map_result.map_75 or 0.0
                results.map_50_95 = map_result.map_50_95 or 0.0

                # Per-class AP
                for class_name, ap_result in map_result.ap_per_class.items():
                    results.ap_per_class[class_name] = ap_result.ap

            if "pr_result" in accuracy_results:
                pr_result = accuracy_results["pr_result"]
                results.precision = pr_result.precision
                results.recall = pr_result.recall
                results.f1_score = pr_result.f1_score

            results.num_images = accuracy_results.get("num_images", 0)
            results.num_predictions = accuracy_results.get("num_predictions", 0)
            results.num_ground_truths = accuracy_results.get("num_ground_truths", 0)

        # Run benchmark
        benchmark_results = self.run_benchmark()
        results.fps = benchmark_results.get("fps", 0.0)
        results.latency_ms = benchmark_results.get("latency_ms", 0.0)
        results.latency_std_ms = benchmark_results.get("latency_std_ms", 0.0)
        results.throughput = benchmark_results.get("throughput", 0.0)

        # Record total time
        results.evaluation_time_seconds = time.time() - start_time

        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Model: {results.model_name}")
        logger.info(f"mAP@0.5: {results.map_50:.4f}")
        logger.info(f"mAP@[.5:.95]: {results.map_50_95:.4f}")
        logger.info(f"FPS: {results.fps:.2f}")
        logger.info(f"Latency: {results.latency_ms:.2f}ms")
        logger.info(f"Total time: {results.evaluation_time_seconds:.2f}s")
        logger.info("=" * 60)

        self.results.append(results)
        return results

    def generate_report(
        self,
        results: Optional[EvaluationResults] = None,
        output_path: Optional[str] = None,
    ) -> list[str]:
        """
        Generate evaluation report.

        Args:
            results: EvaluationResults to include (uses latest if not provided)
            output_path: Path for output files

        Returns:
            List of generated file paths
        """
        if results is None:
            if not self.results:
                raise ValueError("No results available. Run evaluation first.")
            results = self.results[-1]

        output_path = output_path or self.config.output_dir
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating report to: {output_path}")

        from app.baseline.comparison.benchmark_history import BenchmarkHistory
        from app.baseline.comparison.comparison_report import ComparisonReport

        # Save current results to benchmark history
        history = BenchmarkHistory()
        history.save_from_evaluation_results(results)

        # Create comparison report
        report = ComparisonReport(
            title="Vehicle Detection Baseline Evaluation",
            description="Baseline evaluation for the Vehicle Detection & Monitoring Survey project.",
        )

        # Add evaluated model
        report.add_model_results(
            model_name=results.model_name,
            precision=results.precision,
            recall=results.recall,
            f1_score=results.f1_score,
        )
        # Manually set accuracy metrics
        report.models[results.model_name].map_50 = results.map_50
        report.models[results.model_name].map_75 = results.map_75
        report.models[results.model_name].map_50_95 = results.map_50_95
        report.models[results.model_name].fps = results.fps
        report.models[results.model_name].latency_ms = results.latency_ms
        report.models[results.model_name].ap_per_class = results.ap_per_class

        # Load previous benchmark results for comparison (excluding current model)
        report.load_from_benchmark_history(exclude_models=[results.model_name])

        # Set dataset info
        if self.dataset:
            stats = self.dataset.statistics
            report.set_dataset_info(
                name=self.dataset.__class__.__name__,
                num_images=stats.total_images,
                num_annotations=stats.total_annotations,
                classes=list(self.VEHICLE_CLASSES.values()),
                split=self.config.dataset_split,
            )

        # Generate reports
        report_base = output_path / f"baseline_report_{results.model_name}"
        generated_files = report.generate_report(
            str(report_base),
            formats=self.config.report_formats,
        )

        # Also save raw results as JSON
        results_file = output_path / f"results_{results.model_name}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
        generated_files.append(str(results_file))

        logger.info(f"Generated {len(generated_files)} report files")
        return generated_files


def main():
    """Main entry point for baseline evaluation."""
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation for Vehicle Detection Survey"
    )

    # Model options
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolov8n.pt",
        help="Path to model file (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        help="Multiple model paths for batch evaluation (e.g., --models yolov8n.pt yolov8s.pt yolov8m.pt)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name for the model (default: derived from filename)",
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="coco",
        choices=["coco", "yolo", "voc", "custom"],
        help="Dataset format (default: coco)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to use (default: val)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to evaluate (for quick testing)",
    )

    # Evaluation options
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)",
    )

    # Benchmark options
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run only FPS benchmark, skip accuracy evaluation",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/baseline",
        help="Output directory for reports (default: results/baseline)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report only (requires existing results)",
    )

    args = parser.parse_args()

    # Determine which models to evaluate
    model_paths = args.models if args.models else [args.model]

    if len(model_paths) > 1:
        # Batch mode: evaluate multiple models
        print(f"\n{'='*60}")
        print(f"BATCH EVALUATION: {len(model_paths)} models")
        print(f"{'='*60}")

        all_results = []
        for i, model_path in enumerate(model_paths, 1):
            print(f"\n[{i}/{len(model_paths)}] Evaluating: {model_path}")
            print("-" * 40)

            config = EvaluationConfig(
                model_path=model_path,
                model_name=None,
                dataset_path=args.dataset,
                dataset_format=args.dataset_format,
                dataset_split=args.split,
                max_images=args.max_images,
                confidence_threshold=args.conf,
                iou_threshold=args.iou,
                device=args.device,
                benchmark_iterations=args.benchmark_iterations,
                benchmark_warmup=args.benchmark_warmup,
                output_dir=args.output,
            )

            evaluator = BaselineEvaluator(config)

            if args.benchmark_only:
                evaluator.load_model()
                benchmark_results = evaluator.run_benchmark()
                print(f"  FPS: {benchmark_results['fps']:.1f}, Latency: {benchmark_results['latency_ms']:.2f}ms")

                from app.baseline.comparison.benchmark_history import BenchmarkHistory
                history = BenchmarkHistory()
                history.save_result(
                    model_name=Path(model_path).stem,
                    model_path=model_path,
                    fps=benchmark_results['fps'],
                    latency_ms=benchmark_results['latency_ms'],
                    latency_std_ms=benchmark_results['latency_std_ms'],
                    throughput=benchmark_results['throughput'],
                )
            else:
                results = evaluator.run_full_evaluation()
                all_results.append(results)
                evaluator.generate_report(results)

        print(f"\n{'='*60}")
        print(f"BATCH EVALUATION COMPLETE: {len(model_paths)} models")
        print(f"{'='*60}")
        print(f"Results saved to: {args.output}")

    else:
        # Single model mode
        config = EvaluationConfig(
            model_path=args.model,
            model_name=args.model_name,
            dataset_path=args.dataset,
            dataset_format=args.dataset_format,
            dataset_split=args.split,
            max_images=args.max_images,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            benchmark_iterations=args.benchmark_iterations,
            benchmark_warmup=args.benchmark_warmup,
            output_dir=args.output,
        )

        evaluator = BaselineEvaluator(config)

        if args.benchmark_only:
            evaluator.load_model()
            benchmark_results = evaluator.run_benchmark()
            print(benchmark_results["benchmark_result"])
        elif args.report_only:
            results_file = Path(args.output) / f"results_{Path(args.model).stem}.json"
            if results_file.exists():
                with open(results_file, "r") as f:
                    data = json.load(f)
                results = EvaluationResults(**data.get("model", {}))
                evaluator.generate_report(results)
            else:
                print(f"No existing results found at {results_file}")
                sys.exit(1)
        else:
            results = evaluator.run_full_evaluation()
            evaluator.generate_report(results)

        print("\nBaseline evaluation completed!")
        print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
