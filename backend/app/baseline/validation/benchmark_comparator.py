"""
Benchmark Comparator Module
===========================
Compare evaluation results with official published benchmarks from
Ultralytics and other sources to validate result accuracy.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BenchmarkSource(Enum):
    """Sources of benchmark data."""

    ULTRALYTICS = "ultralytics"
    COCO_LEADERBOARD = "coco_leaderboard"
    PAPERS_WITH_CODE = "papers_with_code"
    CUSTOM = "custom"


@dataclass
class OfficialBenchmark:
    """Official benchmark data for a model."""

    model_name: str
    source: BenchmarkSource
    dataset: str
    map_50: Optional[float] = None
    map_50_95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    fps: Optional[float] = None
    fps_device: str = "V100"
    input_size: int = 640
    url: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "source": self.source.value,
            "dataset": self.dataset,
            "map_50": self.map_50,
            "map_50_95": self.map_50_95,
            "precision": self.precision,
            "recall": self.recall,
            "fps": self.fps,
            "fps_device": self.fps_device,
            "input_size": self.input_size,
            "url": self.url,
            "notes": self.notes,
        }


@dataclass
class ComparisonResult:
    """Result of comparing evaluated results with official benchmarks."""

    model_name: str
    metric_name: str
    our_value: float
    official_value: float
    difference: float
    percentage_diff: float
    is_within_tolerance: bool
    tolerance: float
    status: str  # "PASS", "WARN", "FAIL"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "metric_name": self.metric_name,
            "our_value": self.our_value,
            "official_value": self.official_value,
            "difference": self.difference,
            "percentage_diff": self.percentage_diff,
            "is_within_tolerance": self.is_within_tolerance,
            "tolerance": self.tolerance,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass
class ValidationSummary:
    """Summary of benchmark validation."""

    model_name: str
    total_metrics: int
    passed: int
    warned: int
    failed: int
    overall_status: str
    comparisons: list[ComparisonResult] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "total_metrics": self.total_metrics,
            "passed": self.passed,
            "warned": self.warned,
            "failed": self.failed,
            "overall_status": self.overall_status,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "recommendations": self.recommendations,
        }


class OfficialBenchmarks:
    """
    Database of official published benchmarks for various models.

    Data sourced from:
    - Ultralytics official documentation
    - COCO leaderboard
    - Original papers
    """

    # YOLOv8 official benchmarks from Ultralytics (COCO val2017)
    # Source: https://docs.ultralytics.com/models/yolov8/
    # Note: Official reports mAP@0.5:0.95, mAP@0.5 is approximately 1.4x higher
    YOLOV8_BENCHMARKS = {
        "yolov8n": OfficialBenchmark(
            model_name="yolov8n",
            source=BenchmarkSource.ULTRALYTICS,
            dataset="COCO val2017",
            map_50=52.2,  # Estimated: mAP@0.5 ≈ 1.4 × mAP@0.5:0.95
            map_50_95=37.3,  # Official from Ultralytics docs
            fps=1010.0,  # 1/0.99ms on A100 TensorRT
            fps_device="A100 TensorRT",
            input_size=640,
            url="https://docs.ultralytics.com/models/yolov8/",
            notes="Nano model, 3.2M params, 8.7 GFLOPs",
        ),
        "yolov8s": OfficialBenchmark(
            model_name="yolov8s",
            source=BenchmarkSource.ULTRALYTICS,
            dataset="COCO val2017",
            map_50=62.9,  # Estimated: mAP@0.5 ≈ 1.4 × mAP@0.5:0.95
            map_50_95=44.9,  # Official from Ultralytics docs
            fps=833.0,  # 1/1.20ms on A100 TensorRT
            fps_device="A100 TensorRT",
            input_size=640,
            url="https://docs.ultralytics.com/models/yolov8/",
            notes="Small model, 11.2M params, 28.6 GFLOPs",
        ),
        "yolov8m": OfficialBenchmark(
            model_name="yolov8m",
            source=BenchmarkSource.ULTRALYTICS,
            dataset="COCO val2017",
            map_50=70.3,  # Estimated: mAP@0.5 ≈ 1.4 × mAP@0.5:0.95
            map_50_95=50.2,  # Official from Ultralytics docs
            fps=546.0,  # 1/1.83ms on A100 TensorRT
            fps_device="A100 TensorRT",
            input_size=640,
            url="https://docs.ultralytics.com/models/yolov8/",
            notes="Medium model, 25.9M params, 78.9 GFLOPs",
        ),
        "yolov8l": OfficialBenchmark(
            model_name="yolov8l",
            source=BenchmarkSource.ULTRALYTICS,
            dataset="COCO val2017",
            map_50=74.1,  # Estimated: mAP@0.5 ≈ 1.4 × mAP@0.5:0.95
            map_50_95=52.9,  # Official from Ultralytics docs
            fps=418.0,  # 1/2.39ms on A100 TensorRT
            fps_device="A100 TensorRT",
            input_size=640,
            url="https://docs.ultralytics.com/models/yolov8/",
            notes="Large model, 43.7M params, 165.2 GFLOPs",
        ),
        "yolov8x": OfficialBenchmark(
            model_name="yolov8x",
            source=BenchmarkSource.ULTRALYTICS,
            dataset="COCO val2017",
            map_50=75.5,  # Estimated: mAP@0.5 ≈ 1.4 × mAP@0.5:0.95
            map_50_95=53.9,  # Official from Ultralytics docs
            fps=283.0,  # 1/3.53ms on A100 TensorRT
            fps_device="A100 TensorRT",
            input_size=640,
            url="https://docs.ultralytics.com/models/yolov8/",
            notes="Extra-large model, 68.2M params, 257.8 GFLOPs",
        ),
    }

    # YOLOv12 official benchmarks from Ultralytics (COCO val2017)
    # Source: https://docs.ultralytics.com/models/yolo12/
    # Note: Speed measured on T4 TensorRT FP16
    YOLOV12_BENCHMARKS = {
        "yolov12n": OfficialBenchmark(
            model_name="yolov12n",
            source=BenchmarkSource.ULTRALYTICS,
            dataset="COCO val2017",
            map_50=56.8,  # Estimated: mAP@0.5 ≈ 1.4 × mAP@0.5:0.95
            map_50_95=40.6,  # Official from Ultralytics docs
            fps=610.0,  # 1/1.64ms on T4 TensorRT
            fps_device="T4 TensorRT FP16",
            input_size=640,
            url="https://docs.ultralytics.com/models/yolo12/",
            notes="YOLO12 Nano, 2.6M params, 6.5 GFLOPs, attention-based",
        ),
        "yolov12s": OfficialBenchmark(
            model_name="yolov12s",
            source=BenchmarkSource.ULTRALYTICS,
            dataset="COCO val2017",
            map_50=67.2,  # Estimated: mAP@0.5 ≈ 1.4 × mAP@0.5:0.95
            map_50_95=48.0,  # Official from Ultralytics docs
            fps=383.0,  # 1/2.61ms on T4 TensorRT
            fps_device="T4 TensorRT FP16",
            input_size=640,
            url="https://docs.ultralytics.com/models/yolo12/",
            notes="YOLO12 Small, 9.3M params, 21.4 GFLOPs, attention-based",
        ),
        "yolov12m": OfficialBenchmark(
            model_name="yolov12m",
            source=BenchmarkSource.ULTRALYTICS,
            dataset="COCO val2017",
            map_50=73.5,  # Estimated: mAP@0.5 ≈ 1.4 × mAP@0.5:0.95
            map_50_95=52.5,  # Official from Ultralytics docs
            fps=206.0,  # 1/4.86ms on T4 TensorRT
            fps_device="T4 TensorRT FP16",
            input_size=640,
            url="https://docs.ultralytics.com/models/yolo12/",
            notes="YOLO12 Medium, 20.2M params, 67.5 GFLOPs, attention-based",
        ),
        "yolov12l": OfficialBenchmark(
            model_name="yolov12l",
            source=BenchmarkSource.ULTRALYTICS,
            dataset="COCO val2017",
            map_50=75.2,  # Estimated: mAP@0.5 ≈ 1.4 × mAP@0.5:0.95
            map_50_95=53.7,  # Official from Ultralytics docs
            fps=148.0,  # 1/6.77ms on T4 TensorRT
            fps_device="T4 TensorRT FP16",
            input_size=640,
            url="https://docs.ultralytics.com/models/yolo12/",
            notes="YOLO12 Large, 26.4M params, 88.9 GFLOPs, attention-based",
        ),
        "yolov12x": OfficialBenchmark(
            model_name="yolov12x",
            source=BenchmarkSource.ULTRALYTICS,
            dataset="COCO val2017",
            map_50=77.3,  # Estimated: mAP@0.5 ≈ 1.4 × mAP@0.5:0.95
            map_50_95=55.2,  # Official from Ultralytics docs
            fps=85.0,  # 1/11.79ms on T4 TensorRT
            fps_device="T4 TensorRT FP16",
            input_size=640,
            url="https://docs.ultralytics.com/models/yolo12/",
            notes="YOLO12 Extra-large, 59.1M params, 199.0 GFLOPs, attention-based",
        ),
    }

    # Other model benchmarks for reference
    OTHER_BENCHMARKS = {
        "yolov5n": OfficialBenchmark(
            model_name="yolov5n",
            source=BenchmarkSource.ULTRALYTICS,
            dataset="COCO val2017",
            map_50=45.7,
            map_50_95=28.0,
            fps=160.0,
            fps_device="V100",
            input_size=640,
            url="https://docs.ultralytics.com/models/yolov5/",
        ),
        "yolov5s": OfficialBenchmark(
            model_name="yolov5s",
            source=BenchmarkSource.ULTRALYTICS,
            dataset="COCO val2017",
            map_50=56.8,
            map_50_95=37.4,
            fps=120.0,
            fps_device="V100",
            input_size=640,
            url="https://docs.ultralytics.com/models/yolov5/",
        ),
        "faster_rcnn_r50": OfficialBenchmark(
            model_name="faster_rcnn_r50",
            source=BenchmarkSource.COCO_LEADERBOARD,
            dataset="COCO val2017",
            map_50=58.0,
            map_50_95=37.9,
            fps=15.0,
            fps_device="V100",
            input_size=800,
            notes="Faster R-CNN with ResNet-50 FPN",
        ),
        "detr_r50": OfficialBenchmark(
            model_name="detr_r50",
            source=BenchmarkSource.PAPERS_WITH_CODE,
            dataset="COCO val2017",
            map_50=63.1,
            map_50_95=42.0,
            fps=28.0,
            fps_device="V100",
            input_size=800,
            notes="DETR with ResNet-50 backbone",
        ),
    }

    @classmethod
    def get_all_benchmarks(cls) -> dict[str, OfficialBenchmark]:
        """Get all available benchmarks."""
        all_benchmarks = {}
        all_benchmarks.update(cls.YOLOV8_BENCHMARKS)
        all_benchmarks.update(cls.YOLOV12_BENCHMARKS)
        all_benchmarks.update(cls.OTHER_BENCHMARKS)
        return all_benchmarks

    @classmethod
    def get_benchmark(cls, model_name: str) -> Optional[OfficialBenchmark]:
        """
        Get benchmark for a specific model.

        Args:
            model_name: Name of the model (e.g., 'yolov8n', 'yolov12x')

        Returns:
            OfficialBenchmark if found, None otherwise
        """
        model_name = model_name.lower().replace("-", "").replace("_", "")
        all_benchmarks = cls.get_all_benchmarks()

        # Direct match
        if model_name in all_benchmarks:
            return all_benchmarks[model_name]

        # Try partial match
        for key, benchmark in all_benchmarks.items():
            if model_name in key or key in model_name:
                return benchmark

        return None

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List all models with available benchmarks."""
        return list(cls.get_all_benchmarks().keys())

    @classmethod
    def print_summary(cls):
        """Print summary of all available benchmarks."""
        print("\n" + "=" * 80)
        print("Official Benchmarks Database")
        print("=" * 80)

        print("\nYOLOv8 Models (Ultralytics):")
        print("-" * 80)
        print(
            f"{'Model':<12} {'mAP@0.5':<10} {'mAP@.5:.95':<12} {'FPS':<8} {'Device':<8}"
        )
        print("-" * 80)
        for name, b in cls.YOLOV8_BENCHMARKS.items():
            print(
                f"{name:<12} {b.map_50:<10.1f} {b.map_50_95:<12.1f} {b.fps:<8.1f} {b.fps_device:<8}"
            )

        print("\nYOLOv12 Models (Ultralytics):")
        print("-" * 80)
        for name, b in cls.YOLOV12_BENCHMARKS.items():
            print(
                f"{name:<12} {b.map_50:<10.1f} {b.map_50_95:<12.1f} {b.fps:<8.1f} {b.fps_device:<8}"
            )

        print("\nOther Reference Models:")
        print("-" * 80)
        for name, b in cls.OTHER_BENCHMARKS.items():
            map50 = b.map_50 if b.map_50 else 0
            map5095 = b.map_50_95 if b.map_50_95 else 0
            fps = b.fps if b.fps else 0
            print(
                f"{name:<12} {map50:<10.1f} {map5095:<12.1f} {fps:<8.1f} {b.fps_device:<8}"
            )

        print("=" * 80 + "\n")


class BenchmarkComparator:
    """
    Compare evaluation results with official published benchmarks.

    This class validates that our evaluation results are consistent with
    officially published benchmarks, accounting for:
    - Different hardware (GPU types affect FPS)
    - Different dataset subsets (we may use vehicle classes only)
    - Measurement variance
    """

    # Default tolerances for comparison
    DEFAULT_TOLERANCES = {
        "map_50": 0.10,  # 10% relative tolerance for mAP@0.5
        "map_50_95": 0.10,  # 10% relative tolerance for mAP@0.5:0.95
        "precision": 0.10,  # 10% for precision
        "recall": 0.10,  # 10% for recall
        "fps": 0.50,  # 50% for FPS (hardware dependent)
    }

    # Expected differences when evaluating on vehicle classes only vs all COCO classes
    VEHICLE_SUBSET_ADJUSTMENTS = {
        "map_50": -0.05,  # Vehicle classes may perform differently
        "map_50_95": -0.05,
    }

    def __init__(
        self,
        tolerances: Optional[dict[str, float]] = None,
        adjust_for_vehicle_subset: bool = True,
    ):
        """
        Initialize BenchmarkComparator.

        Args:
            tolerances: Custom tolerances for each metric (relative, e.g., 0.1 = 10%)
            adjust_for_vehicle_subset: Adjust expectations for vehicle-only evaluation
        """
        self.tolerances = tolerances or self.DEFAULT_TOLERANCES.copy()
        self.adjust_for_vehicle_subset = adjust_for_vehicle_subset

    def compare_with_official(
        self,
        our_results: dict[str, Any],
        model_name: str,
    ) -> ValidationSummary:
        """
        Compare our evaluation results with official benchmarks.

        Args:
            our_results: Dictionary with our evaluation metrics
            model_name: Name of the model to compare

        Returns:
            ValidationSummary with comparison results
        """
        official = OfficialBenchmarks.get_benchmark(model_name)

        if official is None:
            return ValidationSummary(
                model_name=model_name,
                total_metrics=0,
                passed=0,
                warned=0,
                failed=0,
                overall_status="NO_BENCHMARK",
                recommendations=[
                    f"No official benchmark found for model '{model_name}'",
                    f"Available models: {', '.join(OfficialBenchmarks.list_available_models()[:10])}...",
                ],
            )

        comparisons = []

        # Extract our metrics
        our_metrics = self._extract_metrics(our_results)

        # Compare each metric
        metric_mappings = [
            ("map_50", "mAP@0.5", official.map_50),
            ("map_50_95", "mAP@[.5:.95]", official.map_50_95),
            ("fps", "fps", official.fps),
        ]

        for official_key, our_key, official_value in metric_mappings:
            if official_value is None:
                continue

            our_value = our_metrics.get(our_key)
            if our_value is None:
                continue

            comparison = self._compare_metric(
                model_name=model_name,
                metric_name=our_key,
                our_value=our_value,
                official_value=official_value,
                tolerance=self.tolerances.get(official_key, 0.1),
            )
            comparisons.append(comparison)

        # Calculate summary
        passed = sum(1 for c in comparisons if c.status == "PASS")
        warned = sum(1 for c in comparisons if c.status == "WARN")
        failed = sum(1 for c in comparisons if c.status == "FAIL")
        total = len(comparisons)

        if failed > 0:
            overall_status = "FAIL"
        elif warned > 0:
            overall_status = "WARN"
        elif passed > 0:
            overall_status = "PASS"
        else:
            overall_status = "UNKNOWN"

        # Generate recommendations
        recommendations = self._generate_recommendations(comparisons, official)

        return ValidationSummary(
            model_name=model_name,
            total_metrics=total,
            passed=passed,
            warned=warned,
            failed=failed,
            overall_status=overall_status,
            comparisons=comparisons,
            recommendations=recommendations,
        )

    def _extract_metrics(self, results: dict[str, Any]) -> dict[str, float]:
        """Extract metrics from various result formats."""
        metrics = {}

        # Handle nested format (from baseline evaluation)
        if "accuracy" in results:
            acc = results["accuracy"]
            metrics["mAP@0.5"] = acc.get("mAP@0.5", 0) * 100  # Convert to percentage
            metrics["mAP@[.5:.95]"] = acc.get("mAP@[.5:.95]", 0) * 100
            metrics["precision"] = acc.get("precision", 0) * 100
            metrics["recall"] = acc.get("recall", 0) * 100

        if "speed" in results:
            speed = results["speed"]
            metrics["fps"] = speed.get("fps", 0)
            metrics["latency_ms"] = speed.get("latency_ms", 0)

        # Handle flat format
        if "mAP@0.5" in results:
            val = results["mAP@0.5"]
            metrics["mAP@0.5"] = val * 100 if val <= 1 else val

        if "mAP@[.5:.95]" in results:
            val = results["mAP@[.5:.95]"]
            metrics["mAP@[.5:.95]"] = val * 100 if val <= 1 else val

        if "fps" in results:
            metrics["fps"] = results["fps"]

        # Handle aggregated format (from cross-validation)
        if "aggregated" in results:
            agg = results["aggregated"]
            if "metrics" in agg and "mean" in agg["metrics"]:
                mean_metrics = agg["metrics"]["mean"]
                for key, val in mean_metrics.items():
                    if val <= 1:
                        metrics[key] = val * 100
                    else:
                        metrics[key] = val

            if "speed" in agg and "mean" in agg["speed"]:
                mean_speed = agg["speed"]["mean"]
                metrics.update(mean_speed)

        return metrics

    def _compare_metric(
        self,
        model_name: str,
        metric_name: str,
        our_value: float,
        official_value: float,
        tolerance: float,
    ) -> ComparisonResult:
        """Compare a single metric with official value."""
        difference = our_value - official_value
        percentage_diff = (
            (difference / official_value * 100) if official_value != 0 else 0
        )

        # Apply vehicle subset adjustment if applicable
        adjusted_tolerance = tolerance
        if self.adjust_for_vehicle_subset and metric_name in [
            "mAP@0.5",
            "mAP@[.5:.95]",
        ]:
            # Allow more negative deviation for vehicle-only evaluation
            adjusted_tolerance = tolerance + 0.15  # 15% extra tolerance

        is_within_tolerance = abs(percentage_diff) <= (adjusted_tolerance * 100)

        # Determine status
        if is_within_tolerance:
            if abs(percentage_diff) <= (tolerance * 50):  # Within half tolerance
                status = "PASS"
            else:
                status = "WARN"
        else:
            # Check if we're better than official (not a problem)
            if our_value > official_value:
                status = "WARN"  # Better than expected, might indicate different test conditions
            else:
                status = "FAIL"

        notes = ""
        if self.adjust_for_vehicle_subset and metric_name in [
            "mAP@0.5",
            "mAP@[.5:.95]",
        ]:
            notes = "Evaluated on vehicle classes only (6 classes vs 80 COCO classes)"

        return ComparisonResult(
            model_name=model_name,
            metric_name=metric_name,
            our_value=our_value,
            official_value=official_value,
            difference=difference,
            percentage_diff=percentage_diff,
            is_within_tolerance=is_within_tolerance,
            tolerance=adjusted_tolerance,
            status=status,
            notes=notes,
        )

    def _generate_recommendations(
        self,
        comparisons: list[ComparisonResult],
        official: OfficialBenchmark,
    ) -> list[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []

        for comp in comparisons:
            if comp.status == "FAIL":
                if comp.our_value < comp.official_value:
                    if "mAP" in comp.metric_name:
                        recommendations.append(
                            f"mAP lower than expected: Consider checking if dataset matches "
                            f"official evaluation (vehicle classes vs all COCO classes)"
                        )
                    elif comp.metric_name == "fps":
                        recommendations.append(
                            f"FPS lower than expected: Official benchmark uses {official.fps_device}. "
                            f"Hardware differences can cause 2-3x variation."
                        )
                else:
                    recommendations.append(
                        f"{comp.metric_name} higher than official benchmark: "
                        f"Verify evaluation methodology matches official setup"
                    )
            elif comp.status == "WARN":
                if comp.our_value > comp.official_value:
                    recommendations.append(
                        f"{comp.metric_name} slightly higher than expected: "
                        f"May indicate different test conditions or dataset subset"
                    )

        if not recommendations:
            recommendations.append(
                "Results are consistent with official benchmarks. Validation passed!"
            )

        return recommendations

    def print_comparison(self, summary: ValidationSummary):
        """Print formatted comparison results."""
        print(f"\n{'=' * 70}")
        print(f"Benchmark Validation: {summary.model_name}")
        print(f"{'=' * 70}")
        print(f"Overall Status: {summary.overall_status}")
        print(f"Metrics Compared: {summary.total_metrics}")
        print(f"  ✓ Passed: {summary.passed}")
        print(f"  ⚠ Warned: {summary.warned}")
        print(f"  ✗ Failed: {summary.failed}")

        if summary.comparisons:
            print(f"\nDetailed Comparison:")
            print("-" * 70)
            print(
                f"{'Metric':<20} {'Ours':<12} {'Official':<12} {'Diff %':<10} {'Status':<8}"
            )
            print("-" * 70)

            for comp in summary.comparisons:
                status_symbol = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(
                    comp.status, "?"
                )
                print(
                    f"{comp.metric_name:<20} {comp.our_value:<12.2f} "
                    f"{comp.official_value:<12.2f} {comp.percentage_diff:>+8.1f}%  "
                    f"{status_symbol} {comp.status}"
                )

        if summary.recommendations:
            print(f"\nRecommendations:")
            print("-" * 70)
            for i, rec in enumerate(summary.recommendations, 1):
                print(f"  {i}. {rec}")

        print(f"{'=' * 70}\n")


def compare_with_benchmarks(
    results: dict[str, Any],
    model_name: str,
    print_results: bool = True,
) -> ValidationSummary:
    """
    Convenience function to compare results with official benchmarks.

    Args:
        results: Evaluation results dictionary
        model_name: Name of the model
        print_results: Whether to print comparison results

    Returns:
        ValidationSummary
    """
    comparator = BenchmarkComparator()
    summary = comparator.compare_with_official(results, model_name)

    if print_results:
        comparator.print_comparison(summary)

    return summary
