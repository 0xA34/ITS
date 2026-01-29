"""
Benchmark Comparator Module
===========================
Compare evaluation results with your own previously benchmarked models
from the benchmark history.

This module validates result consistency by:
- Comparing with your own previous benchmark results
- Detecting anomalies or regressions
- Providing recommendations based on comparisons
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing evaluated results with previous benchmarks."""

    model_name: str
    metric_name: str
    current_value: float
    previous_value: float
    difference: float
    percentage_diff: float
    status: str  # "IMPROVED", "STABLE", "DEGRADED"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "difference": self.difference,
            "percentage_diff": self.percentage_diff,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass
class ValidationSummary:
    """Summary of benchmark validation."""

    model_name: str
    total_metrics: int
    improved: int
    stable: int
    degraded: int
    overall_status: str
    comparisons: list[ComparisonResult] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "total_metrics": self.total_metrics,
            "improved": self.improved,
            "stable": self.stable,
            "degraded": self.degraded,
            "overall_status": self.overall_status,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "recommendations": self.recommendations,
        }


class BenchmarkComparator:
    """
    Compare evaluation results with your own previous benchmarks.

    This class validates that your current evaluation results are consistent
    with your own previous runs, helping detect:
    - Regressions (worse performance than before)
    - Improvements (better performance)
    - Anomalies (unexpected changes)
    """

    # Threshold for considering a change significant
    SIGNIFICANCE_THRESHOLD = 0.05  # 5% change is considered significant

    def __init__(
        self,
        significance_threshold: float = 0.05,
    ):
        """
        Initialize BenchmarkComparator.

        Args:
            significance_threshold: Threshold for significant change (default 5%)
        """
        self.significance_threshold = significance_threshold

    def compare_with_history(
        self,
        current_results: dict[str, Any],
        model_name: str,
    ) -> ValidationSummary:
        """
        Compare current evaluation results with previous benchmarks from history.

        Args:
            current_results: Dictionary with current evaluation metrics
            model_name: Name of the model to compare

        Returns:
            ValidationSummary with comparison results
        """
        try:
            from app.baseline.comparison.benchmark_history import BenchmarkHistory

            history = BenchmarkHistory()
            previous = history.get_result(model_name)

            if previous is None:
                return ValidationSummary(
                    model_name=model_name,
                    total_metrics=0,
                    improved=0,
                    stable=0,
                    degraded=0,
                    overall_status="NO_HISTORY",
                    recommendations=[
                        f"No previous benchmark found for model '{model_name}'",
                        "Run baseline evaluation first to establish a benchmark.",
                    ],
                )

            comparisons = []
            current_metrics = self._extract_metrics(current_results)

            # Compare each metric
            metric_mappings = [
                ("map_50", "mAP@0.5", previous.map_50),
                ("map_50_95", "mAP@[.5:.95]", previous.map_50_95),
                ("precision", "precision", previous.precision),
                ("recall", "recall", previous.recall),
                ("fps", "fps", previous.fps),
            ]

            for key, display_name, previous_value in metric_mappings:
                if previous_value is None or previous_value == 0:
                    continue

                current_value = current_metrics.get(display_name)
                if current_value is None:
                    continue

                comparison = self._compare_metric(
                    model_name=model_name,
                    metric_name=display_name,
                    current_value=current_value,
                    previous_value=previous_value,
                )
                comparisons.append(comparison)

            # Calculate summary
            improved = sum(1 for c in comparisons if c.status == "IMPROVED")
            stable = sum(1 for c in comparisons if c.status == "STABLE")
            degraded = sum(1 for c in comparisons if c.status == "DEGRADED")
            total = len(comparisons)

            if degraded > 0:
                overall_status = "REGRESSION"
            elif improved > 0:
                overall_status = "IMPROVED"
            else:
                overall_status = "STABLE"

            recommendations = self._generate_recommendations(comparisons)

            return ValidationSummary(
                model_name=model_name,
                total_metrics=total,
                improved=improved,
                stable=stable,
                degraded=degraded,
                overall_status=overall_status,
                comparisons=comparisons,
                recommendations=recommendations,
            )

        except ImportError as e:
            logger.error(f"Failed to import benchmark_history: {e}")
            return ValidationSummary(
                model_name=model_name,
                total_metrics=0,
                improved=0,
                stable=0,
                degraded=0,
                overall_status="ERROR",
                recommendations=[f"Error loading benchmark history: {e}"],
            )

    def _extract_metrics(self, results: dict[str, Any]) -> dict[str, float]:
        """Extract metrics from various result formats."""
        metrics = {}

        # Handle nested format (from baseline evaluation)
        if "accuracy" in results:
            acc = results["accuracy"]
            metrics["mAP@0.5"] = acc.get("mAP@0.5", 0)
            metrics["mAP@[.5:.95]"] = acc.get("mAP@[.5:.95]", 0)
            metrics["precision"] = acc.get("precision", 0)
            metrics["recall"] = acc.get("recall", 0)

        if "speed" in results:
            speed = results["speed"]
            metrics["fps"] = speed.get("fps", 0)

        # Handle flat format
        for key in ["mAP@0.5", "mAP@[.5:.95]", "precision", "recall", "fps"]:
            if key in results:
                metrics[key] = results[key]

        return metrics

    def _compare_metric(
        self,
        model_name: str,
        metric_name: str,
        current_value: float,
        previous_value: float,
    ) -> ComparisonResult:
        """Compare a single metric with previous value."""
        difference = current_value - previous_value
        percentage_diff = (
            (difference / previous_value * 100) if previous_value != 0 else 0
        )

        # Determine status
        threshold_pct = self.significance_threshold * 100
        if percentage_diff > threshold_pct:
            status = "IMPROVED"
        elif percentage_diff < -threshold_pct:
            status = "DEGRADED"
        else:
            status = "STABLE"

        return ComparisonResult(
            model_name=model_name,
            metric_name=metric_name,
            current_value=current_value,
            previous_value=previous_value,
            difference=difference,
            percentage_diff=percentage_diff,
            status=status,
        )

    def _generate_recommendations(
        self,
        comparisons: list[ComparisonResult],
    ) -> list[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []

        degraded = [c for c in comparisons if c.status == "DEGRADED"]
        improved = [c for c in comparisons if c.status == "IMPROVED"]

        if degraded:
            metrics_list = ", ".join(c.metric_name for c in degraded)
            recommendations.append(
                f"Performance regression detected in: {metrics_list}. "
                "Check for changes in test conditions or model configuration."
            )

        if improved:
            metrics_list = ", ".join(c.metric_name for c in improved)
            recommendations.append(
                f"Performance improved in: {metrics_list}. "
                "Consider updating the benchmark history."
            )

        if not recommendations:
            recommendations.append(
                "Results are consistent with previous benchmarks. Validation passed!"
            )

        return recommendations

    def print_comparison(self, summary: ValidationSummary):
        """Print formatted comparison results."""
        print(f"\n{'=' * 70}")
        print(f"Benchmark Validation: {summary.model_name}")
        print(f"{'=' * 70}")
        print(f"Overall Status: {summary.overall_status}")
        print(f"Metrics Compared: {summary.total_metrics}")
        print(f"  ↑ Improved: {summary.improved}")
        print(f"  = Stable:   {summary.stable}")
        print(f"  ↓ Degraded: {summary.degraded}")

        if summary.comparisons:
            print(f"\nDetailed Comparison:")
            print("-" * 70)
            print(
                f"{'Metric':<20} {'Current':<12} {'Previous':<12} {'Diff %':<10} {'Status':<8}"
            )
            print("-" * 70)

            for comp in summary.comparisons:
                status_symbol = {"IMPROVED": "↑", "STABLE": "=", "DEGRADED": "↓"}.get(
                    comp.status, "?"
                )
                print(
                    f"{comp.metric_name:<20} {comp.current_value:<12.4f} "
                    f"{comp.previous_value:<12.4f} {comp.percentage_diff:>+8.1f}%  "
                    f"{status_symbol} {comp.status}"
                )

        if summary.recommendations:
            print(f"\nRecommendations:")
            print("-" * 70)
            for i, rec in enumerate(summary.recommendations, 1):
                print(f"  {i}. {rec}")

        print(f"{'=' * 70}\n")


def compare_with_history(
    results: dict[str, Any],
    model_name: str,
    print_results: bool = True,
) -> ValidationSummary:
    """
    Convenience function to compare results with benchmark history.

    Args:
        results: Evaluation results dictionary
        model_name: Name of the model
        print_results: Whether to print comparison results

    Returns:
        ValidationSummary
    """
    comparator = BenchmarkComparator()
    summary = comparator.compare_with_history(results, model_name)

    if print_results:
        comparator.print_comparison(summary)

    return summary
