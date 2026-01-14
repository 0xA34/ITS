"""
Statistical Analyzer Module
===========================
Calculate confidence intervals, perform significance tests, and analyze
the statistical properties of baseline evaluation results.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""

    metric_name: str
    mean: float
    std: float
    lower: float
    upper: float
    confidence_level: float
    n_samples: int

    def __str__(self) -> str:
        return f"{self.metric_name}: {self.mean:.4f} [{self.lower:.4f}, {self.upper:.4f}] (95% CI)"

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "mean": self.mean,
            "std": self.std,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "n_samples": self.n_samples,
        }


@dataclass
class SignificanceTestResult:
    """Result of a statistical significance test."""

    test_name: str
    model_a: str
    model_b: str
    metric: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    effect_size: Optional[float] = None
    conclusion: str = ""

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "alpha": self.alpha,
            "effect_size": self.effect_size,
            "conclusion": self.conclusion,
        }


@dataclass
class StatisticalSummary:
    """Complete statistical summary for a model's results."""

    model_name: str
    n_runs: int
    confidence_intervals: dict[str, ConfidenceInterval] = field(default_factory=dict)
    normality_tests: dict[str, dict] = field(default_factory=dict)
    stability_score: float = 0.0  # Lower is more stable
    reliability_rating: str = ""  # "High", "Medium", "Low"

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_runs": self.n_runs,
            "confidence_intervals": {
                k: v.to_dict() for k, v in self.confidence_intervals.items()
            },
            "normality_tests": self.normality_tests,
            "stability_score": self.stability_score,
            "reliability_rating": self.reliability_rating,
        }


class StatisticalAnalyzer:
    """
    Analyzer for computing statistical properties of evaluation results.

    This class provides methods to:
    - Calculate confidence intervals for metrics
    - Perform normality tests
    - Run significance tests between models
    - Assess result stability and reliability
    """

    # T-distribution critical values for common confidence levels and degrees of freedom
    # Format: {df: {confidence_level: t_value}}
    T_CRITICAL = {
        1: {0.90: 6.314, 0.95: 12.706, 0.99: 63.657},
        2: {0.90: 2.920, 0.95: 4.303, 0.99: 9.925},
        3: {0.90: 2.353, 0.95: 3.182, 0.99: 5.841},
        4: {0.90: 2.132, 0.95: 2.776, 0.99: 4.604},
        5: {0.90: 2.015, 0.95: 2.571, 0.99: 4.032},
        6: {0.90: 1.943, 0.95: 2.447, 0.99: 3.707},
        7: {0.90: 1.895, 0.95: 2.365, 0.99: 3.499},
        8: {0.90: 1.860, 0.95: 2.306, 0.99: 3.355},
        9: {0.90: 1.833, 0.95: 2.262, 0.99: 3.250},
        10: {0.90: 1.812, 0.95: 2.228, 0.99: 3.169},
        15: {0.90: 1.753, 0.95: 2.131, 0.99: 2.947},
        20: {0.90: 1.725, 0.95: 2.086, 0.99: 2.845},
        30: {0.90: 1.697, 0.95: 2.042, 0.99: 2.750},
        float("inf"): {0.90: 1.645, 0.95: 1.960, 0.99: 2.576},  # Z-values for large n
    }

    def __init__(self, results: Optional[Any] = None):
        """
        Initialize StatisticalAnalyzer.

        Args:
            results: CrossValidationResult or dict with run results
        """
        self.results = results
        self._values_cache: dict[str, list[float]] = {}

    def set_results(self, results: Any):
        """Set or update the results to analyze."""
        self.results = results
        self._values_cache.clear()

    def _get_t_critical(self, df: int, confidence: float) -> float:
        """Get t-critical value for given degrees of freedom and confidence level."""
        # Find the closest df in our table
        available_dfs = sorted([d for d in self.T_CRITICAL.keys() if d != float("inf")])

        if df >= 30:
            # Use z-value approximation for large samples
            return self.T_CRITICAL[float("inf")].get(confidence, 1.96)

        for available_df in available_dfs:
            if df <= available_df:
                return self.T_CRITICAL[available_df].get(confidence, 2.0)

        return self.T_CRITICAL[30].get(confidence, 2.0)

    def _extract_metric_values(self, metric_name: str) -> list[float]:
        """Extract values for a specific metric from all runs."""
        cache_key = metric_name
        if cache_key in self._values_cache:
            return self._values_cache[cache_key]

        values = []

        if self.results is None:
            return values

        # Handle CrossValidationResult
        if hasattr(self.results, "runs"):
            for run in self.results.runs:
                if hasattr(run, "metrics") and metric_name in run.metrics:
                    values.append(run.metrics[metric_name])
                elif hasattr(run, "speed") and metric_name in run.speed:
                    values.append(run.speed[metric_name])
                elif hasattr(run, "per_class_ap") and metric_name in run.per_class_ap:
                    values.append(run.per_class_ap[metric_name])

        # Handle dict format
        elif isinstance(self.results, dict):
            if "runs" in self.results:
                for run in self.results["runs"]:
                    if "metrics" in run and metric_name in run["metrics"]:
                        values.append(run["metrics"][metric_name])
                    elif "speed" in run and metric_name in run["speed"]:
                        values.append(run["speed"][metric_name])
                    elif "per_class_ap" in run and metric_name in run["per_class_ap"]:
                        values.append(run["per_class_ap"][metric_name])

        self._values_cache[cache_key] = values
        return values

    def compute_confidence_interval(
        self,
        values: Optional[list[float]] = None,
        metric_name: str = "metric",
        confidence_level: float = 0.95,
    ) -> ConfidenceInterval:
        """
        Compute confidence interval for a set of values.

        Args:
            values: List of metric values (if None, extracts from results)
            metric_name: Name of the metric
            confidence_level: Confidence level (default 0.95 for 95% CI)

        Returns:
            ConfidenceInterval object
        """
        if values is None:
            values = self._extract_metric_values(metric_name)

        if not values or len(values) < 2:
            return ConfidenceInterval(
                metric_name=metric_name,
                mean=values[0] if values else 0.0,
                std=0.0,
                lower=values[0] if values else 0.0,
                upper=values[0] if values else 0.0,
                confidence_level=confidence_level,
                n_samples=len(values),
            )

        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample standard deviation
        se = std / np.sqrt(n)  # Standard error

        # Get t-critical value
        df = n - 1
        t_crit = self._get_t_critical(df, confidence_level)

        margin = t_crit * se
        lower = mean - margin
        upper = mean + margin

        return ConfidenceInterval(
            metric_name=metric_name,
            mean=float(mean),
            std=float(std),
            lower=float(lower),
            upper=float(upper),
            confidence_level=confidence_level,
            n_samples=n,
        )

    def compute_all_confidence_intervals(
        self,
        confidence_level: float = 0.95,
    ) -> dict[str, ConfidenceInterval]:
        """
        Compute confidence intervals for all metrics in results.

        Args:
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary mapping metric names to ConfidenceInterval objects
        """
        intervals = {}

        # Standard metrics
        standard_metrics = [
            "mAP@0.5",
            "mAP@0.75",
            "mAP@[.5:.95]",
            "precision",
            "recall",
            "f1_score",
        ]

        # Speed metrics
        speed_metrics = ["fps", "latency_ms", "latency_std_ms"]

        # Per-class metrics (extract from first run to get class names)
        per_class_metrics = []
        if self.results is not None:
            if hasattr(self.results, "runs") and self.results.runs:
                first_run = self.results.runs[0]
                if hasattr(first_run, "per_class_ap"):
                    per_class_metrics = list(first_run.per_class_ap.keys())
            elif isinstance(self.results, dict) and "runs" in self.results:
                if self.results["runs"]:
                    first_run = self.results["runs"][0]
                    if "per_class_ap" in first_run:
                        per_class_metrics = list(first_run["per_class_ap"].keys())

        all_metrics = standard_metrics + speed_metrics + per_class_metrics

        for metric in all_metrics:
            values = self._extract_metric_values(metric)
            if values:
                intervals[metric] = self.compute_confidence_interval(
                    values=values,
                    metric_name=metric,
                    confidence_level=confidence_level,
                )

        return intervals

    def test_normality(self, values: list[float], metric_name: str = "metric") -> dict:
        """
        Test if values follow a normal distribution using Shapiro-Wilk test approximation.

        Note: This is a simplified implementation. For production, consider using scipy.stats.

        Args:
            values: List of values to test
            metric_name: Name of the metric

        Returns:
            Dictionary with test results
        """
        if len(values) < 3:
            return {
                "metric": metric_name,
                "test": "shapiro-wilk",
                "statistic": None,
                "p_value": None,
                "is_normal": None,
                "message": "Not enough samples for normality test (n < 3)",
            }

        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)

        if std == 0:
            return {
                "metric": metric_name,
                "test": "shapiro-wilk",
                "statistic": None,
                "p_value": 1.0,
                "is_normal": True,
                "message": "All values are identical (zero variance)",
            }

        # Calculate skewness and kurtosis as simple normality indicators
        z_scores = [(x - mean) / std for x in values]
        skewness = np.mean([z**3 for z in z_scores])
        kurtosis = np.mean([z**4 for z in z_scores]) - 3  # Excess kurtosis

        # Simple heuristic: data is approximately normal if |skewness| < 2 and |kurtosis| < 7
        is_approximately_normal = abs(skewness) < 2 and abs(kurtosis) < 7

        # Jarque-Bera test statistic approximation
        jb_stat = (n / 6) * (skewness**2 + (kurtosis**2) / 4)

        # Approximate p-value (chi-square with 2 df)
        # This is a rough approximation
        p_value_approx = math.exp(-jb_stat / 2) if jb_stat < 20 else 0.0

        return {
            "metric": metric_name,
            "test": "jarque-bera-approximation",
            "statistic": float(jb_stat),
            "p_value": float(p_value_approx),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "is_normal": is_approximately_normal,
            "message": (
                "Data appears normally distributed"
                if is_approximately_normal
                else "Data may not be normally distributed"
            ),
        }

    def paired_t_test(
        self,
        values_a: list[float],
        values_b: list[float],
        model_a: str = "Model A",
        model_b: str = "Model B",
        metric: str = "metric",
        alpha: float = 0.05,
    ) -> SignificanceTestResult:
        """
        Perform paired t-test to compare two models.

        Args:
            values_a: Metric values for model A
            values_b: Metric values for model B
            model_a: Name of model A
            model_b: Name of model B
            metric: Name of the metric being compared
            alpha: Significance level

        Returns:
            SignificanceTestResult
        """
        if len(values_a) != len(values_b):
            raise ValueError(
                "Both value lists must have the same length for paired t-test"
            )

        if len(values_a) < 2:
            return SignificanceTestResult(
                test_name="paired_t_test",
                model_a=model_a,
                model_b=model_b,
                metric=metric,
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                alpha=alpha,
                conclusion="Not enough samples for t-test",
            )

        n = len(values_a)
        differences = [a - b for a, b in zip(values_a, values_b)]
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)

        if std_diff == 0:
            return SignificanceTestResult(
                test_name="paired_t_test",
                model_a=model_a,
                model_b=model_b,
                metric=metric,
                statistic=float("inf") if mean_diff != 0 else 0.0,
                p_value=0.0 if mean_diff != 0 else 1.0,
                is_significant=mean_diff != 0,
                alpha=alpha,
                effect_size=float("inf") if mean_diff != 0 else 0.0,
                conclusion=(
                    f"{model_a} is significantly different from {model_b}"
                    if mean_diff != 0
                    else "No difference between models"
                ),
            )

        # Calculate t-statistic
        se = std_diff / np.sqrt(n)
        t_stat = mean_diff / se
        df = n - 1

        # Get critical t-value for two-tailed test
        confidence = 1 - alpha
        t_crit = self._get_t_critical(df, confidence)

        # Approximate p-value (two-tailed)
        # Using approximation: p ≈ 2 * (1 - CDF(|t|))
        # For large |t|, p approaches 0
        abs_t = abs(t_stat)
        if abs_t > 10:
            p_value = 0.0001
        elif abs_t > t_crit:
            # Rough approximation
            p_value = alpha * (t_crit / abs_t) ** 2
        else:
            p_value = 2 * (1 - (0.5 + 0.5 * math.erf(abs_t / math.sqrt(2))))

        is_significant = abs_t > t_crit

        # Cohen's d effect size
        pooled_std = std_diff
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0

        # Interpret effect size
        if abs(effect_size) < 0.2:
            effect_interpretation = "negligible"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "small"
        elif abs(effect_size) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        if is_significant:
            better_model = model_a if mean_diff > 0 else model_b
            conclusion = (
                f"{better_model} performs significantly better "
                f"(p={p_value:.4f}, effect size: {effect_interpretation})"
            )
        else:
            conclusion = f"No significant difference between {model_a} and {model_b} (p={p_value:.4f})"

        return SignificanceTestResult(
            test_name="paired_t_test",
            model_a=model_a,
            model_b=model_b,
            metric=metric,
            statistic=float(t_stat),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=alpha,
            effect_size=float(effect_size),
            conclusion=conclusion,
        )

    def compute_stability_score(self, values: list[float]) -> float:
        """
        Compute a stability score based on coefficient of variation.

        Lower scores indicate more stable/consistent results.

        Args:
            values: List of metric values across runs

        Returns:
            Stability score (coefficient of variation as percentage)
        """
        if not values or len(values) < 2:
            return 0.0

        mean = np.mean(values)
        std = np.std(values, ddof=1)

        if mean == 0:
            return 0.0 if std == 0 else 100.0

        cv = (std / abs(mean)) * 100
        return float(cv)

    def get_reliability_rating(self, stability_score: float) -> str:
        """
        Convert stability score to reliability rating.

        Args:
            stability_score: Coefficient of variation percentage

        Returns:
            Reliability rating string
        """
        if stability_score < 1.0:
            return "Excellent"
        elif stability_score < 3.0:
            return "High"
        elif stability_score < 5.0:
            return "Good"
        elif stability_score < 10.0:
            return "Medium"
        else:
            return "Low"

    def generate_summary(self, confidence_level: float = 0.95) -> StatisticalSummary:
        """
        Generate a complete statistical summary for the results.

        Args:
            confidence_level: Confidence level for intervals

        Returns:
            StatisticalSummary object
        """
        model_name = "Unknown"
        n_runs = 0

        if self.results is not None:
            if hasattr(self.results, "model_name"):
                model_name = self.results.model_name
            elif isinstance(self.results, dict) and "model" in self.results:
                model_name = self.results["model"].get("name", "Unknown")

            if hasattr(self.results, "num_runs"):
                n_runs = self.results.num_runs
            elif isinstance(self.results, dict):
                n_runs = self.results.get("num_runs", 0)

        summary = StatisticalSummary(
            model_name=model_name,
            n_runs=n_runs,
        )

        # Compute confidence intervals
        summary.confidence_intervals = self.compute_all_confidence_intervals(
            confidence_level=confidence_level
        )

        # Run normality tests on main metrics
        main_metrics = ["mAP@0.5", "mAP@[.5:.95]", "precision", "recall"]
        for metric in main_metrics:
            values = self._extract_metric_values(metric)
            if values:
                summary.normality_tests[metric] = self.test_normality(values, metric)

        # Compute overall stability score (average CV of main metrics)
        stability_scores = []
        for metric in main_metrics:
            values = self._extract_metric_values(metric)
            if values:
                stability_scores.append(self.compute_stability_score(values))

        if stability_scores:
            summary.stability_score = float(np.mean(stability_scores))
            summary.reliability_rating = self.get_reliability_rating(
                summary.stability_score
            )

        return summary

    def print_summary(self, confidence_level: float = 0.95):
        """Print a formatted summary of statistical analysis."""
        summary = self.generate_summary(confidence_level)

        print(f"\n{'=' * 60}")
        print(f"Statistical Analysis: {summary.model_name}")
        print(f"{'=' * 60}")
        print(f"Number of runs: {summary.n_runs}")
        print(f"Stability Score: {summary.stability_score:.2f}% (CV)")
        print(f"Reliability Rating: {summary.reliability_rating}")

        print(f"\n{int(confidence_level * 100)}% Confidence Intervals:")
        print("-" * 60)

        for name, ci in summary.confidence_intervals.items():
            if name in [
                "mAP@0.5",
                "mAP@0.75",
                "mAP@[.5:.95]",
                "precision",
                "recall",
                "f1_score",
            ]:
                print(
                    f"  {name}: {ci.mean:.4f} ± {ci.std:.4f} [{ci.lower:.4f}, {ci.upper:.4f}]"
                )

        print(f"\nSpeed Metrics:")
        print("-" * 60)
        for name, ci in summary.confidence_intervals.items():
            if name in ["fps", "latency_ms"]:
                unit = "fps" if name == "fps" else "ms"
                print(
                    f"  {name}: {ci.mean:.2f} ± {ci.std:.2f} [{ci.lower:.2f}, {ci.upper:.2f}] {unit}"
                )

        print(f"\nNormality Tests:")
        print("-" * 60)
        for name, test in summary.normality_tests.items():
            status = "✓ Normal" if test.get("is_normal") else "✗ Non-normal"
            print(f"  {name}: {status}")

        print(f"{'=' * 60}\n")
