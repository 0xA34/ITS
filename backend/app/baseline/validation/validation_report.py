"""
Validation Report Generator Module
===================================
Generate comprehensive validation reports combining cross-validation results,
statistical analysis, benchmark comparisons, and confusion matrices.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationReportData:
    """Complete validation report data."""

    model_name: str
    timestamp: str

    # Cross-validation results
    cross_validation: Optional[dict] = None

    # Statistical analysis
    statistical_analysis: Optional[dict] = None

    # Benchmark comparison
    benchmark_comparison: Optional[dict] = None

    # Confusion matrix
    confusion_matrix: Optional[dict] = None

    # Overall validation status
    overall_status: str = "UNKNOWN"
    validation_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "cross_validation": self.cross_validation,
            "statistical_analysis": self.statistical_analysis,
            "benchmark_comparison": self.benchmark_comparison,
            "confusion_matrix": self.confusion_matrix,
            "overall_status": self.overall_status,
            "validation_score": self.validation_score,
            "recommendations": self.recommendations,
        }


class ValidationReport:
    """
    Generate comprehensive validation reports.

    This class combines results from various validation components:
    - Cross-validation (multiple runs)
    - Statistical analysis (confidence intervals, significance tests)
    - Benchmark comparison (official benchmarks)
    - Confusion matrix analysis

    And generates reports in multiple formats (JSON, Markdown, HTML).
    """

    def __init__(
        self,
        model_name: str,
        cross_validation_results: Optional[Any] = None,
        statistical_summary: Optional[Any] = None,
        benchmark_comparison: Optional[Any] = None,
        confusion_matrix_data: Optional[Any] = None,
    ):
        """
        Initialize ValidationReport.

        Args:
            model_name: Name of the model being validated
            cross_validation_results: Results from CrossValidator
            statistical_summary: Results from StatisticalAnalyzer
            benchmark_comparison: Results from BenchmarkComparator
            confusion_matrix_data: Results from ConfusionMatrixGenerator
        """
        self.model_name = model_name
        self.cross_validation = cross_validation_results
        self.statistical_summary = statistical_summary
        self.benchmark_comparison = benchmark_comparison
        self.confusion_matrix = confusion_matrix_data
        self.timestamp = datetime.now().isoformat()

    def _convert_to_dict(self, obj: Any) -> Optional[dict]:
        """Convert various result objects to dictionaries."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return None

    def _calculate_validation_score(self) -> tuple[float, str]:
        """
        Calculate an overall validation score.

        Returns:
            Tuple of (score, status)
            Score is 0-100, status is "PASS", "WARN", or "FAIL"
        """
        scores = []

        # Score from statistical reliability
        if self.statistical_summary is not None:
            stat_dict = self._convert_to_dict(self.statistical_summary)
            if stat_dict:
                reliability = stat_dict.get("reliability_rating", "")
                reliability_scores = {
                    "Excellent": 100,
                    "High": 90,
                    "Good": 75,
                    "Medium": 50,
                    "Low": 25,
                }
                if reliability in reliability_scores:
                    scores.append(reliability_scores[reliability])

        # Score from benchmark comparison
        if self.benchmark_comparison is not None:
            bench_dict = self._convert_to_dict(self.benchmark_comparison)
            if bench_dict:
                status = bench_dict.get("overall_status", "")
                if status == "PASS":
                    scores.append(100)
                elif status == "WARN":
                    scores.append(70)
                elif status == "FAIL":
                    scores.append(30)

        # Score from cross-validation consistency
        if self.cross_validation is not None:
            cv_dict = self._convert_to_dict(self.cross_validation)
            if cv_dict and "aggregated" in cv_dict:
                agg = cv_dict["aggregated"]
                if "metrics" in agg and "std" in agg["metrics"]:
                    std_metrics = agg["metrics"]["std"]
                    # Lower std = higher score
                    avg_std = (
                        sum(std_metrics.values()) / len(std_metrics)
                        if std_metrics
                        else 0
                    )
                    if avg_std < 0.01:
                        scores.append(100)
                    elif avg_std < 0.02:
                        scores.append(85)
                    elif avg_std < 0.05:
                        scores.append(70)
                    else:
                        scores.append(50)

        if not scores:
            return 0.0, "UNKNOWN"

        avg_score = sum(scores) / len(scores)

        if avg_score >= 80:
            status = "PASS"
        elif avg_score >= 50:
            status = "WARN"
        else:
            status = "FAIL"

        return avg_score, status

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Check statistical reliability
        if self.statistical_summary is not None:
            stat_dict = self._convert_to_dict(self.statistical_summary)
            if stat_dict:
                reliability = stat_dict.get("reliability_rating", "")
                if reliability in ["Low", "Medium"]:
                    recommendations.append(
                        "Consider running more evaluation iterations to improve statistical reliability."
                    )

                stability = stat_dict.get("stability_score", 0)
                if stability > 5:
                    recommendations.append(
                        f"High variance detected (CV={stability:.2f}%). Results may not be stable."
                    )

        # Check benchmark comparison
        if self.benchmark_comparison is not None:
            bench_dict = self._convert_to_dict(self.benchmark_comparison)
            if bench_dict:
                if bench_dict.get("overall_status") == "FAIL":
                    recommendations.append(
                        "Results significantly differ from official benchmarks. "
                        "Verify evaluation methodology matches official setup."
                    )

                # Add specific recommendations from comparison
                if "recommendations" in bench_dict:
                    recommendations.extend(bench_dict["recommendations"])

        # Check cross-validation
        if self.cross_validation is not None:
            cv_dict = self._convert_to_dict(self.cross_validation)
            if cv_dict:
                num_runs = cv_dict.get("num_runs", 0)
                if num_runs < 3:
                    recommendations.append(
                        "Only {num_runs} evaluation run(s) performed. "
                        "Recommend at least 3-5 runs for reliable statistics."
                    )

        if not recommendations:
            recommendations.append(
                "Validation passed! Results are statistically reliable and "
                "consistent with official benchmarks."
            )

        return recommendations

    def compile_report(self) -> ValidationReportData:
        """Compile all validation data into a report."""
        score, status = self._calculate_validation_score()
        recommendations = self._generate_recommendations()

        return ValidationReportData(
            model_name=self.model_name,
            timestamp=self.timestamp,
            cross_validation=self._convert_to_dict(self.cross_validation),
            statistical_analysis=self._convert_to_dict(self.statistical_summary),
            benchmark_comparison=self._convert_to_dict(self.benchmark_comparison),
            confusion_matrix=self._convert_to_dict(self.confusion_matrix),
            overall_status=status,
            validation_score=score,
            recommendations=recommendations,
        )

    def generate_json(self) -> str:
        """Generate JSON report."""
        report = self.compile_report()
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)

    def generate_markdown(self) -> str:
        """Generate Markdown report."""
        report = self.compile_report()

        lines = []
        lines.append(f"# Validation Report: {report.model_name}")
        lines.append("")
        lines.append(f"**Generated:** {report.timestamp}")
        lines.append(f"**Overall Status:** {report.overall_status}")
        lines.append(f"**Validation Score:** {report.validation_score:.1f}/100")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        status_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "UNKNOWN": "❓"}

        if report.overall_status == "PASS":
            status_text = "Results are validated and reliable."
        elif report.overall_status == "WARN":
            status_text = "Results are acceptable but have some concerns."
        elif report.overall_status == "FAIL":
            status_text = "Results have significant issues that need attention."
        else:
            status_text = "Unable to determine validation status."

        lines.append(
            f"{status_emoji.get(report.overall_status, '❓')} **{report.overall_status}** - {status_text}"
        )
        lines.append("")

        # Cross-validation results
        if report.cross_validation:
            lines.append("## Cross-Validation Results")
            lines.append("")
            cv = report.cross_validation
            lines.append(f"- **Number of runs:** {cv.get('num_runs', 'N/A')}")

            if "aggregated" in cv and "metrics" in cv["aggregated"]:
                metrics = cv["aggregated"]["metrics"]
                if "mean" in metrics:
                    lines.append("")
                    lines.append("### Aggregated Metrics (mean ± std)")
                    lines.append("")
                    lines.append("| Metric | Mean | Std |")
                    lines.append("|--------|------|-----|")

                    mean_metrics = metrics["mean"]
                    std_metrics = metrics.get("std", {})

                    for key in [
                        "mAP@0.5",
                        "mAP@0.75",
                        "mAP@[.5:.95]",
                        "precision",
                        "recall",
                        "f1_score",
                    ]:
                        if key in mean_metrics:
                            mean_val = mean_metrics[key]
                            std_val = std_metrics.get(key, 0)
                            lines.append(f"| {key} | {mean_val:.4f} | ±{std_val:.4f} |")
            lines.append("")

        # Statistical analysis
        if report.statistical_analysis:
            lines.append("## Statistical Analysis")
            lines.append("")
            stat = report.statistical_analysis

            lines.append(
                f"- **Stability Score:** {stat.get('stability_score', 0):.2f}% (CV)"
            )
            lines.append(
                f"- **Reliability Rating:** {stat.get('reliability_rating', 'N/A')}"
            )
            lines.append("")

            if "confidence_intervals" in stat:
                lines.append("### 95% Confidence Intervals")
                lines.append("")
                lines.append("| Metric | Mean | Lower | Upper |")
                lines.append("|--------|------|-------|-------|")

                for key, ci in stat["confidence_intervals"].items():
                    if key in ["mAP@0.5", "mAP@[.5:.95]", "precision", "recall"]:
                        lines.append(
                            f"| {ci['metric_name']} | {ci['mean']:.4f} | "
                            f"{ci['lower']:.4f} | {ci['upper']:.4f} |"
                        )
            lines.append("")

        # Benchmark comparison
        if report.benchmark_comparison:
            lines.append("## Benchmark Comparison")
            lines.append("")
            bench = report.benchmark_comparison

            lines.append(f"- **Status:** {bench.get('overall_status', 'N/A')}")
            lines.append(
                f"- **Passed:** {bench.get('passed', 0)}/{bench.get('total_metrics', 0)}"
            )
            lines.append("")

            if "comparisons" in bench and bench["comparisons"]:
                lines.append("### Detailed Comparison")
                lines.append("")
                lines.append("| Metric | Ours | Official | Diff % | Status |")
                lines.append("|--------|------|----------|--------|--------|")

                for comp in bench["comparisons"]:
                    status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(
                        comp["status"], ""
                    )
                    lines.append(
                        f"| {comp['metric_name']} | {comp['our_value']:.2f} | "
                        f"{comp['official_value']:.2f} | {comp['percentage_diff']:+.1f}% | "
                        f"{status_icon} {comp['status']} |"
                    )
            lines.append("")

        # Confusion matrix summary
        if report.confusion_matrix:
            lines.append("## Confusion Matrix Summary")
            lines.append("")
            cm = report.confusion_matrix

            lines.append(
                f"- **Total Predictions:** {cm.get('total_predictions', 'N/A')}"
            )
            lines.append(
                f"- **Total Ground Truths:** {cm.get('total_ground_truths', 'N/A')}"
            )

            if "per_class_metrics" in cm:
                lines.append("")
                lines.append("### Per-Class Performance")
                lines.append("")
                lines.append("| Class | Precision | Recall | F1 |")
                lines.append("|-------|-----------|--------|-----|")

                for class_name, m in cm["per_class_metrics"].items():
                    lines.append(
                        f"| {class_name} | {m['precision']:.4f} | "
                        f"{m['recall']:.4f} | {m['f1_score']:.4f} |"
                    )
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

        return "\n".join(lines)

    def generate_html(self) -> str:
        """Generate HTML report."""
        report = self.compile_report()

        status_colors = {
            "PASS": "#4caf50",
            "WARN": "#ff9800",
            "FAIL": "#f44336",
            "UNKNOWN": "#9e9e9e",
        }
        status_color = status_colors.get(report.overall_status, "#9e9e9e")

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Validation Report - {report.model_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 40px;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .header .timestamp {{
            opacity: 0.8;
        }}
        .content {{
            padding: 40px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.1em;
            background: {status_color};
            color: white;
            margin-bottom: 20px;
        }}
        .score-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
        }}
        .score-value {{
            font-size: 4em;
            font-weight: bold;
            color: {status_color};
        }}
        .score-label {{
            color: #666;
            font-size: 1.2em;
        }}
        h2 {{
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .metric-good {{ color: #4caf50; font-weight: bold; }}
        .metric-medium {{ color: #ff9800; font-weight: bold; }}
        .metric-poor {{ color: #f44336; font-weight: bold; }}
        .recommendation {{
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px 20px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
        }}
        .card h3 {{
            margin-top: 0;
            color: #555;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Validation Report</h1>
            <div class="model-name">Model: <strong>{report.model_name}</strong></div>
            <div class="timestamp">Generated: {report.timestamp}</div>
        </div>

        <div class="content">
            <div class="status-badge">{report.overall_status}</div>

            <div class="score-card">
                <div class="score-value">{report.validation_score:.0f}</div>
                <div class="score-label">Validation Score (out of 100)</div>
            </div>
"""

        # Cross-validation section
        if report.cross_validation:
            cv = report.cross_validation
            html += """
            <h2>📊 Cross-Validation Results</h2>
"""
            html += (
                f"<p><strong>Number of runs:</strong> {cv.get('num_runs', 'N/A')}</p>"
            )

            if "aggregated" in cv and "metrics" in cv["aggregated"]:
                metrics = cv["aggregated"]["metrics"]
                if "mean" in metrics:
                    html += """
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
"""
                    mean_m = metrics["mean"]
                    std_m = metrics.get("std", {})
                    min_m = metrics.get("min", {})
                    max_m = metrics.get("max", {})

                    for key in [
                        "mAP@0.5",
                        "mAP@0.75",
                        "mAP@[.5:.95]",
                        "precision",
                        "recall",
                        "f1_score",
                    ]:
                        if key in mean_m:
                            html += f"""
                <tr>
                    <td>{key}</td>
                    <td><strong>{mean_m[key]:.4f}</strong></td>
                    <td>±{std_m.get(key, 0):.4f}</td>
                    <td>{min_m.get(key, 0):.4f}</td>
                    <td>{max_m.get(key, 0):.4f}</td>
                </tr>
"""
                    html += "</table>"

        # Statistical analysis section
        if report.statistical_analysis:
            stat = report.statistical_analysis
            html += """
            <h2>📈 Statistical Analysis</h2>
            <div class="grid">
                <div class="card">
                    <h3>Stability Score</h3>
"""
            html += f"<p style='font-size: 2em; font-weight: bold;'>{stat.get('stability_score', 0):.2f}%</p>"
            html += "<p>Coefficient of Variation (lower is better)</p>"
            html += """
                </div>
                <div class="card">
                    <h3>Reliability Rating</h3>
"""
            html += f"<p style='font-size: 2em; font-weight: bold;'>{stat.get('reliability_rating', 'N/A')}</p>"
            html += """
                </div>
            </div>
"""

        # Benchmark comparison section
        if report.benchmark_comparison:
            bench = report.benchmark_comparison
            html += """
            <h2>🎯 Benchmark Comparison</h2>
"""
            html += (
                f"<p><strong>Status:</strong> {bench.get('overall_status', 'N/A')} | "
            )
            html += f"<strong>Passed:</strong> {bench.get('passed', 0)}/{bench.get('total_metrics', 0)}</p>"

            if "comparisons" in bench and bench["comparisons"]:
                html += """
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Our Value</th>
                    <th>Official</th>
                    <th>Difference</th>
                    <th>Status</th>
                </tr>
"""
                for comp in bench["comparisons"]:
                    diff_class = (
                        "metric-good"
                        if abs(comp["percentage_diff"]) < 5
                        else "metric-medium"
                        if abs(comp["percentage_diff"]) < 15
                        else "metric-poor"
                    )
                    status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(
                        comp["status"], ""
                    )
                    html += f"""
                <tr>
                    <td>{comp["metric_name"]}</td>
                    <td>{comp["our_value"]:.2f}</td>
                    <td>{comp["official_value"]:.2f}</td>
                    <td class="{diff_class}">{comp["percentage_diff"]:+.1f}%</td>
                    <td>{status_icon} {comp["status"]}</td>
                </tr>
"""
                html += "</table>"

        # Recommendations section
        html += """
            <h2>💡 Recommendations</h2>
"""
        for rec in report.recommendations:
            html += f'<div class="recommendation">{rec}</div>'

        html += """
        </div>
    </div>
</body>
</html>
"""
        return html

    def save(self, output_path: str, formats: Optional[list[str]] = None):
        """
        Save validation report to files.

        Args:
            output_path: Base output path (without extension)
            formats: List of formats ('json', 'md', 'html'). Default: all.
        """
        if formats is None:
            formats = ["json", "md", "html"]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if "json" in formats:
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(self.generate_json())
            logger.info(f"Saved validation report JSON to {json_path}")

        if "md" in formats:
            md_path = output_path.with_suffix(".md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(self.generate_markdown())
            logger.info(f"Saved validation report Markdown to {md_path}")

        if "html" in formats:
            html_path = output_path.with_suffix(".html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self.generate_html())
            logger.info(f"Saved validation report HTML to {html_path}")

    def print_summary(self):
        """Print a summary of the validation report."""
        report = self.compile_report()

        print(f"\n{'=' * 60}")
        print(f"Validation Report: {report.model_name}")
        print(f"{'=' * 60}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Overall Status: {report.overall_status}")
        print(f"Validation Score: {report.validation_score:.1f}/100")

        print(f"\nRecommendations:")
        print("-" * 60)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

        print(f"{'=' * 60}\n")


def generate_validation_report(
    model_name: str,
    cross_validation_results: Optional[Any] = None,
    statistical_summary: Optional[Any] = None,
    benchmark_comparison: Optional[Any] = None,
    confusion_matrix_data: Optional[Any] = None,
    output_path: Optional[str] = None,
    formats: Optional[list[str]] = None,
) -> ValidationReport:
    """
    Convenience function to generate a validation report.

    Args:
        model_name: Name of the model
        cross_validation_results: CrossValidationResult object or dict
        statistical_summary: StatisticalSummary object or dict
        benchmark_comparison: ValidationSummary object or dict
        confusion_matrix_data: ConfusionMatrixData object or dict
        output_path: Path to save reports (optional)
        formats: List of output formats (optional)

    Returns:
        ValidationReport instance
    """
    report = ValidationReport(
        model_name=model_name,
        cross_validation_results=cross_validation_results,
        statistical_summary=statistical_summary,
        benchmark_comparison=benchmark_comparison,
        confusion_matrix_data=confusion_matrix_data,
    )

    if output_path:
        report.save(output_path, formats)

    return report
