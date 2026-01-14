"""
Model Comparison and Report Generation Module
==============================================
Compare detection models and generate comprehensive reports.

Features:
    - Compare accuracy metrics (mAP, Precision, Recall)
    - Compare speed metrics (FPS, Latency)
    - Compare model size and complexity
    - Generate HTML, Markdown, and JSON reports
    - Visualizations (charts, tables)

Usage:
    from app.baseline.comparison import ComparisonReport

    report = ComparisonReport()
    report.add_model_results("YOLOv8n", accuracy_results, benchmark_results)
    report.add_model_results("YOLOv8s", accuracy_results, benchmark_results)
    report.add_reference_methods(reference_data)
    report.generate_report("output/comparison_report")
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for a single model."""

    model_name: str

    # Accuracy metrics
    map_50: float = 0.0
    map_75: float = 0.0
    map_50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Per-class AP
    ap_per_class: dict[str, float] = field(default_factory=dict)

    # Speed metrics
    fps: float = 0.0
    latency_ms: float = 0.0
    latency_std_ms: float = 0.0

    # Model info
    params_millions: float = 0.0
    flops_billions: float = 0.0
    model_size_mb: float = 0.0

    # Additional info
    device: str = ""
    image_size: tuple[int, int] = (640, 640)
    source: str = "Evaluated"  # "Evaluated", "Reference", "Official"

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "accuracy": {
                "mAP@0.5": self.map_50,
                "mAP@0.75": self.map_75,
                "mAP@[.5:.95]": self.map_50_95,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "ap_per_class": self.ap_per_class,
            },
            "speed": {
                "fps": self.fps,
                "latency_ms": self.latency_ms,
                "latency_std_ms": self.latency_std_ms,
            },
            "model_info": {
                "params_millions": self.params_millions,
                "flops_billions": self.flops_billions,
                "model_size_mb": self.model_size_mb,
            },
            "config": {
                "device": self.device,
                "image_size": self.image_size,
                "source": self.source,
            },
        }


@dataclass
class ComparisonResult:
    """Result of model comparison."""

    models: list[ModelMetrics] = field(default_factory=list)
    best_accuracy: Optional[str] = None
    best_speed: Optional[str] = None
    best_balanced: Optional[str] = None  # Best accuracy/speed trade-off
    comparison_date: str = ""
    dataset_info: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.comparison_date:
            self.comparison_date = datetime.now().isoformat()


class ComparisonReport:
    """
    Generate comparison reports for multiple detection models.

    Usage:
        report = ComparisonReport(title="Vehicle Detection Model Comparison")

        # Add evaluated model results
        report.add_model_results(
            model_name="YOLOv8n",
            map_result=map_result,  # MAPResult object
            benchmark_result=benchmark_result,  # BenchmarkResult object
        )

        # Add reference methods from papers/official sources
        report.add_reference_method(
            model_name="Faster R-CNN",
            map_50=0.42,
            fps=15,
            source="Torchvision Official"
        )

        # Generate reports
        report.generate_report("output/comparison")
    """

    def __init__(
        self,
        title: str = "Vehicle Detection Model Comparison",
        description: str = "",
    ):
        """
        Initialize comparison report.

        Args:
            title: Report title
            description: Report description
        """
        self.title = title
        self.description = description
        self.models: dict[str, ModelMetrics] = {}
        self.dataset_info: dict[str, Any] = {}
        self.notes: list[str] = []
        self.created_at = datetime.now()

    def add_model_results(
        self,
        model_name: str,
        map_result: Optional[Any] = None,  # MAPResult
        benchmark_result: Optional[Any] = None,  # BenchmarkResult
        precision: float = 0.0,
        recall: float = 0.0,
        f1_score: float = 0.0,
        params_millions: float = 0.0,
        flops_billions: float = 0.0,
        model_size_mb: float = 0.0,
    ):
        """
        Add results for an evaluated model.

        Args:
            model_name: Name of the model
            map_result: MAPResult object from evaluation
            benchmark_result: BenchmarkResult object from FPS benchmark
            precision: Overall precision
            recall: Overall recall
            f1_score: Overall F1 score
            params_millions: Number of parameters in millions
            flops_billions: FLOPs in billions
            model_size_mb: Model file size in MB
        """
        metrics = ModelMetrics(model_name=model_name, source="Evaluated")

        # Extract from MAPResult
        if map_result is not None:
            metrics.map_50 = getattr(map_result, "map_50", 0.0) or 0.0
            metrics.map_75 = getattr(map_result, "map_75", 0.0) or 0.0
            metrics.map_50_95 = getattr(map_result, "map_50_95", 0.0) or 0.0

            # Extract per-class AP
            ap_per_class = getattr(map_result, "ap_per_class", {})
            if ap_per_class:
                for class_name, ap_result in ap_per_class.items():
                    if hasattr(ap_result, "ap"):
                        metrics.ap_per_class[class_name] = ap_result.ap
                    else:
                        metrics.ap_per_class[class_name] = float(ap_result)

        # Extract from BenchmarkResult
        if benchmark_result is not None:
            metrics.fps = getattr(benchmark_result, "fps", 0.0)
            metrics.latency_ms = getattr(benchmark_result, "latency_ms", 0.0)
            metrics.latency_std_ms = getattr(benchmark_result, "latency_std_ms", 0.0)
            metrics.device = getattr(benchmark_result, "device", "")
            metrics.image_size = getattr(benchmark_result, "image_size", (640, 640))

        # Set additional metrics
        metrics.precision = precision
        metrics.recall = recall
        metrics.f1_score = f1_score
        metrics.params_millions = params_millions
        metrics.flops_billions = flops_billions
        metrics.model_size_mb = model_size_mb

        self.models[model_name] = metrics
        logger.info(f"Added model: {model_name}")

    def add_reference_method(
        self,
        model_name: str,
        map_50: float = 0.0,
        map_75: float = 0.0,
        map_50_95: float = 0.0,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        params_millions: float = 0.0,
        source: str = "Reference",
    ):
        """
        Add reference method from papers or official sources.

        Args:
            model_name: Name of the method
            map_50: mAP@0.5
            map_75: mAP@0.75
            map_50_95: mAP@[.5:.95]
            fps: Frames per second
            latency_ms: Latency in milliseconds
            params_millions: Number of parameters in millions
            source: Source of the metrics
        """
        metrics = ModelMetrics(
            model_name=model_name,
            map_50=map_50,
            map_75=map_75,
            map_50_95=map_50_95,
            fps=fps,
            latency_ms=latency_ms if latency_ms > 0 else (1000 / fps if fps > 0 else 0),
            params_millions=params_millions,
            source=source,
        )

        self.models[model_name] = metrics
        logger.info(f"Added reference method: {model_name} ({source})")

    def add_reference_methods_from_config(self, reference_methods: dict[str, dict]):
        """
        Add multiple reference methods from configuration.

        Args:
            reference_methods: Dictionary of method name to metrics
        """
        for name, data in reference_methods.items():
            self.add_reference_method(
                model_name=name,
                map_50=data.get("mAP50", 0.0),
                map_75=data.get("mAP75", 0.0),
                map_50_95=data.get("mAP50-95", 0.0),
                fps=data.get("fps", 0.0),
                params_millions=data.get("params_m", 0.0),
                source=data.get("source", "Reference"),
            )

    def set_dataset_info(
        self,
        name: str,
        num_images: int = 0,
        num_annotations: int = 0,
        classes: Optional[list[str]] = None,
        split: str = "val",
    ):
        """Set information about the evaluation dataset."""
        self.dataset_info = {
            "name": name,
            "num_images": num_images,
            "num_annotations": num_annotations,
            "classes": classes or [],
            "split": split,
        }

    def add_note(self, note: str):
        """Add a note to the report."""
        self.notes.append(note)

    def _calculate_rankings(self) -> dict[str, list[tuple[str, float]]]:
        """Calculate rankings for different metrics."""
        rankings = {
            "accuracy_map50": [],
            "accuracy_map50_95": [],
            "speed_fps": [],
            "speed_latency": [],
            "efficiency": [],  # mAP/params ratio
        }

        for name, metrics in self.models.items():
            rankings["accuracy_map50"].append((name, metrics.map_50))
            rankings["accuracy_map50_95"].append((name, metrics.map_50_95))
            rankings["speed_fps"].append((name, metrics.fps))
            rankings["speed_latency"].append((name, metrics.latency_ms))

            # Efficiency: mAP per million parameters
            if metrics.params_millions > 0:
                efficiency = metrics.map_50 / metrics.params_millions
            else:
                efficiency = 0.0
            rankings["efficiency"].append((name, efficiency))

        # Sort rankings
        rankings["accuracy_map50"].sort(key=lambda x: x[1], reverse=True)
        rankings["accuracy_map50_95"].sort(key=lambda x: x[1], reverse=True)
        rankings["speed_fps"].sort(key=lambda x: x[1], reverse=True)
        rankings["speed_latency"].sort(key=lambda x: x[1])  # Lower is better
        rankings["efficiency"].sort(key=lambda x: x[1], reverse=True)

        return rankings

    def _find_best_models(self) -> dict[str, str]:
        """Find best models for different criteria."""
        rankings = self._calculate_rankings()

        best = {}

        if rankings["accuracy_map50"]:
            best["accuracy"] = rankings["accuracy_map50"][0][0]

        if rankings["speed_fps"]:
            best["speed"] = rankings["speed_fps"][0][0]

        if rankings["efficiency"]:
            best["efficiency"] = rankings["efficiency"][0][0]

        # Best balanced: highest (normalized_mAP + normalized_FPS) / 2
        if len(self.models) > 0:
            map_values = [m.map_50 for m in self.models.values()]
            fps_values = [m.fps for m in self.models.values()]

            max_map = max(map_values) if map_values else 1.0
            max_fps = max(fps_values) if fps_values else 1.0

            best_balanced_score = -1
            best_balanced_name = None

            for name, metrics in self.models.items():
                norm_map = metrics.map_50 / max_map if max_map > 0 else 0
                norm_fps = metrics.fps / max_fps if max_fps > 0 else 0
                balanced_score = (norm_map + norm_fps) / 2

                if balanced_score > best_balanced_score:
                    best_balanced_score = balanced_score
                    best_balanced_name = name

            if best_balanced_name:
                best["balanced"] = best_balanced_name

        return best

    def generate_comparison_result(self) -> ComparisonResult:
        """Generate comparison result object."""
        best = self._find_best_models()

        return ComparisonResult(
            models=list(self.models.values()),
            best_accuracy=best.get("accuracy"),
            best_speed=best.get("speed"),
            best_balanced=best.get("balanced"),
            dataset_info=self.dataset_info,
            notes=self.notes,
        )

    def generate_markdown(self) -> str:
        """Generate Markdown report."""
        lines = []
        best = self._find_best_models()
        rankings = self._calculate_rankings()

        # Title
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"*Generated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("")

        if self.description:
            lines.append(self.description)
            lines.append("")

        # Dataset Info
        if self.dataset_info:
            lines.append("## Dataset Information")
            lines.append("")
            lines.append(f"- **Name**: {self.dataset_info.get('name', 'N/A')}")
            lines.append(f"- **Split**: {self.dataset_info.get('split', 'N/A')}")
            lines.append(
                f"- **Images**: {self.dataset_info.get('num_images', 'N/A'):,}"
            )
            lines.append(
                f"- **Annotations**: {self.dataset_info.get('num_annotations', 'N/A'):,}"
            )
            if self.dataset_info.get("classes"):
                lines.append(
                    f"- **Classes**: {', '.join(self.dataset_info['classes'])}"
                )
            lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Best Accuracy (mAP@0.5)**: {best.get('accuracy', 'N/A')}")
        lines.append(f"- **Best Speed (FPS)**: {best.get('speed', 'N/A')}")
        lines.append(f"- **Best Balanced**: {best.get('balanced', 'N/A')}")
        lines.append(f"- **Best Efficiency**: {best.get('efficiency', 'N/A')}")
        lines.append("")

        # Accuracy Comparison Table
        lines.append("## Accuracy Comparison")
        lines.append("")
        lines.append(
            "| Model | mAP@0.5 | mAP@0.75 | mAP@[.5:.95] | Precision | Recall | F1 | Source |"
        )
        lines.append(
            "|-------|---------|----------|--------------|-----------|--------|-------|--------|"
        )

        for name, metrics in sorted(
            self.models.items(), key=lambda x: x[1].map_50, reverse=True
        ):
            map50 = f"{metrics.map_50:.4f}" if metrics.map_50 > 0 else "-"
            map75 = f"{metrics.map_75:.4f}" if metrics.map_75 > 0 else "-"
            map5095 = f"{metrics.map_50_95:.4f}" if metrics.map_50_95 > 0 else "-"
            prec = f"{metrics.precision:.4f}" if metrics.precision > 0 else "-"
            rec = f"{metrics.recall:.4f}" if metrics.recall > 0 else "-"
            f1 = f"{metrics.f1_score:.4f}" if metrics.f1_score > 0 else "-"

            # Highlight best accuracy
            name_display = f"**{name}**" if name == best.get("accuracy") else name

            lines.append(
                f"| {name_display} | {map50} | {map75} | {map5095} | {prec} | {rec} | {f1} | {metrics.source} |"
            )

        lines.append("")

        # Speed Comparison Table
        lines.append("## Speed Comparison")
        lines.append("")
        lines.append("| Model | FPS | Latency (ms) | Device | Image Size | Source |")
        lines.append("|-------|-----|--------------|--------|------------|--------|")

        for name, metrics in sorted(
            self.models.items(), key=lambda x: x[1].fps, reverse=True
        ):
            fps = f"{metrics.fps:.1f}" if metrics.fps > 0 else "-"
            latency = f"{metrics.latency_ms:.2f}" if metrics.latency_ms > 0 else "-"
            if metrics.latency_std_ms > 0:
                latency += f" ± {metrics.latency_std_ms:.2f}"
            device = metrics.device or "-"
            size = (
                f"{metrics.image_size[0]}x{metrics.image_size[1]}"
                if metrics.image_size
                else "-"
            )

            # Highlight best speed
            name_display = f"**{name}**" if name == best.get("speed") else name

            lines.append(
                f"| {name_display} | {fps} | {latency} | {device} | {size} | {metrics.source} |"
            )

        lines.append("")

        # Model Complexity Table
        lines.append("## Model Complexity")
        lines.append("")
        lines.append("| Model | Parameters (M) | FLOPs (B) | Size (MB) |")
        lines.append("|-------|----------------|-----------|-----------|")

        for name, metrics in sorted(
            self.models.items(), key=lambda x: x[1].params_millions
        ):
            params = (
                f"{metrics.params_millions:.1f}" if metrics.params_millions > 0 else "-"
            )
            flops = (
                f"{metrics.flops_billions:.1f}" if metrics.flops_billions > 0 else "-"
            )
            size = f"{metrics.model_size_mb:.1f}" if metrics.model_size_mb > 0 else "-"

            lines.append(f"| {name} | {params} | {flops} | {size} |")

        lines.append("")

        # Per-Class AP (if available)
        all_classes = set()
        for metrics in self.models.values():
            all_classes.update(metrics.ap_per_class.keys())

        if all_classes:
            lines.append("## Per-Class Average Precision (AP@0.5)")
            lines.append("")

            # Header
            header = "| Model |"
            separator = "|-------|"
            for cls in sorted(all_classes):
                header += f" {cls} |"
                separator += "------|"

            lines.append(header)
            lines.append(separator)

            for name, metrics in sorted(self.models.items()):
                row = f"| {name} |"
                for cls in sorted(all_classes):
                    ap = metrics.ap_per_class.get(cls, 0.0)
                    row += f" {ap:.3f} |" if ap > 0 else " - |"
                lines.append(row)

            lines.append("")

        # Rankings
        lines.append("## Rankings")
        lines.append("")

        lines.append("### By Accuracy (mAP@0.5)")
        for i, (name, value) in enumerate(rankings["accuracy_map50"][:10], 1):
            lines.append(f"{i}. {name}: {value:.4f}")
        lines.append("")

        lines.append("### By Speed (FPS)")
        for i, (name, value) in enumerate(rankings["speed_fps"][:10], 1):
            lines.append(f"{i}. {name}: {value:.1f} FPS")
        lines.append("")

        lines.append("### By Efficiency (mAP/Parameters)")
        for i, (name, value) in enumerate(rankings["efficiency"][:10], 1):
            if value > 0:
                lines.append(f"{i}. {name}: {value:.4f}")
        lines.append("")

        # Notes
        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")

        # Conclusion
        lines.append("## Conclusion")
        lines.append("")
        lines.append("Based on the evaluation results:")
        lines.append("")

        if best.get("accuracy"):
            acc_model = self.models.get(best["accuracy"])
            if acc_model:
                lines.append(
                    f"- **{best['accuracy']}** achieves the highest accuracy "
                    f"with mAP@0.5 = {acc_model.map_50:.4f}"
                )

        if best.get("speed"):
            speed_model = self.models.get(best["speed"])
            if speed_model:
                lines.append(
                    f"- **{best['speed']}** is the fastest "
                    f"with {speed_model.fps:.1f} FPS"
                )

        if best.get("balanced"):
            lines.append(
                f"- **{best['balanced']}** offers the best balance "
                f"between accuracy and speed"
            )

        lines.append("")
        lines.append("---")
        lines.append(
            "*This report was generated by the ITS Baseline Evaluation System.*"
        )

        return "\n".join(lines)

    def generate_json(self) -> dict:
        """Generate JSON report."""
        best = self._find_best_models()
        rankings = self._calculate_rankings()

        return {
            "title": self.title,
            "description": self.description,
            "generated_at": self.created_at.isoformat(),
            "dataset": self.dataset_info,
            "models": {
                name: metrics.to_dict() for name, metrics in self.models.items()
            },
            "summary": {
                "best_accuracy": best.get("accuracy"),
                "best_speed": best.get("speed"),
                "best_balanced": best.get("balanced"),
                "best_efficiency": best.get("efficiency"),
                "num_models": len(self.models),
            },
            "rankings": {
                key: [{"model": name, "value": value} for name, value in values]
                for key, values in rankings.items()
            },
            "notes": self.notes,
        }

    def generate_html(self) -> str:
        """Generate HTML report with charts."""
        md_content = self.generate_markdown()
        best = self._find_best_models()

        # Convert Markdown tables to HTML (simplified)
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #444;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .best {{
            background-color: #e8f5e9;
            font-weight: bold;
        }}
        .summary-box {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-item {{
            background: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-item h3 {{
            margin: 0;
            color: #1976d2;
            font-size: 14px;
        }}
        .summary-item .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 10px;
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
        }}
        .bar {{
            height: 30px;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            border-radius: 4px;
            margin: 5px 0;
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            font-weight: bold;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.title}</h1>
        <p><em>Generated: {self.created_at.strftime("%Y-%m-%d %H:%M:%S")}</em></p>

        <div class="summary-box">
            <div class="summary-item">
                <h3>Best Accuracy</h3>
                <div class="value">{best.get("accuracy", "N/A")}</div>
            </div>
            <div class="summary-item">
                <h3>Best Speed</h3>
                <div class="value">{best.get("speed", "N/A")}</div>
            </div>
            <div class="summary-item">
                <h3>Best Balanced</h3>
                <div class="value">{best.get("balanced", "N/A")}</div>
            </div>
            <div class="summary-item">
                <h3>Models Compared</h3>
                <div class="value">{len(self.models)}</div>
            </div>
        </div>

        <h2>Accuracy Comparison (mAP@0.5)</h2>
        <div class="chart-container">
"""

        # Add bar chart for accuracy
        max_map = max([m.map_50 for m in self.models.values()]) or 1.0
        for name, metrics in sorted(
            self.models.items(), key=lambda x: x[1].map_50, reverse=True
        ):
            width = (metrics.map_50 / max_map) * 100 if max_map > 0 else 0
            html_content += f"""
            <div style="margin: 10px 0;">
                <span>{name}</span>
                <div class="bar" style="width: {width}%">{metrics.map_50:.4f}</div>
            </div>
"""

        html_content += """
        </div>

        <h2>Speed Comparison (FPS)</h2>
        <div class="chart-container">
"""

        # Add bar chart for FPS
        max_fps = max([m.fps for m in self.models.values()]) or 1.0
        for name, metrics in sorted(
            self.models.items(), key=lambda x: x[1].fps, reverse=True
        ):
            width = (metrics.fps / max_fps) * 100 if max_fps > 0 else 0
            html_content += f"""
            <div style="margin: 10px 0;">
                <span>{name}</span>
                <div class="bar" style="width: {width}%; background: linear-gradient(90deg, #2196F3, #03A9F4);">{metrics.fps:.1f}</div>
            </div>
"""

        html_content += """
        </div>

        <h2>Detailed Comparison Table</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>mAP@0.5</th>
                <th>mAP@0.75</th>
                <th>FPS</th>
                <th>Latency (ms)</th>
                <th>Params (M)</th>
                <th>Source</th>
            </tr>
"""

        for name, metrics in sorted(
            self.models.items(), key=lambda x: x[1].map_50, reverse=True
        ):
            row_class = "best" if name == best.get("accuracy") else ""
            html_content += f"""
            <tr class="{row_class}">
                <td>{name}</td>
                <td>{metrics.map_50:.4f}</td>
                <td>{f"{metrics.map_75:.4f}" if metrics.map_75 > 0 else "-"}</td>
                <td>{f"{metrics.fps:.1f}" if metrics.fps > 0 else "-"}</td>
                <td>{f"{metrics.latency_ms:.2f}" if metrics.latency_ms > 0 else "-"}</td>
                <td>{f"{metrics.params_millions:.1f}" if metrics.params_millions > 0 else "-"}</td>
                <td>{metrics.source}</td>
            </tr>
"""

        html_content += f"""
        </table>

        <div class="footer">
            <p>Generated by ITS Baseline Evaluation System</p>
            <p>Dataset: {self.dataset_info.get("name", "N/A")} |
               Images: {f"{self.dataset_info.get('num_images'):,}" if isinstance(self.dataset_info.get("num_images"), (int, float)) else "N/A"}</p>
        </div>
    </div>
</body>
</html>
"""

        return html_content

    def generate_report(
        self,
        output_path: str,
        formats: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Generate reports in multiple formats.

        Args:
            output_path: Base path for output files (without extension)
            formats: List of formats to generate ('json', 'markdown', 'html')

        Returns:
            List of generated file paths
        """
        if formats is None:
            formats = ["json", "markdown", "html"]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generated_files = []

        if "json" in formats:
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.generate_json(), f, indent=2, ensure_ascii=False)
            generated_files.append(str(json_path))
            logger.info(f"Generated JSON report: {json_path}")

        if "markdown" in formats:
            md_path = output_path.with_suffix(".md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(self.generate_markdown())
            generated_files.append(str(md_path))
            logger.info(f"Generated Markdown report: {md_path}")

        if "html" in formats:
            html_path = output_path.with_suffix(".html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self.generate_html())
            generated_files.append(str(html_path))
            logger.info(f"Generated HTML report: {html_path}")

        return generated_files


def compare_with_baseline(
    evaluated_results: dict[str, ModelMetrics],
    baseline_results: dict[str, ModelMetrics],
) -> dict[str, dict[str, float]]:
    """
    Compare evaluated results with baseline results.

    Args:
        evaluated_results: Results from evaluation
        baseline_results: Baseline/reference results

    Returns:
        Dictionary with improvement percentages
    """
    comparisons = {}

    for model_name, eval_metrics in evaluated_results.items():
        if model_name not in baseline_results:
            continue

        base_metrics = baseline_results[model_name]

        def safe_improvement(eval_val, base_val):
            if base_val == 0:
                return 0.0
            return ((eval_val - base_val) / base_val) * 100

        comparisons[model_name] = {
            "map_50_improvement": safe_improvement(
                eval_metrics.map_50, base_metrics.map_50
            ),
            "fps_improvement": safe_improvement(eval_metrics.fps, base_metrics.fps),
            "latency_improvement": -safe_improvement(
                eval_metrics.latency_ms, base_metrics.latency_ms
            ),  # Lower is better
        }

    return comparisons
