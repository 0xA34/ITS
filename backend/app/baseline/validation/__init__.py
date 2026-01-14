"""
Validation Module for Baseline Evaluation
==========================================

This module provides validation tools to ensure baseline evaluation results
are statistically reliable and comparable to published benchmarks.

Components:
    - CrossValidator: Run multiple evaluation rounds with different seeds
    - StatisticalAnalyzer: Calculate confidence intervals, significance tests
    - BenchmarkComparator: Compare results with official published benchmarks
    - ConfusionMatrixGenerator: Generate and visualize confusion matrices
    - ValidationReport: Generate comprehensive validation reports

Usage:
    from app.baseline.validation import (
        CrossValidator,
        StatisticalAnalyzer,
        BenchmarkComparator,
        ConfusionMatrixGenerator,
        ValidationReport,
    )

    # Run cross-validation
    validator = CrossValidator(model_path="yolov8n.pt", dataset_path="data/coco")
    results = validator.run(num_runs=5)

    # Analyze statistics
    analyzer = StatisticalAnalyzer(results)
    stats = analyzer.compute_confidence_intervals()

    # Compare with benchmarks
    comparator = BenchmarkComparator()
    comparison = comparator.compare_with_official(results, model_name="yolov8n")

    # Generate report
    report = ValidationReport(results, stats, comparison)
    report.generate("results/validation")
"""

from app.baseline.validation.benchmark_comparator import (
    BenchmarkComparator,
    OfficialBenchmarks,
)
from app.baseline.validation.confusion_matrix import ConfusionMatrixGenerator
from app.baseline.validation.cross_validator import CrossValidator
from app.baseline.validation.statistical_analyzer import StatisticalAnalyzer
from app.baseline.validation.validation_report import ValidationReport

__all__ = [
    "CrossValidator",
    "StatisticalAnalyzer",
    "BenchmarkComparator",
    "OfficialBenchmarks",
    "ConfusionMatrixGenerator",
    "ValidationReport",
]
