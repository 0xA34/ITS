"""
Validation Runner Script
========================
Main script to run complete validation pipeline for baseline evaluation results.

Usage:
    # Run full validation for a single model
    python -m app.baseline.validation.run_validation --model yolov8n --results results/baseline

    # Run validation with cross-validation (multiple runs)
    python -m app.baseline.validation.run_validation --model yolov8n --cross-validate --runs 5

    # Run validation for all models in results directory
    python -m app.baseline.validation.run_validation --all --results results/baseline

    # Quick validation (skip cross-validation)
    python -m app.baseline.validation.run_validation --model yolov8n --quick
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_baseline_results(results_dir: str, model_name: str) -> Optional[dict]:
    """Load baseline results for a model."""
    results_path = Path(results_dir)

    # Try different file naming patterns
    possible_files = [
        results_path / f"results_{model_name}.json",
        results_path / f"baseline_report_{model_name}.json",
        results_path / f"{model_name}_results.json",
        results_path / f"{model_name}.json",
    ]

    for file_path in possible_files:
        if file_path.exists():
            logger.info(f"Loading baseline results from {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)

    logger.warning(
        f"No baseline results found for model '{model_name}' in {results_dir}"
    )
    return None


def find_all_models(results_dir: str) -> list[str]:
    """Find all models with baseline results."""
    results_path = Path(results_dir)
    models = set()

    if not results_path.exists():
        return []

    for file in results_path.glob("*.json"):
        name = file.stem
        # Extract model name from various patterns
        for prefix in ["results_", "baseline_report_"]:
            if name.startswith(prefix):
                models.add(name[len(prefix) :])
                break
        else:
            # If no known prefix, use the filename as model name
            if "_results" in name:
                models.add(name.replace("_results", ""))

    return sorted(list(models))


def run_cross_validation(
    model_name: str,
    model_path: str,
    dataset_path: str,
    num_runs: int = 5,
    sample_ratio: float = 0.8,
    max_images: Optional[int] = None,
    verbose: bool = True,
) -> Optional[Any]:
    """Run cross-validation for a model."""
    try:
        from app.baseline.validation.cross_validator import CrossValidator

        # Determine actual model path
        model_file = Path(model_path)
        if not model_file.exists():
            model_file = Path(f"{model_name}.pt")
        if not model_file.exists():
            model_file = Path("models") / f"{model_name}.pt"
        if not model_file.exists():
            logger.warning(
                f"Model file not found for {model_name}, skipping cross-validation"
            )
            return None

        validator = CrossValidator(
            model_path=str(model_file),
            dataset_path=dataset_path,
            max_images=max_images,
        )

        results = validator.run(
            num_runs=num_runs,
            sample_ratio=sample_ratio,
            verbose=verbose,
        )

        return results

    except ImportError as e:
        logger.error(f"Failed to import cross_validator: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during cross-validation: {e}")
        return None


def run_statistical_analysis(
    cross_validation_results: Any = None,
    baseline_results: dict = None,
) -> Optional[Any]:
    """Run statistical analysis on results."""
    try:
        from app.baseline.validation.statistical_analyzer import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer()

        if cross_validation_results is not None:
            analyzer.set_results(cross_validation_results)
            return analyzer.generate_summary()
        elif baseline_results is not None:
            # Create a pseudo cross-validation result from single baseline
            # This provides limited statistical info but still useful
            logger.info(
                "Single baseline result provided - statistical analysis will be limited"
            )
            return None

        return None

    except ImportError as e:
        logger.error(f"Failed to import statistical_analyzer: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during statistical analysis: {e}")
        return None


def run_benchmark_comparison(
    model_name: str,
    baseline_results: dict,
) -> Optional[Any]:
    """Compare results with previous benchmarks from history."""
    try:
        from app.baseline.validation.benchmark_comparator import BenchmarkComparator

        comparator = BenchmarkComparator()
        comparison = comparator.compare_with_history(baseline_results, model_name)

        if comparison.total_metrics > 0:
            comparator.print_comparison(comparison)

        return comparison

    except ImportError as e:
        logger.error(f"Failed to import benchmark_comparator: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during benchmark comparison: {e}")
        return None


def generate_confusion_matrix(
    model_name: str,
    model_path: str,
    dataset_path: str,
    max_images: int = 500,
    output_dir: str = "results/validation",
    verbose: bool = True,
) -> Optional[Any]:
    """
    Generate confusion matrix for a model by running inference on dataset.

    Args:
        model_name: Name of the model
        model_path: Path to model file
        dataset_path: Path to dataset
        max_images: Maximum images to evaluate
        output_dir: Directory to save confusion matrix
        verbose: Print progress

    Returns:
        ConfusionMatrixData or None if failed
    """
    try:
        from pathlib import Path

        from app.baseline.validation.confusion_matrix import ConfusionMatrixGenerator

        # Vehicle class mapping
        VEHICLE_CLASSES = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

        # Try to find model file
        model_file = Path(model_path)
        if not model_file.exists():
            clean_name = model_name.replace(".pt", "")
            model_file = Path(f"{clean_name}.pt")
        if not model_file.exists():
            model_file = Path("models") / f"{clean_name}.pt"
        if not model_file.exists():
            logger.warning(f"Model file not found: {model_path}")
            return None

        # Check if dataset exists
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            return None

        if verbose:
            logger.info(f"Generating confusion matrix for {model_name}")

        # Load model
        try:
            from ultralytics import YOLO

            model = YOLO(str(model_file))
        except ImportError:
            logger.error("ultralytics package not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

        # Initialize confusion matrix generator
        cm_generator = ConfusionMatrixGenerator(
            class_names=VEHICLE_CLASSES,
            iou_threshold=0.5,
            confidence_threshold=0.25,
        )

        # Load dataset
        try:
            from app.baseline.datasets import CustomDataset
            from pathlib import Path

            # Check for standard COCO structure
            ds_path = Path(dataset_path)
            annotation_file = None
            images_dir = None
            
            coco_ann = ds_path / "annotations" / "instances_val2017.json"
            if coco_ann.exists():
                annotation_file = str(coco_ann)
                images_dir = str(ds_path / "val2017")
                logger.info(f"Detected standard COCO structure at {dataset_path}")

            dataset = CustomDataset(
                data_path=dataset_path,
                split="val",
                annotation_file=annotation_file,
                images_dir=images_dir,
            )
            annotations = dataset.annotations[:max_images]
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None

        # Run inference and collect predictions
        import cv2

        for i, ann in enumerate(annotations):
            if verbose and (i + 1) % 50 == 0:
                logger.info(f"  Processing image {i + 1}/{len(annotations)}")

            image_path = ann.image_path
            if not image_path or not Path(image_path).exists():
                continue

            # Get ground truth boxes
            gt_boxes = []
            gt_classes = []
            for obj in ann.annotations:
                bbox = obj.bbox
                # Check if bbox is a tuple/list or BoundingBox object
                if hasattr(bbox, "to_xyxy"):
                     # Convert to [x1, y1, x2, y2]
                     x1, y1, x2, y2 = bbox.to_xyxy()
                     box_list = [x1, y1, x2, y2]
                elif hasattr(bbox, "x1"):
                     box_list = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
                else:
                     box_list = list(bbox)

                class_id = obj.class_id
                if len(box_list) == 4 and class_id in VEHICLE_CLASSES:
                    gt_boxes.append(box_list)
                    gt_classes.append(class_id)

            if not gt_boxes:
                continue

            # Run inference
            try:
                results = model.predict(
                    image_path,
                    conf=0.25,
                    iou=0.45,
                    verbose=False,
                )
                result = results[0]

                # Get predictions
                pred_boxes = []
                pred_classes = []
                pred_confidences = []

                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        if class_id in VEHICLE_CLASSES:
                            pred_boxes.append(box.xyxy[0].cpu().numpy().tolist())
                            pred_classes.append(class_id)
                            pred_confidences.append(float(box.conf.item()))

                # Add to confusion matrix
                cm_generator.add_batch(
                    predictions=[
                        {"bbox": b, "class_id": c, "confidence": conf}
                        for b, c, conf in zip(pred_boxes, pred_classes, pred_confidences)
                    ],
                    ground_truths=[
                        {"bbox": b, "class_id": c}
                        for b, c in zip(gt_boxes, gt_classes)
                    ],
                )

            except Exception as e:
                logger.debug(f"Error processing image {image_path}: {e}")
                continue

        # Save confusion matrix
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cm_generator.save(
            str(output_path / f"confusion_matrix_{model_name}"),
            formats=["json", "txt", "html"],
        )

        if verbose:
            cm_generator.print_matrix()

        return cm_generator.get_matrix_data()

    except ImportError as e:
        logger.error(f"Failed to import confusion_matrix: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during confusion matrix generation: {e}")
        return None


def run_full_validation(
    model_name: str,
    results_dir: str = "results/baseline",
    dataset_path: str = "data/coco",
    output_dir: str = "results/validation",
    cross_validate: bool = False,
    num_runs: int = 5,
    generate_cm: bool = False,
    max_images_cm: int = 500,
    quick: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run full validation pipeline for a model.

    Args:
        model_name: Name of the model to validate
        results_dir: Directory containing baseline results
        dataset_path: Path to dataset (for cross-validation)
        output_dir: Directory to save validation results
        cross_validate: Whether to run cross-validation
        num_runs: Number of cross-validation runs
        generate_cm: Whether to generate confusion matrix
        max_images_cm: Max images for confusion matrix
        quick: Quick mode - skip time-consuming operations
        verbose: Print detailed output

    Returns:
        Dictionary with validation results
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Validation Pipeline: {model_name}")
        print(f"{'=' * 60}\n")

    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "baseline_results": None,
        "cross_validation": None,
        "statistical_analysis": None,
        "benchmark_comparison": None,
        "confusion_matrix": None,
        "overall_status": "UNKNOWN",
    }

    # Step 1: Load baseline results
    if verbose:
        print("[1/6] Loading baseline results...")

    baseline_results = load_baseline_results(results_dir, model_name)
    if baseline_results is None:
        logger.error(f"Cannot proceed without baseline results for {model_name}")
        results["overall_status"] = "FAIL"
        return results

    results["baseline_results"] = baseline_results
    if verbose:
        print(f"      ✓ Loaded baseline results")

    # Step 2: Run cross-validation (if enabled)
    cv_results = None
    if cross_validate and not quick:
        if verbose:
            print(f"[2/6] Running cross-validation ({num_runs} runs)...")

        model_path = baseline_results.get("model", {}).get("path", f"{model_name}.pt")
        cv_results = run_cross_validation(
            model_name=model_name,
            model_path=model_path,
            dataset_path=dataset_path,
            num_runs=num_runs,
            verbose=verbose,
        )

        if cv_results is not None:
            results["cross_validation"] = (
                cv_results.to_dict() if hasattr(cv_results, "to_dict") else cv_results
            )
            if verbose:
                print(f"      ✓ Cross-validation completed")
        else:
            if verbose:
                print(f"      ⚠ Cross-validation skipped or failed")
    else:
        if verbose:
            print("[2/6] Skipping cross-validation (use --cross-validate to enable)")

    # Step 3: Statistical analysis
    if verbose:
        print("[3/6] Running statistical analysis...")

    stat_summary = run_statistical_analysis(
        cross_validation_results=cv_results,
        baseline_results=baseline_results,
    )

    if stat_summary is not None:
        results["statistical_analysis"] = (
            stat_summary.to_dict() if hasattr(stat_summary, "to_dict") else stat_summary
        )
        if verbose:
            print(f"      ✓ Statistical analysis completed")
    else:
        if verbose:
            print(f"      ⚠ Statistical analysis skipped (requires cross-validation)")

    # Step 4: Benchmark comparison
    if verbose:
        print("[4/6] Comparing with previous benchmarks...")

    benchmark_comparison = run_benchmark_comparison(model_name, baseline_results)

    if benchmark_comparison is not None:
        results["benchmark_comparison"] = (
            benchmark_comparison.to_dict()
            if hasattr(benchmark_comparison, "to_dict")
            else benchmark_comparison
        )
        if verbose:
            print(f"      ✓ Benchmark comparison completed")
    else:
        if verbose:
            print(f"      ⚠ Benchmark comparison skipped")

    # Step 5: Generate confusion matrix (if enabled)
    if generate_cm and not quick:
        if verbose:
            print(f"[5/6] Generating confusion matrix...")

        model_path = baseline_results.get("model", {}).get("path", f"{model_name}.pt")
        cm_data = generate_confusion_matrix(
            model_name=model_name,
            model_path=model_path,
            dataset_path=dataset_path,
            max_images=max_images_cm,
            output_dir=output_dir,
            verbose=verbose,
        )

        if cm_data is not None:
            results["confusion_matrix"] = (
                cm_data.to_dict() if hasattr(cm_data, "to_dict") else cm_data
            )
            if verbose:
                print(f"      ✓ Confusion matrix generated")
        else:
            if verbose:
                print(f"      ⚠ Confusion matrix generation failed")
    else:
        if verbose:
            print("[5/6] Skipping confusion matrix (use --confusion-matrix to enable)")

    # Step 6: Generate validation report
    if verbose:
        print("[6/6] Generating validation report...")

    try:
        from app.baseline.validation.validation_report import ValidationReport

        report = ValidationReport(
            model_name=model_name,
            cross_validation_results=results.get("cross_validation"),
            statistical_summary=results.get("statistical_analysis"),
            benchmark_comparison=results.get("benchmark_comparison"),
            confusion_matrix_data=results.get("confusion_matrix"),
        )

        # Save reports
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_base = output_path / f"validation_{model_name}"
        report.save(str(report_base), formats=["json", "md", "html"])

        compiled = report.compile_report()
        results["overall_status"] = compiled.overall_status
        results["validation_score"] = compiled.validation_score
        results["recommendations"] = compiled.recommendations

        if verbose:
            print(f"      ✓ Reports saved to {output_path}")
            report.print_summary()

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        if verbose:
            print(f"      ✗ Failed to generate report: {e}")

    return results


def main():
    """Main entry point for validation runner."""
    parser = argparse.ArgumentParser(
        description="Run validation pipeline for baseline evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a single model
  python -m app.baseline.validation.run_validation --model yolov8n

  # Validate multiple specific models (batch mode)
  python -m app.baseline.validation.run_validation --models yolov8n yolov8s yolov8m

  # Validate with cross-validation
  python -m app.baseline.validation.run_validation --model yolov8n --cross-validate --runs 5

  # Validate all models in results directory
  python -m app.baseline.validation.run_validation --all

  # Quick validation (skip cross-validation)
  python -m app.baseline.validation.run_validation --model yolov8n --quick
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Name of a single model to validate (e.g., yolov8n)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        help="Multiple model names for batch validation (e.g., --models yolov8n yolov8s yolov8m)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all models found in results directory",
    )

    parser.add_argument(
        "--results",
        type=str,
        default="results/baseline",
        help="Directory containing baseline results (default: results/baseline)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/coco",
        help="Path to dataset for cross-validation (default: data/coco)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/validation",
        help="Output directory for validation reports (default: results/validation)",
    )

    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run cross-validation (multiple evaluation runs)",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of cross-validation runs (default: 5)",
    )

    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Generate confusion matrix (requires model and dataset)",
    )

    parser.add_argument(
        "--cm-images",
        type=int,
        default=500,
        help="Max images for confusion matrix generation (default: 500)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode - skip cross-validation and confusion matrix",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all models with available baseline results",
    )

    args = parser.parse_args()

    if args.list_models:
        models = find_all_models(args.results)
        print("\nModels with baseline results:")
        print("-" * 40)
        for model in models:
            print(f"  - {model}")
        print()
        return 0

    # Validate arguments
    if not args.model and not args.models and not args.all:
        parser.error("One of --model, --models, or --all is required")

    verbose = not args.quiet

    # Determine which models to validate
    if args.all:
        models = find_all_models(args.results)
        if not models:
            print(f"No models found in {args.results}")
            return 1
    elif args.models:
        models = args.models
    else:
        models = [args.model]

    # Run validation
    if len(models) > 1:
        # Batch mode
        if verbose:
            print(f"\n{'='*60}")
            print(f"BATCH VALIDATION: {len(models)} models")
            print(f"{'='*60}")
            print(f"Models: {', '.join(models)}\n")

        all_results = {}
        for i, model in enumerate(models, 1):
            if verbose:
                print(f"\n[{i}/{len(models)}] Validating: {model}")
                print("-" * 40)

            result = run_full_validation(
                model_name=model,
                results_dir=args.results,
                dataset_path=args.dataset,
                output_dir=args.output,
                cross_validate=args.cross_validate,
                num_runs=args.runs,
                generate_cm=args.confusion_matrix,
                max_images_cm=args.cm_images,
                quick=args.quick,
                verbose=verbose,
            )
            all_results[model] = result

        # Print summary
        if verbose:
            print(f"\n{'='*60}")
            print("BATCH VALIDATION SUMMARY")
            print(f"{'='*60}")
            print(f"{'Model':<20} {'Status':<15} {'Score':<10}")
            print("-" * 60)
            for model, result in all_results.items():
                status = result.get("overall_status", "UNKNOWN")
                score = result.get("validation_score", 0)
                print(f"{model:<20} {status:<15} {score:.1f}")
            print(f"{'='*60}")

    else:
        # Single model mode
        result = run_full_validation(
            model_name=models[0],
            results_dir=args.results,
            dataset_path=args.dataset,
            output_dir=args.output,
            cross_validate=args.cross_validate,
            num_runs=args.runs,
            generate_cm=args.confusion_matrix,
            max_images_cm=args.cm_images,
            quick=args.quick,
            verbose=verbose,
        )

        if result.get("overall_status") == "FAIL":
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
