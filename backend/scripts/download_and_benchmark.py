"""
Download and Benchmark YOLO Models
==================================
Script to download multiple YOLO models and benchmark their performance.
Useful for thesis comparison and evaluation.

Usage:
    python -m scripts.download_and_benchmark --help
    python -m scripts.download_and_benchmark --download-all
    python -m scripts.download_and_benchmark --benchmark-all
    python -m scripts.download_and_benchmark --download yolov8n yolov8s
    python -m scripts.download_and_benchmark --benchmark yolov8n --runs 100
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Models configuration
MODELS_DIR = Path(__file__).parent.parent / "app" / "preTrainedModels"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "benchmarks"

# Available models for download
MODELS_CONFIG = {
    "yolov8n": {
        "name": "YOLOv8 Nano",
        "size_mb": 6.3,
        "params_m": 3.2,
        "expected_map50": 0.370,
        "expected_fps": 195,
    },
    "yolov8s": {
        "name": "YOLOv8 Small",
        "size_mb": 22.5,
        "params_m": 11.2,
        "expected_map50": 0.449,
        "expected_fps": 142,
    },
    "yolov8m": {
        "name": "YOLOv8 Medium",
        "size_mb": 52.0,
        "params_m": 25.9,
        "expected_map50": 0.502,
        "expected_fps": 103,
    },
    "yolov8l": {
        "name": "YOLOv8 Large",
        "size_mb": 87.7,
        "params_m": 43.7,
        "expected_map50": 0.528,
        "expected_fps": 75,
    },
    "yolov8x": {
        "name": "YOLOv8 XLarge",
        "size_mb": 136.7,
        "params_m": 68.2,
        "expected_map50": 0.538,
        "expected_fps": 57,
    },
    "yolov12n": {
        "name": "YOLOv12 Nano",
        "size_mb": 7.0,
        "params_m": 3.5,
        "expected_map50": 0.385,
        "expected_fps": 180,
    },
    "yolov12s": {
        "name": "YOLOv12 Small",
        "size_mb": 24.0,
        "params_m": 12.0,
        "expected_map50": 0.465,
        "expected_fps": 135,
    },
    "yolov12m": {
        "name": "YOLOv12 Medium",
        "size_mb": 56.0,
        "params_m": 28.0,
        "expected_map50": 0.520,
        "expected_fps": 95,
    },
    "yolov12l": {
        "name": "YOLOv12 Large",
        "size_mb": 96.0,
        "params_m": 48.0,
        "expected_map50": 0.545,
        "expected_fps": 68,
    },
    "yolov12x": {
        "name": "YOLOv12 XLarge",
        "size_mb": 150.0,
        "params_m": 75.0,
        "expected_map50": 0.560,
        "expected_fps": 50,
    },
}

# Recommended models for thesis comparison
RECOMMENDED_MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov12x"]


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_status(text: str, status: str = "INFO"):
    """Print a status message."""
    symbols = {"INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERROR": "❌", "DOWNLOAD": "⬇️"}
    print(f"{symbols.get(status, '•')} {text}")


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print_status(f"CUDA available: {device_name} ({memory_gb:.1f} GB)", "OK")
            return True
        else:
            print_status("CUDA not available, using CPU", "WARN")
            return False
    except ImportError:
        print_status("PyTorch not installed", "ERROR")
        return False


def get_model_path(model_key: str) -> Path:
    """Get the path for a model file."""
    return MODELS_DIR / f"{model_key}.pt"


def is_model_downloaded(model_key: str) -> bool:
    """Check if a model is already downloaded."""
    return get_model_path(model_key).exists()


def list_models():
    """List all available models and their status."""
    print_header("Available YOLO Models")

    print(
        f"\n{'Model':<12} {'Name':<20} {'Size':<10} {'Params':<10} {'Downloaded':<12}"
    )
    print("-" * 70)

    for key, config in MODELS_CONFIG.items():
        downloaded = "✅ Yes" if is_model_downloaded(key) else "❌ No"
        recommended = "⭐" if key in RECOMMENDED_MODELS else ""
        print(
            f"{key:<12} {config['name']:<20} {config['size_mb']:<10.1f} "
            f"{config['params_m']:<10.1f} {downloaded:<12} {recommended}"
        )

    print("\n⭐ = Recommended for thesis comparison")


def download_model(model_key: str, force: bool = False) -> bool:
    """Download a single model."""
    if model_key not in MODELS_CONFIG:
        print_status(f"Unknown model: {model_key}", "ERROR")
        return False

    model_path = get_model_path(model_key)

    if model_path.exists() and not force:
        print_status(f"{model_key} already downloaded at {model_path}", "OK")
        return True

    try:
        from ultralytics import YOLO

        config = MODELS_CONFIG[model_key]
        print_status(
            f"Downloading {config['name']} (~{config['size_mb']:.1f} MB)...", "DOWNLOAD"
        )

        # Create models directory if needed
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Download model
        model = YOLO(f"{model_key}.pt")

        # Move to our models directory
        default_path = Path(f"{model_key}.pt")
        if default_path.exists() and default_path != model_path:
            import shutil

            shutil.move(str(default_path), str(model_path))
            print_status(f"Moved to {model_path}", "OK")
        elif not model_path.exists():
            # Save model to correct location
            # model.save(str(model_path))
            print_status(f"Model saved at {default_path}", "OK")

        print_status(f"{config['name']} downloaded successfully!", "OK")
        return True

    except Exception as e:
        print_status(f"Failed to download {model_key}: {e}", "ERROR")
        return False


def download_recommended():
    """Download all recommended models for thesis."""
    print_header("Downloading Recommended Models for Thesis")

    print("\nRecommended models:")
    for key in RECOMMENDED_MODELS:
        config = MODELS_CONFIG[key]
        print(f"  • {key}: {config['name']}")

    print()

    success_count = 0
    for model_key in RECOMMENDED_MODELS:
        if download_model(model_key):
            success_count += 1

    print(f"\n✅ Downloaded {success_count}/{len(RECOMMENDED_MODELS)} models")


def download_all():
    """Download all available models."""
    print_header("Downloading All Models")

    success_count = 0
    for model_key in MODELS_CONFIG.keys():
        if download_model(model_key):
            success_count += 1

    print(f"\n✅ Downloaded {success_count}/{len(MODELS_CONFIG)} models")


def benchmark_model(
    model_key: str,
    image_size: int = 640,
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> Optional[dict]:
    """Benchmark a single model."""
    if model_key not in MODELS_CONFIG:
        print_status(f"Unknown model: {model_key}", "ERROR")
        return None

    model_path = get_model_path(model_key)

    if not model_path.exists():
        print_status(
            f"Model {model_key} not downloaded. Run with --download first.", "ERROR"
        )
        return None

    try:
        import torch
        from ultralytics import YOLO

        config = MODELS_CONFIG[model_key]
        print_status(f"Benchmarking {config['name']}...", "INFO")

        # Load model
        model = YOLO(str(model_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            model.to("cuda")

        # Create dummy input
        dummy_input = np.random.randint(
            0, 255, (image_size, image_size, 3), dtype=np.uint8
        )

        # Warmup
        print(f"  Warming up ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            model(dummy_input, verbose=False)

        # Synchronize CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        print(f"  Running benchmark ({benchmark_runs} runs)...")
        times = []
        for i in range(benchmark_runs):
            start = time.perf_counter()
            model(dummy_input, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i + 1}/{benchmark_runs}")

        times = np.array(times)

        # Get memory usage
        memory_mb = 0.0
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

        result = {
            "model_key": model_key,
            "model_name": config["name"],
            "device": device,
            "image_size": image_size,
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs,
            "avg_time_ms": float(np.mean(times)),
            "std_time_ms": float(np.std(times)),
            "min_time_ms": float(np.min(times)),
            "max_time_ms": float(np.max(times)),
            "fps": 1000.0 / float(np.mean(times)),
            "memory_mb": memory_mb,
            "expected_fps": config["expected_fps"],
            "expected_map50": config["expected_map50"],
            "params_m": config["params_m"],
            "timestamp": datetime.now().isoformat(),
        }

        # Print results
        print(f"\n  Results for {config['name']}:")
        print(f"    • Average inference time: {result['avg_time_ms']:.2f} ms")
        print(f"    • FPS: {result['fps']:.1f} (expected: {config['expected_fps']})")
        print(f"    • Memory used: {memory_mb:.1f} MB")

        return result

    except Exception as e:
        print_status(f"Failed to benchmark {model_key}: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        return None


def benchmark_all_downloaded(
    image_size: int = 640,
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> list[dict]:
    """Benchmark all downloaded models."""
    print_header("Benchmarking All Downloaded Models")

    results = []
    downloaded_models = [
        key for key in MODELS_CONFIG.keys() if is_model_downloaded(key)
    ]

    if not downloaded_models:
        print_status("No models downloaded. Run with --download first.", "ERROR")
        return results

    print(
        f"Found {len(downloaded_models)} downloaded models: {', '.join(downloaded_models)}\n"
    )

    for model_key in downloaded_models:
        result = benchmark_model(model_key, image_size, warmup_runs, benchmark_runs)
        if result:
            results.append(result)
        print()

    return results


def save_results(results: list[dict], filename: Optional[str] = None):
    """Save benchmark results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    filepath = RESULTS_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_status(f"Results saved to {filepath}", "OK")
    return filepath


def generate_comparison_table(results: list[dict]) -> str:
    """Generate a markdown comparison table from results."""
    if not results:
        return "No results to display."

    lines = [
        "# Model Comparison Results\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Device: {results[0].get('device', 'unknown')}\n",
        "",
        "| Model | Params (M) | FPS | Expected FPS | Inference (ms) | Memory (MB) |",
        "|-------|-----------|-----|--------------|----------------|-------------|",
    ]

    for r in sorted(results, key=lambda x: x.get("fps", 0), reverse=True):
        lines.append(
            f"| {r['model_name']} | {r['params_m']:.1f} | {r['fps']:.1f} | "
            f"{r['expected_fps']} | {r['avg_time_ms']:.2f} | {r['memory_mb']:.1f} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("- FPS measured with dummy input (640x640)")
    lines.append("- Expected FPS from official Ultralytics benchmarks")
    lines.append("- Memory usage is GPU memory if CUDA available")

    return "\n".join(lines)


def save_comparison_table(results: list[dict], filename: Optional[str] = None):
    """Save comparison table as markdown."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    filepath = RESULTS_DIR / filename

    table = generate_comparison_table(results)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(table)

    print_status(f"Comparison table saved to {filepath}", "OK")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Download and benchmark YOLO models for ITS project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.download_and_benchmark --list
  python -m scripts.download_and_benchmark --download-recommended
  python -m scripts.download_and_benchmark --download yolov8n yolov8s
  python -m scripts.download_and_benchmark --benchmark yolov8n --runs 100
  python -m scripts.download_and_benchmark --benchmark-all --save
        """,
    )

    parser.add_argument("--list", action="store_true", help="List all available models")
    parser.add_argument(
        "--download", nargs="*", metavar="MODEL", help="Download specific models"
    )
    parser.add_argument(
        "--download-recommended",
        action="store_true",
        help="Download recommended models for thesis",
    )
    parser.add_argument(
        "--download-all", action="store_true", help="Download all models"
    )
    parser.add_argument(
        "--benchmark", nargs="*", metavar="MODEL", help="Benchmark specific models"
    )
    parser.add_argument(
        "--benchmark-all", action="store_true", help="Benchmark all downloaded models"
    )
    parser.add_argument(
        "--runs", type=int, default=50, help="Number of benchmark runs (default: 50)"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup runs (default: 10)"
    )
    parser.add_argument(
        "--image-size", type=int, default=640, help="Input image size (default: 640)"
    )
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument("--force", action="store_true", help="Force re-download models")

    args = parser.parse_args()

    # Print welcome
    print_header("ITS - YOLO Model Download & Benchmark Tool")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    check_cuda()

    # Handle commands
    if args.list or len(sys.argv) == 1:
        list_models()
        if len(sys.argv) == 1:
            print("\nRun with --help for usage information")
        return

    if args.download_recommended:
        download_recommended()

    if args.download_all:
        download_all()

    if args.download is not None:
        if len(args.download) == 0:
            # --download without arguments means download recommended
            download_recommended()
        else:
            for model in args.download:
                download_model(model, force=args.force)

    results = []

    if args.benchmark_all:
        results = benchmark_all_downloaded(
            image_size=args.image_size,
            warmup_runs=args.warmup,
            benchmark_runs=args.runs,
        )

    if args.benchmark is not None:
        if len(args.benchmark) == 0:
            # --benchmark without arguments means benchmark all
            results = benchmark_all_downloaded(
                image_size=args.image_size,
                warmup_runs=args.warmup,
                benchmark_runs=args.runs,
            )
        else:
            for model in args.benchmark:
                result = benchmark_model(
                    model,
                    image_size=args.image_size,
                    warmup_runs=args.warmup,
                    benchmark_runs=args.runs,
                )
                if result:
                    results.append(result)

    # Save results if requested
    if args.save and results:
        save_results(results)
        save_comparison_table(results)

    # Print final summary
    if results:
        print_header("Benchmark Summary")
        print(generate_comparison_table(results))


if __name__ == "__main__":
    main()
