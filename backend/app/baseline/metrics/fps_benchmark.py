"""
FPS (Frames Per Second) Benchmark Module
=========================================
Measure inference speed and latency of detection models.

Metrics:
    - FPS: Frames processed per second
    - Latency: Time to process a single frame (ms)
    - Throughput: Total frames processed over time

Usage:
    from app.baseline.metrics.fps_benchmark import FPSBenchmark

    benchmark = FPSBenchmark(model_path="yolov8n.pt")
    result = benchmark.run(num_iterations=100, warmup=10)
    print(result)
"""

import gc
import logging
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of FPS benchmark."""

    model_name: str
    fps: float
    latency_ms: float
    latency_std_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput: float  # frames per total time including overhead
    num_iterations: int
    warmup_iterations: int
    image_size: tuple[int, int]
    device: str
    batch_size: int = 1

    # System info
    cpu_info: str = ""
    gpu_info: str = ""
    platform_info: str = ""

    # Additional timing breakdown
    preprocess_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "fps": self.fps,
            "latency_ms": self.latency_ms,
            "latency_std_ms": self.latency_std_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "throughput": self.throughput,
            "num_iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
            "image_size": self.image_size,
            "device": self.device,
            "batch_size": self.batch_size,
            "cpu_info": self.cpu_info,
            "gpu_info": self.gpu_info,
            "platform_info": self.platform_info,
            "preprocess_time_ms": self.preprocess_time_ms,
            "inference_time_ms": self.inference_time_ms,
            "postprocess_time_ms": self.postprocess_time_ms,
        }

    def __str__(self) -> str:
        return f"""
{"=" * 60}
FPS Benchmark Results: {self.model_name}
{"=" * 60}
Performance:
  FPS:              {self.fps:.2f}
  Latency:          {self.latency_ms:.2f} ± {self.latency_std_ms:.2f} ms
  Min/Max Latency:  {self.min_latency_ms:.2f} / {self.max_latency_ms:.2f} ms
  Throughput:       {self.throughput:.2f} frames/sec

Timing Breakdown:
  Preprocess:       {self.preprocess_time_ms:.2f} ms
  Inference:        {self.inference_time_ms:.2f} ms
  Postprocess:      {self.postprocess_time_ms:.2f} ms

Configuration:
  Image Size:       {self.image_size[0]}x{self.image_size[1]}
  Batch Size:       {self.batch_size}
  Device:           {self.device}
  Iterations:       {self.num_iterations} (warmup: {self.warmup_iterations})

System:
  Platform:         {self.platform_info}
  CPU:              {self.cpu_info}
  GPU:              {self.gpu_info}
{"=" * 60}
"""


@dataclass
class LatencyMeasurement:
    """Single latency measurement with breakdown."""

    total_ms: float
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0


class FPSBenchmark:
    """
    Benchmark class for measuring model inference speed.

    Supports:
        - YOLO models (ultralytics)
        - Custom inference functions
        - Multiple image sizes
        - GPU and CPU benchmarking

    Usage:
        # With YOLO model
        benchmark = FPSBenchmark(model_path="yolov8n.pt")
        result = benchmark.run()

        # With custom function
        def my_inference(image):
            # Your inference code
            return detections

        benchmark = FPSBenchmark(inference_fn=my_inference)
        result = benchmark.run()
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        inference_fn: Optional[Callable] = None,
        image_size: tuple[int, int] = (640, 640),
        device: str = "auto",
        batch_size: int = 1,
        model_name: Optional[str] = None,
    ):
        """
        Initialize FPS benchmark.

        Args:
            model_path: Path to YOLO model file
            inference_fn: Custom inference function (alternative to model_path)
            image_size: Input image size (width, height)
            device: Device for inference ('cuda', 'cpu', 'auto')
            batch_size: Batch size for inference
            model_name: Name for the model (for reporting)
        """
        self.model_path = model_path
        self.inference_fn = inference_fn
        self.image_size = image_size
        self.batch_size = batch_size
        self.model_name = model_name

        # Determine device
        if device == "auto":
            self.device = self._detect_device()
        else:
            self.device = device

        # Model will be loaded lazily
        self._model = None

        # System info
        self._cpu_info = self._get_cpu_info()
        self._gpu_info = self._get_gpu_info()
        self._platform_info = self._get_platform_info()

    def _detect_device(self) -> str:
        """Detect available device (CUDA or CPU)."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _get_cpu_info(self) -> str:
        """Get CPU information."""
        try:
            if platform.system() == "Windows":
                return platform.processor()
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            elif platform.system() == "Darwin":
                import subprocess

                return (
                    subprocess.check_output(
                        ["sysctl", "-n", "machdep.cpu.brand_string"]
                    )
                    .decode()
                    .strip()
                )
        except Exception:
            pass
        return platform.processor() or "Unknown CPU"

    def _get_gpu_info(self) -> str:
        """Get GPU information."""
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                return f"{gpu_name} ({gpu_memory:.1f} GB)"
        except Exception:
            pass
        return "N/A (CPU only)"

    def _get_platform_info(self) -> str:
        """Get platform information."""
        return f"{platform.system()} {platform.release()} ({platform.machine()})"

    def _load_model(self):
        """Load YOLO model if model_path is specified."""
        if self._model is not None:
            return self._model

        if self.model_path is None:
            if self.inference_fn is None:
                raise ValueError("Either model_path or inference_fn must be provided")
            return None

        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)

            # Move to device
            if self.device == "cuda":
                import torch

                if torch.cuda.is_available():
                    self._model.to("cuda")

            # Set model name if not provided
            if self.model_name is None:
                self.model_name = Path(self.model_path).stem

            return self._model
        except ImportError:
            raise ImportError(
                "ultralytics package is required for YOLO models. "
                "Install with: pip install ultralytics"
            )

    def _generate_dummy_image(self) -> np.ndarray:
        """Generate a random dummy image for benchmarking."""
        return np.random.randint(
            0, 255, (self.image_size[1], self.image_size[0], 3), dtype=np.uint8
        )

    def _run_inference(self, image: np.ndarray) -> LatencyMeasurement:
        """
        Run single inference and measure time.

        Returns:
            LatencyMeasurement with timing breakdown
        """
        if self.inference_fn is not None:
            # Custom inference function
            start = time.perf_counter()
            _ = self.inference_fn(image)
            end = time.perf_counter()
            return LatencyMeasurement(total_ms=(end - start) * 1000)

        # YOLO inference with timing breakdown
        model = self._load_model()

        # Preprocess timing (YOLO handles this internally, estimate)
        start_total = time.perf_counter()

        # Run inference
        results = model.predict(
            source=image,
            verbose=False,
            conf=0.25,
            iou=0.45,
            device=self.device if self.device != "auto" else None,
        )

        end_total = time.perf_counter()

        total_ms = (end_total - start_total) * 1000

        # Try to get timing breakdown from YOLO results
        preprocess_ms = 0.0
        inference_ms = 0.0
        postprocess_ms = 0.0

        if results and len(results) > 0:
            speed = results[0].speed
            if speed:
                preprocess_ms = speed.get("preprocess", 0.0)
                inference_ms = speed.get("inference", 0.0)
                postprocess_ms = speed.get("postprocess", 0.0)

        return LatencyMeasurement(
            total_ms=total_ms,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
        )

    def run(
        self,
        num_iterations: int = 100,
        warmup: int = 10,
        image: Optional[np.ndarray] = None,
        images: Optional[list[np.ndarray]] = None,
    ) -> BenchmarkResult:
        """
        Run FPS benchmark.

        Args:
            num_iterations: Number of benchmark iterations
            warmup: Number of warmup iterations (not counted)
            image: Optional specific image to use
            images: Optional list of images to cycle through

        Returns:
            BenchmarkResult with detailed metrics
        """
        # Load model if needed
        if self.model_path:
            self._load_model()

        # Prepare test images
        if images is not None:
            test_images = images
        elif image is not None:
            test_images = [image]
        else:
            # Generate dummy images
            test_images = [self._generate_dummy_image() for _ in range(10)]

        # Warmup
        logger.info(f"Running {warmup} warmup iterations...")
        for i in range(warmup):
            img = test_images[i % len(test_images)]
            _ = self._run_inference(img)

        # Force garbage collection
        gc.collect()
        if self.device == "cuda":
            try:
                import torch

                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Benchmark
        logger.info(f"Running {num_iterations} benchmark iterations...")
        measurements: list[LatencyMeasurement] = []
        total_start = time.perf_counter()

        for i in range(num_iterations):
            img = test_images[i % len(test_images)]
            measurement = self._run_inference(img)
            measurements.append(measurement)

            # Sync CUDA if needed
            if self.device == "cuda":
                try:
                    import torch

                    torch.cuda.synchronize()
                except Exception:
                    pass

        total_end = time.perf_counter()
        total_time = total_end - total_start

        # Calculate statistics
        latencies = np.array([m.total_ms for m in measurements])
        preprocess_times = np.array([m.preprocess_ms for m in measurements])
        inference_times = np.array([m.inference_ms for m in measurements])
        postprocess_times = np.array([m.postprocess_ms for m in measurements])

        mean_latency = float(np.mean(latencies))
        std_latency = float(np.std(latencies))
        min_latency = float(np.min(latencies))
        max_latency = float(np.max(latencies))

        # FPS = 1000 / latency_ms
        fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

        # Throughput = iterations / total_time
        throughput = num_iterations / total_time if total_time > 0 else 0.0

        # Model name
        model_name = self.model_name or "Unknown Model"

        return BenchmarkResult(
            model_name=model_name,
            fps=fps,
            latency_ms=mean_latency,
            latency_std_ms=std_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            throughput=throughput,
            num_iterations=num_iterations,
            warmup_iterations=warmup,
            image_size=self.image_size,
            device=self.device,
            batch_size=self.batch_size,
            cpu_info=self._cpu_info,
            gpu_info=self._gpu_info,
            platform_info=self._platform_info,
            preprocess_time_ms=float(np.mean(preprocess_times)),
            inference_time_ms=float(np.mean(inference_times)),
            postprocess_time_ms=float(np.mean(postprocess_times)),
        )

    def run_multiple_sizes(
        self,
        sizes: list[tuple[int, int]],
        num_iterations: int = 50,
        warmup: int = 5,
    ) -> dict[str, BenchmarkResult]:
        """
        Run benchmark at multiple image sizes.

        Args:
            sizes: List of (width, height) tuples
            num_iterations: Iterations per size
            warmup: Warmup iterations per size

        Returns:
            Dictionary mapping size string to BenchmarkResult
        """
        results = {}

        for width, height in sizes:
            logger.info(f"Benchmarking at {width}x{height}...")
            self.image_size = (width, height)
            result = self.run(num_iterations=num_iterations, warmup=warmup)
            results[f"{width}x{height}"] = result

        return results


def benchmark_model(
    model_path: str,
    num_iterations: int = 100,
    warmup: int = 10,
    image_size: tuple[int, int] = (640, 640),
    device: str = "auto",
) -> BenchmarkResult:
    """
    Convenience function to benchmark a model.

    Args:
        model_path: Path to model file
        num_iterations: Number of benchmark iterations
        warmup: Number of warmup iterations
        image_size: Input image size
        device: Device for inference

    Returns:
        BenchmarkResult
    """
    benchmark = FPSBenchmark(
        model_path=model_path,
        image_size=image_size,
        device=device,
    )
    return benchmark.run(num_iterations=num_iterations, warmup=warmup)


def compare_models(
    model_paths: list[str],
    num_iterations: int = 100,
    warmup: int = 10,
    image_size: tuple[int, int] = (640, 640),
    device: str = "auto",
) -> dict[str, BenchmarkResult]:
    """
    Benchmark and compare multiple models.

    Args:
        model_paths: List of model file paths
        num_iterations: Number of benchmark iterations per model
        warmup: Number of warmup iterations per model
        image_size: Input image size
        device: Device for inference

    Returns:
        Dictionary mapping model name to BenchmarkResult
    """
    results = {}

    for model_path in model_paths:
        model_name = Path(model_path).stem
        logger.info(f"Benchmarking {model_name}...")

        benchmark = FPSBenchmark(
            model_path=model_path,
            image_size=image_size,
            device=device,
            model_name=model_name,
        )
        results[model_name] = benchmark.run(
            num_iterations=num_iterations, warmup=warmup
        )

    return results


def print_comparison_table(results: dict[str, BenchmarkResult]):
    """Print a comparison table of benchmark results."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - FPS BENCHMARK")
    print("=" * 80)
    print(
        f"{'Model':<20} {'FPS':>10} {'Latency (ms)':>15} {'Min/Max (ms)':>20} {'Device':>10}"
    )
    print("-" * 80)

    for model_name, result in sorted(results.items(), key=lambda x: -x[1].fps):
        latency_str = f"{result.latency_ms:.2f} ± {result.latency_std_ms:.2f}"
        minmax_str = f"{result.min_latency_ms:.2f} / {result.max_latency_ms:.2f}"
        print(
            f"{model_name:<20} {result.fps:>10.2f} {latency_str:>15} {minmax_str:>20} {result.device:>10}"
        )

    print("=" * 80)
