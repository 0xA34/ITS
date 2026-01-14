"""
Model Manager Service
====================
Service for managing multiple YOLO models for vehicle detection.
Supports loading, switching, and benchmarking different models.
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Model directory
MODELS_DIR = Path(__file__).parent.parent / "preTrainedModels"


class ModelSize(Enum):
    """Model size categories."""

    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


@dataclass
class ModelInfo:
    """Information about a YOLO model."""

    name: str
    filename: str
    version: str
    size: ModelSize
    description: str
    params_millions: float
    expected_map50: float  # Expected mAP@0.5 on COCO
    expected_fps: float  # Expected FPS on GPU
    downloaded: bool = False

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename

    @property
    def exists(self) -> bool:
        return self.path.exists()


@dataclass
class BenchmarkResult:
    """Results from model benchmarking."""

    model_name: str
    device: str
    image_size: int
    warmup_runs: int
    benchmark_runs: int
    avg_inference_time_ms: float
    std_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float
    fps: float
    memory_used_mb: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "image_size": self.image_size,
            "warmup_runs": self.warmup_runs,
            "benchmark_runs": self.benchmark_runs,
            "avg_inference_time_ms": round(self.avg_inference_time_ms, 2),
            "std_inference_time_ms": round(self.std_inference_time_ms, 2),
            "min_inference_time_ms": round(self.min_inference_time_ms, 2),
            "max_inference_time_ms": round(self.max_inference_time_ms, 2),
            "fps": round(self.fps, 1),
            "memory_used_mb": round(self.memory_used_mb, 1),
            "timestamp": self.timestamp,
        }


def _discover_custom_models() -> dict[str, ModelInfo]:
    """
    Scan preTrainedModels directory for custom .pt files not in the predefined list.
    Returns a dictionary of custom models with default metadata.
    """
    custom_models = {}
    predefined_filenames = {
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov12n.pt", "yolov12s.pt", "yolov12m.pt", "yolov12l.pt", "yolov12x.pt",
    }
    
    if MODELS_DIR.exists():
        for pt_file in MODELS_DIR.glob("*.pt"):
            if pt_file.name not in predefined_filenames:
                model_key = pt_file.stem.replace("-", "_").replace(".", "_")
                display_name = pt_file.stem.replace("-", " ").replace("_", " ").title()
                
                version = "custom"
                if "yolov8" in pt_file.name.lower():
                    version = "v8"
                elif "yolov12" in pt_file.name.lower():
                    version = "v12"
                elif "yolo" in pt_file.name.lower():
                    for i in range(5, 30):
                        if f"yolov{i}" in pt_file.name.lower() or f"yolo{i}" in pt_file.name.lower():
                            version = f"v{i}"
                            break
                
                size = ModelSize.MEDIUM
                if "n" in pt_file.name.lower() and ("nano" in pt_file.name.lower() or pt_file.name.lower().endswith("n.pt")):
                    size = ModelSize.NANO
                elif "s" in pt_file.name.lower() and pt_file.name.lower().endswith("s.pt"):
                    size = ModelSize.SMALL
                elif "l" in pt_file.name.lower() and pt_file.name.lower().endswith("l.pt"):
                    size = ModelSize.LARGE
                elif "x" in pt_file.name.lower() and pt_file.name.lower().endswith("x.pt"):
                    size = ModelSize.XLARGE
                
                custom_models[model_key] = ModelInfo(
                    name=display_name,
                    filename=pt_file.name,
                    version=version,
                    size=size,
                    description=f"Custom model: {pt_file.name}",
                    params_millions=0.0,
                    expected_map50=0.0,
                    expected_fps=0,
                )
    
    return custom_models


# Available models registry
AVAILABLE_MODELS: dict[str, ModelInfo] = {
    "yolov8n": ModelInfo(
        name="YOLOv8 Nano",
        filename="yolov8n.pt",
        version="v8",
        size=ModelSize.NANO,
        description="Fastest YOLOv8 model, suitable for edge devices",
        params_millions=3.2,
        expected_map50=0.370,
        expected_fps=195,
    ),
    "yolov8s": ModelInfo(
        name="YOLOv8 Small",
        filename="yolov8s.pt",
        version="v8",
        size=ModelSize.SMALL,
        description="Fast and accurate, good for real-time applications",
        params_millions=11.2,
        expected_map50=0.449,
        expected_fps=142,
    ),
    "yolov8m": ModelInfo(
        name="YOLOv8 Medium",
        filename="yolov8m.pt",
        version="v8",
        size=ModelSize.MEDIUM,
        description="Balanced speed and accuracy",
        params_millions=25.9,
        expected_map50=0.502,
        expected_fps=103,
    ),
    "yolov8l": ModelInfo(
        name="YOLOv8 Large",
        filename="yolov8l.pt",
        version="v8",
        size=ModelSize.LARGE,
        description="High accuracy, requires more compute",
        params_millions=43.7,
        expected_map50=0.528,
        expected_fps=75,
    ),
    "yolov8x": ModelInfo(
        name="YOLOv8 XLarge",
        filename="yolov8x.pt",
        version="v8",
        size=ModelSize.XLARGE,
        description="Highest accuracy YOLOv8",
        params_millions=68.2,
        expected_map50=0.538,
        expected_fps=57,
    ),
    "yolov12n": ModelInfo(
        name="YOLOv12 Nano",
        filename="yolov12n.pt",
        version="v12",
        size=ModelSize.NANO,
        description="Latest YOLOv12 nano model",
        params_millions=3.5,
        expected_map50=0.385,
        expected_fps=180,
    ),
    "yolov12s": ModelInfo(
        name="YOLOv12 Small",
        filename="yolov12s.pt",
        version="v12",
        size=ModelSize.SMALL,
        description="Latest YOLOv12 small model",
        params_millions=12.0,
        expected_map50=0.465,
        expected_fps=135,
    ),
    "yolov12m": ModelInfo(
        name="YOLOv12 Medium",
        filename="yolov12m.pt",
        version="v12",
        size=ModelSize.MEDIUM,
        description="Latest YOLOv12 medium model",
        params_millions=28.0,
        expected_map50=0.520,
        expected_fps=95,
    ),
    "yolov12l": ModelInfo(
        name="YOLOv12 Large",
        filename="yolov12l.pt",
        version="v12",
        size=ModelSize.LARGE,
        description="Latest YOLOv12 large model",
        params_millions=48.0,
        expected_map50=0.545,
        expected_fps=68,
    ),
    "yolov12x": ModelInfo(
        name="YOLOv12 XLarge",
        filename="yolov12x.pt",
        version="v12",
        size=ModelSize.XLARGE,
        description="Latest YOLOv12 extra large - highest accuracy",
        params_millions=75.0,
        expected_map50=0.560,
        expected_fps=50,
    ),
}

# Merge custom models discovered from preTrainedModels folder
AVAILABLE_MODELS.update(_discover_custom_models())

class ModelManager:
    """
    Manager for multiple YOLO models.
    Handles loading, caching, switching, and benchmarking models.
    """

    _instance: Optional["ModelManager"] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._models: dict[str, Any] = {}  # Cached loaded models
        self._current_model_key: str = "yolov12x"  # Default model
        self._model_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._benchmark_results: dict[str, BenchmarkResult] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Ensure models directory exists
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Check which models are downloaded
        self._update_downloaded_status()

        self._initialized = True
        logger.info(f"ModelManager initialized. Device: {self._device}")

    def _update_downloaded_status(self):
        """Update the downloaded status of all models."""
        for key, info in AVAILABLE_MODELS.items():
            info.downloaded = info.exists

    @property
    def device(self) -> str:
        """Get current device (cuda/cpu)."""
        return self._device

    @property
    def current_model_key(self) -> str:
        """Get current model key."""
        return self._current_model_key

    @property
    def current_model_info(self) -> ModelInfo:
        """Get current model info."""
        return AVAILABLE_MODELS.get(
            self._current_model_key, AVAILABLE_MODELS["yolov12x"]
        )

    def list_available_models(self) -> list[dict]:
        """List all available models with their info."""
        self._update_downloaded_status()
        result = []
        for key, info in AVAILABLE_MODELS.items():
            result.append(
                {
                    "key": key,
                    "name": info.name,
                    "filename": info.filename,
                    "version": info.version,
                    "size": info.size.value,
                    "description": info.description,
                    "params_millions": info.params_millions,
                    "expected_map50": info.expected_map50,
                    "expected_fps": info.expected_fps,
                    "downloaded": info.exists,
                    "is_current": key == self._current_model_key,
                }
            )
        return result

    def list_downloaded_models(self) -> list[dict]:
        """List only downloaded models."""
        return [m for m in self.list_available_models() if m["downloaded"]]

    async def download_model(self, model_key: str) -> dict:
        """
        Download a model if not already present.
        Uses ultralytics auto-download feature.
        """
        if model_key not in AVAILABLE_MODELS:
            return {"success": False, "error": f"Unknown model: {model_key}"}

        info = AVAILABLE_MODELS[model_key]

        if info.exists:
            return {
                "success": True,
                "message": f"Model {info.name} already downloaded",
                "path": str(info.path),
            }

        try:
            logger.info(f"Downloading model: {info.name}")

            # Use ultralytics to download
            from ultralytics import YOLO

            # This will auto-download the model
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self._executor, lambda: YOLO(info.filename)
            )

            # Move the downloaded model to our models directory
            default_path = Path(info.filename)
            if default_path.exists() and not info.exists:
                import shutil

                shutil.move(str(default_path), str(info.path))

            # Cache the loaded model
            self._models[model_key] = model
            info.downloaded = True

            logger.info(f"Model {info.name} downloaded successfully")
            return {
                "success": True,
                "message": f"Model {info.name} downloaded successfully",
                "path": str(info.path),
            }

        except Exception as e:
            logger.error(f"Failed to download model {model_key}: {e}")
            return {"success": False, "error": str(e)}

    async def load_model(self, model_key: str) -> Any:
        """Load a model and cache it."""
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}")

        async with self._model_lock:
            # Return cached model if available
            if model_key in self._models:
                return self._models[model_key]

            info = AVAILABLE_MODELS[model_key]

            # Download if not exists
            if not info.exists:
                result = await self.download_model(model_key)
                if not result["success"]:
                    raise RuntimeError(f"Failed to download model: {result['error']}")
                # Model might be cached by download
                if model_key in self._models:
                    return self._models[model_key]

            # Load the model
            from ultralytics import YOLO

            logger.info(f"Loading model: {info.name}")

            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self._executor, lambda: YOLO(str(info.path))
            )

            # Move to appropriate device
            if self._device == "cuda":
                model.to("cuda")

            self._models[model_key] = model
            logger.info(f"Model {info.name} loaded successfully")

            return model

    async def switch_model(self, model_key: str) -> dict:
        """Switch to a different model."""
        if model_key not in AVAILABLE_MODELS:
            return {"success": False, "error": f"Unknown model: {model_key}"}

        try:
            # Load the new model (will use cache if available)
            await self.load_model(model_key)

            old_key = self._current_model_key
            self._current_model_key = model_key

            logger.info(f"Switched model from {old_key} to {model_key}")

            return {
                "success": True,
                "previous_model": old_key,
                "current_model": model_key,
                "model_info": self.list_available_models()[
                    list(AVAILABLE_MODELS.keys()).index(model_key)
                ],
            }

        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return {"success": False, "error": str(e)}

    async def get_current_model(self) -> Any:
        """Get the currently active model."""
        return await self.load_model(self._current_model_key)

    async def unload_model(self, model_key: str) -> dict:
        """Unload a model from memory."""
        if model_key not in self._models:
            return {"success": False, "error": f"Model {model_key} not loaded"}

        if model_key == self._current_model_key:
            return {"success": False, "error": "Cannot unload current active model"}

        async with self._model_lock:
            del self._models[model_key]

            # Force garbage collection
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"Model {model_key} unloaded")
        return {"success": True, "message": f"Model {model_key} unloaded"}

    async def benchmark_model(
        self,
        model_key: str,
        image_size: int = 640,
        warmup_runs: int = 10,
        benchmark_runs: int = 50,
    ) -> BenchmarkResult:
        """
        Benchmark a model's inference speed.

        Args:
            model_key: Key of the model to benchmark
            image_size: Input image size
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations

        Returns:
            BenchmarkResult with timing statistics
        """
        model = await self.load_model(model_key)

        # Create dummy input
        dummy_input = np.random.randint(
            0, 255, (image_size, image_size, 3), dtype=np.uint8
        )

        logger.info(f"Benchmarking {model_key} with {benchmark_runs} runs...")

        # Warmup
        for _ in range(warmup_runs):
            model(dummy_input, verbose=False)

        # Synchronize CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            model(dummy_input, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        times = np.array(times)

        # Get memory usage
        memory_mb = 0.0
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

        result = BenchmarkResult(
            model_name=model_key,
            device=self._device,
            image_size=image_size,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            avg_inference_time_ms=float(np.mean(times)),
            std_inference_time_ms=float(np.std(times)),
            min_inference_time_ms=float(np.min(times)),
            max_inference_time_ms=float(np.max(times)),
            fps=1000.0 / float(np.mean(times)),
            memory_used_mb=memory_mb,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Cache result
        self._benchmark_results[model_key] = result

        logger.info(
            f"Benchmark {model_key}: {result.fps:.1f} FPS, {result.avg_inference_time_ms:.2f}ms avg"
        )

        return result

    async def benchmark_all_downloaded(
        self,
        image_size: int = 640,
        warmup_runs: int = 10,
        benchmark_runs: int = 50,
    ) -> list[dict]:
        """Benchmark all downloaded models."""
        results = []

        for key, info in AVAILABLE_MODELS.items():
            if info.exists:
                try:
                    result = await self.benchmark_model(
                        key, image_size, warmup_runs, benchmark_runs
                    )
                    results.append(
                        {
                            **result.to_dict(),
                            "model_info": {
                                "name": info.name,
                                "version": info.version,
                                "size": info.size.value,
                                "params_millions": info.params_millions,
                                "expected_map50": info.expected_map50,
                                "expected_fps": info.expected_fps,
                            },
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to benchmark {key}: {e}")
                    results.append(
                        {
                            "model_name": key,
                            "error": str(e),
                        }
                    )

        return results

    def get_benchmark_results(self) -> dict[str, dict]:
        """Get all cached benchmark results."""
        return {k: v.to_dict() for k, v in self._benchmark_results.items()}

    def get_comparison_table(self) -> list[dict]:
        """
        Get a comparison table of all models with expected and actual metrics.
        """
        self._update_downloaded_status()
        table = []

        for key, info in AVAILABLE_MODELS.items():
            row = {
                "key": key,
                "name": info.name,
                "version": info.version,
                "size": info.size.value,
                "params_millions": info.params_millions,
                "expected_map50": info.expected_map50,
                "expected_fps": info.expected_fps,
                "downloaded": info.exists,
                "is_current": key == self._current_model_key,
            }

            # Add actual benchmark if available
            if key in self._benchmark_results:
                result = self._benchmark_results[key]
                row["actual_fps"] = round(result.fps, 1)
                row["actual_inference_ms"] = round(result.avg_inference_time_ms, 2)
                row["memory_mb"] = round(result.memory_used_mb, 1)

            table.append(row)

        return table


# Singleton instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
