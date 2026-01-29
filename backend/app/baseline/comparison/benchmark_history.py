"""
Benchmark History Module
========================
Persistent storage for your own benchmarked model results.

Usage:
    from app.baseline.comparison.benchmark_history import BenchmarkHistory
    
    history = BenchmarkHistory()
    
    # Save a benchmark result
    history.save_result(
        model_name="yolov8n",
        map_50=0.72,
        map_50_95=0.52,
        fps=150,
        ...
    )
    
    # Load all previous results
    all_results = history.load_all()
    
    # Clear history
    history.clear()
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """A single benchmark result."""
    
    model_name: str
    model_path: str = ""
    
    map_50: float = 0.0
    map_75: float = 0.0
    map_50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    fps: float = 0.0
    latency_ms: float = 0.0
    latency_std_ms: float = 0.0
    throughput: float = 0.0
    
    params_millions: float = 0.0
    flops_billions: float = 0.0
    model_size_mb: float = 0.0
    
    device: str = ""
    image_size: tuple[int, int] = (640, 640)
    dataset_name: str = ""
    num_images: int = 0
    
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "map_50": self.map_50,
            "map_75": self.map_75,
            "map_50_95": self.map_50_95,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "fps": self.fps,
            "latency_ms": self.latency_ms,
            "latency_std_ms": self.latency_std_ms,
            "throughput": self.throughput,
            "params_millions": self.params_millions,
            "flops_billions": self.flops_billions,
            "model_size_mb": self.model_size_mb,
            "device": self.device,
            "image_size": list(self.image_size),
            "dataset_name": self.dataset_name,
            "num_images": self.num_images,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkResult":
        image_size = data.get("image_size", [640, 640])
        if isinstance(image_size, list):
            image_size = tuple(image_size)
        
        return cls(
            model_name=data.get("model_name", ""),
            model_path=data.get("model_path", ""),
            map_50=data.get("map_50", 0.0),
            map_75=data.get("map_75", 0.0),
            map_50_95=data.get("map_50_95", 0.0),
            precision=data.get("precision", 0.0),
            recall=data.get("recall", 0.0),
            f1_score=data.get("f1_score", 0.0),
            fps=data.get("fps", 0.0),
            latency_ms=data.get("latency_ms", 0.0),
            latency_std_ms=data.get("latency_std_ms", 0.0),
            throughput=data.get("throughput", 0.0),
            params_millions=data.get("params_millions", 0.0),
            flops_billions=data.get("flops_billions", 0.0),
            model_size_mb=data.get("model_size_mb", 0.0),
            device=data.get("device", ""),
            image_size=image_size,
            dataset_name=data.get("dataset_name", ""),
            num_images=data.get("num_images", 0),
            timestamp=data.get("timestamp", ""),
        )


class BenchmarkHistory:
    """
    Persistent storage for benchmark results.
    
    Stores results in a JSON file so you can compare your own
    benchmarked models across multiple runs.
    """
    
    DEFAULT_HISTORY_FILE = "benchmark_history.json"
    
    def __init__(self, history_file: Optional[str] = None):
        """
        Initialize benchmark history.
        
        Args:
            history_file: Path to history JSON file. If None, uses default
                          location in baseline/results/
        """
        if history_file is None:
            base_dir = Path(__file__).parent.parent / "results"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.history_file = base_dir / self.DEFAULT_HISTORY_FILE
        else:
            self.history_file = Path(history_file)
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._results: dict[str, BenchmarkResult] = {}
        self._load()
    
    def _load(self) -> None:
        """Load history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                for model_name, result_data in data.get("results", {}).items():
                    self._results[model_name] = BenchmarkResult.from_dict(result_data)
                
                logger.info(f"Loaded {len(self._results)} benchmark results from history")
            except Exception as e:
                logger.warning(f"Could not load benchmark history: {e}")
                self._results = {}
        else:
            self._results = {}
    
    def _save(self) -> None:
        """Save history to file."""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "results": {name: result.to_dict() for name, result in self._results.items()},
            }
            
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self._results)} benchmark results to history")
        except Exception as e:
            logger.error(f"Could not save benchmark history: {e}")
    
    def save_result(
        self,
        model_name: str,
        model_path: str = "",
        map_50: float = 0.0,
        map_75: float = 0.0,
        map_50_95: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        f1_score: float = 0.0,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        latency_std_ms: float = 0.0,
        throughput: float = 0.0,
        params_millions: float = 0.0,
        flops_billions: float = 0.0,
        model_size_mb: float = 0.0,
        device: str = "",
        image_size: tuple[int, int] = (640, 640),
        dataset_name: str = "",
        num_images: int = 0,
    ) -> BenchmarkResult:
        """
        Save a benchmark result.
        
        If a result with the same model_name exists, it will be overwritten.
        
        Args:
            model_name: Unique name for this model
            ... (other metrics)
        
        Returns:
            The saved BenchmarkResult
        """
        result = BenchmarkResult(
            model_name=model_name,
            model_path=model_path,
            map_50=map_50,
            map_75=map_75,
            map_50_95=map_50_95,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            fps=fps,
            latency_ms=latency_ms,
            latency_std_ms=latency_std_ms,
            throughput=throughput,
            params_millions=params_millions,
            flops_billions=flops_billions,
            model_size_mb=model_size_mb,
            device=device,
            image_size=image_size,
            dataset_name=dataset_name,
            num_images=num_images,
            timestamp=datetime.now().isoformat(),
        )
        
        self._results[model_name] = result
        self._save()
        
        logger.info(f"Saved benchmark result for model: {model_name}")
        return result
    
    def save_from_evaluation_results(self, results: Any) -> BenchmarkResult:
        """
        Save from an EvaluationResults object.
        
        Args:
            results: EvaluationResults from run_baseline
        
        Returns:
            The saved BenchmarkResult
        """
        return self.save_result(
            model_name=results.model_name,
            model_path=results.model_path,
            map_50=results.map_50,
            map_75=results.map_75,
            map_50_95=results.map_50_95,
            precision=results.precision,
            recall=results.recall,
            f1_score=results.f1_score,
            fps=results.fps,
            latency_ms=results.latency_ms,
            latency_std_ms=results.latency_std_ms,
            throughput=results.throughput,
            dataset_name=results.dataset_name,
            num_images=results.num_images,
        )
    
    def load_all(self) -> dict[str, BenchmarkResult]:
        """
        Load all benchmark results.
        
        Returns:
            Dictionary mapping model_name to BenchmarkResult
        """
        return dict(self._results)
    
    def get_result(self, model_name: str) -> Optional[BenchmarkResult]:
        """
        Get a specific benchmark result.
        
        Args:
            model_name: Name of the model
        
        Returns:
            BenchmarkResult or None if not found
        """
        return self._results.get(model_name)
    
    def get_model_names(self) -> list[str]:
        """Get list of all benchmarked model names."""
        return list(self._results.keys())
    
    def remove_result(self, model_name: str) -> bool:
        """
        Remove a benchmark result.
        
        Args:
            model_name: Name of the model to remove
        
        Returns:
            True if removed, False if not found
        """
        if model_name in self._results:
            del self._results[model_name]
            self._save()
            logger.info(f"Removed benchmark result for model: {model_name}")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all benchmark history."""
        self._results = {}
        self._save()
        logger.info("Cleared all benchmark history")
    
    def __len__(self) -> int:
        return len(self._results)
    
    def __contains__(self, model_name: str) -> bool:
        return model_name in self._results
