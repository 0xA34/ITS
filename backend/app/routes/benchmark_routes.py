"""
Benchmark Routes
================
API routes for model management and benchmarking.
"""

from typing import Optional

from app.services.model_manager import AVAILABLE_MODELS, get_model_manager
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/benchmark", tags=["Benchmark"])


# Request/Response models
class DownloadModelRequest(BaseModel):
    model_key: str


class SwitchModelRequest(BaseModel):
    model_key: str


class BenchmarkRequest(BaseModel):
    model_key: Optional[str] = None  # If None, benchmark current model
    image_size: int = 640
    warmup_runs: int = 10
    benchmark_runs: int = 50


class BenchmarkAllRequest(BaseModel):
    image_size: int = 640
    warmup_runs: int = 10
    benchmark_runs: int = 50


# Routes
@router.get("/models")
async def list_models():
    """
    List all available models with their information.
    Returns both downloaded and not-yet-downloaded models.
    """
    manager = get_model_manager()
    return {
        "models": manager.list_available_models(),
        "current_model": manager.current_model_key,
        "device": manager.device,
    }


@router.get("/models/downloaded")
async def list_downloaded_models():
    """
    List only downloaded models.
    """
    manager = get_model_manager()
    return {
        "models": manager.list_downloaded_models(),
        "current_model": manager.current_model_key,
    }


@router.get("/models/current")
async def get_current_model():
    """
    Get information about the currently active model.
    """
    manager = get_model_manager()
    info = manager.current_model_info
    return {
        "key": manager.current_model_key,
        "name": info.name,
        "version": info.version,
        "size": info.size.value,
        "description": info.description,
        "params_millions": info.params_millions,
        "expected_map50": info.expected_map50,
        "expected_fps": info.expected_fps,
        "device": manager.device,
    }


@router.post("/models/download")
async def download_model(request: DownloadModelRequest):
    """
    Download a model if not already present.
    Uses ultralytics auto-download feature.
    """
    if request.model_key not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {request.model_key}. Available: {list(AVAILABLE_MODELS.keys())}",
        )

    manager = get_model_manager()
    result = await manager.download_model(request.model_key)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@router.post("/models/switch")
async def switch_model(request: SwitchModelRequest):
    """
    Switch to a different model.
    Will download the model if not present.
    """
    if request.model_key not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {request.model_key}. Available: {list(AVAILABLE_MODELS.keys())}",
        )

    manager = get_model_manager()
    result = await manager.switch_model(request.model_key)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@router.post("/models/unload")
async def unload_model(request: DownloadModelRequest):
    """
    Unload a model from memory to free resources.
    Cannot unload the currently active model.
    """
    manager = get_model_manager()
    result = await manager.unload_model(request.model_key)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/run")
async def run_benchmark(request: BenchmarkRequest):
    """
    Run benchmark on a specific model.
    If model_key is not provided, benchmarks the current model.
    """
    manager = get_model_manager()

    model_key = request.model_key or manager.current_model_key

    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_key}",
        )

    try:
        result = await manager.benchmark_model(
            model_key=model_key,
            image_size=request.image_size,
            warmup_runs=request.warmup_runs,
            benchmark_runs=request.benchmark_runs,
        )

        info = AVAILABLE_MODELS[model_key]

        return {
            "success": True,
            "benchmark": result.to_dict(),
            "model_info": {
                "name": info.name,
                "version": info.version,
                "size": info.size.value,
                "params_millions": info.params_millions,
                "expected_map50": info.expected_map50,
                "expected_fps": info.expected_fps,
            },
            "comparison": {
                "expected_fps": info.expected_fps,
                "actual_fps": round(result.fps, 1),
                "fps_difference": round(result.fps - info.expected_fps, 1),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/all")
async def run_benchmark_all(request: BenchmarkAllRequest):
    """
    Run benchmark on all downloaded models.
    This may take a while depending on how many models are downloaded.
    """
    manager = get_model_manager()

    try:
        results = await manager.benchmark_all_downloaded(
            image_size=request.image_size,
            warmup_runs=request.warmup_runs,
            benchmark_runs=request.benchmark_runs,
        )

        return {
            "success": True,
            "device": manager.device,
            "total_models": len(results),
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def get_benchmark_results():
    """
    Get all cached benchmark results.
    """
    manager = get_model_manager()
    return {
        "results": manager.get_benchmark_results(),
        "device": manager.device,
    }


@router.get("/comparison")
async def get_comparison_table():
    """
    Get a comparison table of all models with expected and actual metrics.
    Useful for generating comparison charts/tables in the frontend.
    """
    manager = get_model_manager()
    return {
        "comparison": manager.get_comparison_table(),
        "device": manager.device,
        "current_model": manager.current_model_key,
    }


@router.get("/device")
async def get_device_info():
    """
    Get information about the inference device.
    """
    import torch

    manager = get_model_manager()

    device_info = {
        "device": manager.device,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        device_info.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_memory_total_mb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 1
                ),
                "cuda_memory_allocated_mb": round(
                    torch.cuda.memory_allocated(0) / 1024 / 1024, 1
                ),
                "cuda_memory_cached_mb": round(
                    torch.cuda.memory_reserved(0) / 1024 / 1024, 1
                ),
            }
        )

    return device_info
