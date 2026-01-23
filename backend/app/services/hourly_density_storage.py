"""
Hourly Density Storage Service
Stores and retrieves hourly vehicle counts for density comparison.
"""
import json
import os
from datetime import datetime
from typing import Optional
from app.core.config import settings


def _get_density_dir() -> str:
    """Get the directory for density data files."""
    base_dir = os.path.join(os.path.dirname(settings.ZONES_DIR), "density")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _get_density_file_path(camera_id: str) -> str:
    """Get the file path for a camera's hourly density data."""
    safe_id = "".join(c if c.isalnum() else "_" for c in camera_id)
    return os.path.join(_get_density_dir(), f"{safe_id}_hourly.json")


class HourlyDensityStorage:
    """Stores and retrieves hourly vehicle counts for density comparison."""

    @staticmethod
    def _load_data(camera_id: str) -> dict:
        """Load hourly data from file."""
        path = _get_density_file_path(camera_id)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _save_data(camera_id: str, data: dict):
        """Save hourly data to file."""
        path = _get_density_file_path(camera_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving density data: {e}")

    @staticmethod
    def save_hourly_count(camera_id: str, hour: int, count: int):
        """
        Save a vehicle count for a specific hour.
        Keeps last 7 days of data per hour for averaging.
        """
        data = HourlyDensityStorage._load_data(camera_id)
        hour_key = str(hour)
        
        if hour_key not in data:
            data[hour_key] = []
        
        # Add new count with timestamp
        data[hour_key].append({
            "count": count,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 7 entries per hour (1 week of data)
        data[hour_key] = data[hour_key][-7:]
        
        HourlyDensityStorage._save_data(camera_id, data)

    @staticmethod
    def get_hourly_average(camera_id: str, hour: int) -> float:
        """
        Get the average vehicle count for a specific hour.
        Returns 0 if no historical data exists.
        """
        data = HourlyDensityStorage._load_data(camera_id)
        hour_key = str(hour)
        
        if hour_key not in data or len(data[hour_key]) == 0:
            return 0.0
        
        counts = [entry["count"] for entry in data[hour_key]]
        return sum(counts) / len(counts)

    @staticmethod
    def get_hourly_history(camera_id: str, hour: int) -> list[dict]:
        """Get historical data for a specific hour."""
        data = HourlyDensityStorage._load_data(camera_id)
        hour_key = str(hour)
        return data.get(hour_key, [])

    @staticmethod
    def get_all_hourly_data(camera_id: str) -> dict:
        """Get all hourly data for a camera."""
        return HourlyDensityStorage._load_data(camera_id)
