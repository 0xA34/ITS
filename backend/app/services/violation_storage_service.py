import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from app.core.config import settings
from app.models.detection import (
    ParkingViolation,
    RedLightViolation,
    ZonePolygon,
    Detection,
)

logger = logging.getLogger(__name__)


class ViolationStorageService:
    """Service for storing and managing traffic violations (parking & red light)."""

    @staticmethod
    def _ensure_directory(camera_id: str, violation_type: str) -> Path:
        """Ensure violation directory exists for camera and violation type."""
        base_dir = Path(settings.VIOLATION_DATA_DIR)
        violation_dir = base_dir / camera_id / violation_type
        violation_dir.mkdir(parents=True, exist_ok=True)
        return violation_dir

    @staticmethod
    def _generate_filename(violation_type: str, track_id: int) -> str:
        """Generate unique filename for violation."""
        timestamp = datetime.now()
        unix_ts = int(timestamp.timestamp())
        date_str = timestamp.strftime("%Y-%m-%d")
        return f"{date_str}_{violation_type}_track{track_id}_{unix_ts}"

    @staticmethod
    async def save_annotated_frame(
        frame: np.ndarray, output_path: Path
    ) -> bool:
        """Save annotated frame as JPEG image."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
            )
            logger.info(f"Saved violation frame: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save frame {output_path}: {e}")
            return False

    @staticmethod
    async def _cleanup_old_violations(camera_id: str, violation_type: str):
        """Keep only the most recent MAX_VIOLATIONS_PER_CAMERA violations."""
        try:
            violation_dir = Path(settings.VIOLATION_DATA_DIR) / camera_id / violation_type
            if not violation_dir.exists():
                return

            # Get all JSON files sorted by modification time (newest first)
            json_files = sorted(
                violation_dir.glob("*.json"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )

            # Delete oldest files if we exceed the limit
            max_violations = settings.MAX_VIOLATIONS_PER_CAMERA
            if len(json_files) > max_violations:
                for old_file in json_files[max_violations:]:
                    # Delete JSON file
                    old_file.unlink()
                    # Delete corresponding image if exists
                    img_file = old_file.with_suffix(".jpg")
                    if img_file.exists():
                        img_file.unlink()
                    logger.info(f"Deleted old violation: {old_file.name}")

        except Exception as e:
            logger.error(f"Error cleaning up old violations: {e}")

    @staticmethod
    async def save_parking_violation(
        violation: ParkingViolation,
        camera_id: str,
        annotated_frame: Optional[np.ndarray] = None,
    ) -> bool:
        """Save parking violation to JSON file with optional annotated frame."""
        if not settings.SAVE_VIOLATIONS:
            return False

        try:
            violation_dir = ViolationStorageService._ensure_directory(
                camera_id, "parking"
            )
            
            filename_base = ViolationStorageService._generate_filename(
                "parking", violation.track_id
            )
            
            # Prepare violation data
            violation_data = {
                "violation_type": "parking",
                "camera_id": camera_id,
                "timestamp": datetime.now().isoformat(),
                "track_id": violation.track_id,
                "vehicle_class": violation.vehicle_class,
                "zone_id": violation.zone_id,
                "zone_name": violation.zone_name,
                "bbox": violation.bbox.model_dump(),
                "duration_seconds": violation.duration_seconds,
            }

            # Save annotated frame if enabled and provided
            if settings.SAVE_VIOLATION_FRAMES and annotated_frame is not None:
                img_filename = f"{filename_base}.jpg"
                img_path = violation_dir / img_filename
                await ViolationStorageService.save_annotated_frame(
                    annotated_frame, img_path
                )
                violation_data["image_filename"] = img_filename

            # Save JSON
            json_path = violation_dir / f"{filename_base}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(violation_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Saved parking violation: camera={camera_id}, "
                f"track={violation.track_id}, zone={violation.zone_name}"
            )

            # Cleanup old violations
            await ViolationStorageService._cleanup_old_violations(
                camera_id, "parking"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save parking violation: {e}")
            return False

    @staticmethod
    async def save_red_light_violation(
        violation: RedLightViolation,
        camera_id: str,
        annotated_frame: Optional[np.ndarray] = None,
    ) -> bool:
        """Save red light violation to JSON file with optional annotated frame."""
        if not settings.SAVE_VIOLATIONS:
            return False

        try:
            violation_dir = ViolationStorageService._ensure_directory(
                camera_id, "red_light"
            )
            
            filename_base = ViolationStorageService._generate_filename(
                "redlight", violation.track_id
            )
            
            # Prepare violation data
            violation_data = {
                "violation_type": "red_light",
                "camera_id": camera_id,
                "timestamp": violation.timestamp,
                "track_id": violation.track_id,
                "vehicle_class": violation.vehicle_class,
                "zone_id": violation.zone_id,
                "zone_name": violation.zone_name,
                "bbox": violation.bbox.model_dump(),
            }

            # Save annotated frame if enabled and provided
            if settings.SAVE_VIOLATION_FRAMES and annotated_frame is not None:
                img_filename = f"{filename_base}.jpg"
                img_path = violation_dir / img_filename
                await ViolationStorageService.save_annotated_frame(
                    annotated_frame, img_path
                )
                violation_data["image_filename"] = img_filename

            # Save JSON
            json_path = violation_dir / f"{filename_base}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(violation_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Saved red light violation: camera={camera_id}, "
                f"track={violation.track_id}, zone={violation.zone_name}"
            )

            # Cleanup old violations
            await ViolationStorageService._cleanup_old_violations(
                camera_id, "red_light"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save red light violation: {e}")
            return False

    @staticmethod
    def get_violations(
        camera_id: str,
        violation_type: Optional[str] = None,
        limit: int = 100
    ) -> list[dict]:
        """Get stored violations for a camera."""
        try:
            base_dir = Path(settings.VIOLATION_DATA_DIR) / camera_id
            if not base_dir.exists():
                return []

            violations = []
            
            # Determine which violation types to retrieve
            types_to_check = []
            if violation_type:
                types_to_check.append(violation_type)
            else:
                types_to_check = ["parking", "red_light"]

            for v_type in types_to_check:
                type_dir = base_dir / v_type
                if not type_dir.exists():
                    continue

                json_files = sorted(
                    type_dir.glob("*.json"),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True
                )

                for json_file in json_files[:limit]:
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            violation_data = json.load(f)
                            violations.append(violation_data)
                    except Exception as e:
                        logger.error(f"Error reading {json_file}: {e}")

            return sorted(
                violations,
                key=lambda v: v.get("timestamp", ""),
                reverse=True
            )[:limit]

        except Exception as e:
            logger.error(f"Error getting violations: {e}")
            return []
