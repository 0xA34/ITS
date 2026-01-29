"""
Routes for managing saved traffic violations.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from app.services.violation_storage_service import ViolationStorageService

router = APIRouter(prefix="/api/violations", tags=["violations"])


@router.get("/{camera_id}")
async def get_violations(
    camera_id: str,
    limit: int = 100
):
    """
    Get all stored violations for a camera.
    
    Args:
        camera_id: Camera identifier
        limit: Maximum number of violations to return (default: 100)
    
    Returns:
        List of violations with metadata and image filenames
    """
    try:
        violations = ViolationStorageService.get_violations(
            camera_id=camera_id,
            limit=limit
        )
        return {
            "camera_id": camera_id,
            "count": len(violations),
            "violations": violations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{camera_id}/{violation_type}")
async def get_violations_by_type(
    camera_id: str,
    violation_type: str,
    limit: int = 100
):
    """
    Get stored violations of a specific type for a camera.
    
    Args:
        camera_id: Camera identifier
        violation_type: Type of violation ('parking' or 'red_light')
        limit: Maximum number of violations to return (default: 100)
    
    Returns:
        List of violations of the specified type
    """
    if violation_type not in ["parking", "red_light"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid violation_type. Must be 'parking' or 'red_light'"
        )
    
    try:
        violations = ViolationStorageService.get_violations(
            camera_id=camera_id,
            violation_type=violation_type,
            limit=limit
        )
        return {
            "camera_id": camera_id,
            "violation_type": violation_type,
            "count": len(violations),
            "violations": violations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
