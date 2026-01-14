"""
Counting Service - Track vehicles crossing counting lines
"""
from datetime import datetime
from collections import defaultdict
from typing import Optional
import numpy as np

from app.models.detection import (
    Detection,
    ZonePolygon,
    CountingRecord,
    LineCounts,
    Point,
)


def _get_bbox_center(bbox) -> tuple[float, float]:
    """Get center point of bounding box"""
    return ((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2)


def _point_crosses_line(prev_pos: tuple, curr_pos: tuple, line_start: Point, line_end: Point) -> Optional[str]:
    """
    Check if movement from prev_pos to curr_pos crosses a line.
    Returns "in" or "out" based on crossing direction, or None if no crossing.
    
    Uses cross product to determine which side of line the points are on.
    """
    if prev_pos is None or curr_pos is None:
        return None
    
    # Line vector
    lx = line_end.x - line_start.x
    ly = line_end.y - line_start.y
    
    # Vector from line start to previous position
    px = prev_pos[0] - line_start.x
    py = prev_pos[1] - line_start.y
    
    # Vector from line start to current position
    cx = curr_pos[0] - line_start.x
    cy = curr_pos[1] - line_start.y
    
    # Cross products to determine which side of line
    prev_cross = lx * py - ly * px
    curr_cross = lx * cy - ly * cx
    
    # Check if crossing happened (signs differ)
    if prev_cross * curr_cross < 0:
        # Crossed the line
        if prev_cross < 0:
            return "in"  # Crossed from negative to positive side
        else:
            return "out"  # Crossed from positive to negative side
    
    return None


class CountingManager:
    """
    Manages counting lines for multiple cameras.
    Tracks previous positions and detects crossings.
    """
    
    _instances: dict[str, "CountingManager"] = {}
    
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.prev_positions: dict[int, tuple[float, float]] = {}  # track_id -> (x, y)
        self.line_counts: dict[str, LineCounts] = {}  # line_id -> counts
        self.crossed_tracks: dict[str, set[int]] = defaultdict(set)  # line_id -> set of track_ids that crossed
    
    @classmethod
    def get_instance(cls, camera_id: str) -> "CountingManager":
        if camera_id not in cls._instances:
            cls._instances[camera_id] = CountingManager(camera_id)
        return cls._instances[camera_id]
    
    @classmethod
    def reset(cls, camera_id: str):
        if camera_id in cls._instances:
            del cls._instances[camera_id]
    
    def update(
        self, 
        detections: list[Detection], 
        counting_lines: list[ZonePolygon]
    ) -> tuple[list[CountingRecord], dict[str, LineCounts]]:
        """
        Process detections and check for line crossings.
        Returns list of new crossing records and updated line counts.
        """
        records = []
        
        # Initialize line counts if needed
        for line in counting_lines:
            if line.id not in self.line_counts:
                self.line_counts[line.id] = LineCounts(
                    line_id=line.id,
                    line_name=line.name,
                )
        
        for det in detections:
            if det.track_id is None:
                continue
            
            curr_pos = _get_bbox_center(det.bbox)
            prev_pos = self.prev_positions.get(det.track_id)
            
            # Check each counting line
            for line in counting_lines:
                if len(line.points) < 2:
                    continue
                
                # Skip if already counted this track on this line
                if det.track_id in self.crossed_tracks[line.id]:
                    continue
                
                # Use first two points as line segment
                line_start = line.points[0]
                line_end = line.points[1]
                
                direction = _point_crosses_line(prev_pos, curr_pos, line_start, line_end)
                
                if direction:
                    # Check if this direction matches line's configured direction
                    if line.counting_direction != "both" and line.counting_direction != direction:
                        continue
                    
                    print(f"[COUNTING] Vehicle #{det.track_id} ({det.class_name}) crossed '{line.name}' going {direction}")
                    
                    # Record crossing
                    record = CountingRecord(
                        line_id=line.id,
                        line_name=line.name,
                        track_id=det.track_id,
                        vehicle_class=det.class_name,
                        direction=direction,
                        timestamp=datetime.now().isoformat(),
                    )
                    records.append(record)
                    
                    # Update counts
                    counts = self.line_counts[line.id]
                    if direction == "in":
                        counts.count_in += 1
                    else:
                        counts.count_out += 1
                    counts.total = counts.count_in + counts.count_out
                    
                    # Update by_class
                    if det.class_name not in counts.by_class:
                        counts.by_class[det.class_name] = {"in": 0, "out": 0}
                    counts.by_class[det.class_name][direction] += 1
                    
                    # Mark as crossed
                    self.crossed_tracks[line.id].add(det.track_id)
            
            # Update previous position
            self.prev_positions[det.track_id] = curr_pos
        
        # Clean up old tracks (not seen in current frame)
        current_track_ids = {det.track_id for det in detections if det.track_id is not None}
        old_tracks = set(self.prev_positions.keys()) - current_track_ids
        for track_id in old_tracks:
            del self.prev_positions[track_id]
            # Also remove from crossed_tracks after some time (vehicle left scene)
            for line_id in list(self.crossed_tracks.keys()):
                self.crossed_tracks[line_id].discard(track_id)
        
        return records, self.line_counts
    
    def get_counts(self) -> dict[str, LineCounts]:
        return self.line_counts
    
    def reset_counts(self):
        self.line_counts.clear()
        self.crossed_tracks.clear()
