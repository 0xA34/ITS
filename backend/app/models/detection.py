from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class VehicleClass(str, Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"
    BICYCLE = "bicycle"
    PERSON = "person"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    bbox: BoundingBox
    class_name: str
    class_id: int
    confidence: float
    track_id: Optional[int] = None


class DetectionResult(BaseModel):
    detections: list[Detection]
    vehicle_count: dict[str, int]
    total_count: int
    frame_width: int
    frame_height: int
    processing_time_ms: float


class Point(BaseModel):
    x: float
    y: float


class ZonePolygon(BaseModel):
    id: str
    name: str
    points: list[Point]
    is_parking_zone: bool = False
    is_traffic_light: bool = False
    is_red_light: bool = False
    is_stop_line: bool = False  # Stop line zone for red light violation detection
    linked_traffic_light_id: Optional[str] = None  # ID of linked traffic light zone
    is_counting_line: bool = False  # Counting line for vehicle counting
    counting_direction: str = "both"  # "in", "out", or "both"
    is_ignore_zone: bool = False  # Ignore zone - vehicles here won't be counted for density or even detect
    color: str = "#00FF00"


class ZoneConfig(BaseModel):
    camera_id: str
    zones: list[ZonePolygon]


class ParkingViolation(BaseModel):
    track_id: int
    vehicle_class: str
    zone_id: str
    zone_name: str
    duration_seconds: float
    bbox: BoundingBox


class RedLightViolation(BaseModel):
    track_id: int
    vehicle_class: str
    zone_id: str
    zone_name: str
    bbox: BoundingBox
    timestamp: str


class TrafficStats(BaseModel):
    camera_id: str
    timestamp: str
    vehicle_counts: dict[str, int]
    total_vehicles: int
    parking_violations: list[ParkingViolation]
    red_light_violations: list[RedLightViolation] = Field(default_factory=list)
    zones_occupancy: dict[str, int]


class DetectRequest(BaseModel):
    image_url: str
    camera_id: Optional[str] = None
    include_zones: bool = False


class VehicleCounting(BaseModel):
    track_id: int
    vehicle_class: str
    zone_id: str
    zone_name: str
    crossed_at: str
    direction: str  # "in", "out", or "unknown"


class CountingStats(BaseModel):
    zone_id: str
    zone_name: str
    total_count: int
    count_by_class: dict[str, int] = Field(default_factory=dict)
    count_in: int = 0
    count_out: int = 0
    recent_crossings: list[VehicleCounting] = Field(default_factory=list)


class CountingRecord(BaseModel):
    """Record of a vehicle crossing a counting line"""
    line_id: str
    line_name: str
    track_id: int
    vehicle_class: str
    direction: str  # "in" or "out"
    timestamp: str


class LineCounts(BaseModel):
    """Counting statistics for a single line"""
    line_id: str
    line_name: str
    total: int = 0
    count_in: int = 0
    count_out: int = 0
    by_class: dict[str, dict[str, int]] = Field(default_factory=dict)  # {class_name: {"in": X, "out": Y}}


class DetectResponse(BaseModel):
    success: bool
    result: Optional[DetectionResult] = None
    violations: list[ParkingViolation] = Field(default_factory=list)
    red_light_violations: list[RedLightViolation] = Field(default_factory=list)
    error: Optional[str] = None
    model_info: Optional[dict] = None


class TrafficDensityConfig(BaseModel):
    """Configuration for traffic density tracking."""
    duration_minutes: int = 15


class TrafficDensityResult(BaseModel):
    """Result of traffic density calculation."""
    camera_id: str
    start_time: str
    end_time: str
    duration_minutes: int
    hour: int
    total_vehicles: int
    flow_rate: float  # q = n/t (vehicles per hour)
    hourly_average: float
    density_percentage: float
    density_level: Literal["heavy", "medium", "light"]
    density_label: str
    vehicle_breakdown: dict[str, int] = Field(default_factory=dict)


class TrafficDensityStatus(BaseModel):
    """Current status of traffic density tracking."""
    is_tracking: bool
    current_count: int
    elapsed_minutes: float = 0
    duration_minutes: int = 0
    hour: int = 0

