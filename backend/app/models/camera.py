from pydantic import BaseModel, Field
from typing import Optional

class CameraOut(BaseModel):
    id: str = Field(..., description="Mongo _id as string")
    name: str
    url: str
    location: Optional[str] = None

class CamerasListOut(BaseModel):
    items: list[CameraOut]
    total: int
