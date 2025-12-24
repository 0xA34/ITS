from bson import ObjectId
from app.database.mongo import get_db

def _oid_to_str(doc: dict) -> dict:
    doc["id"] = str(doc["_id"])
    doc.pop("_id", None)
    return doc

class CameraService:
    @staticmethod
    async def list_cameras(limit: int = 50, skip: int = 0) -> tuple[list[dict], int]:
        db = get_db()
        col = db["link"]

        cursor = col.find({}, {"name": 1, "url": 1, "location": 1}).skip(skip).limit(limit)
        items = [ _oid_to_str(doc) async for doc in cursor ]
        total = await col.count_documents({})
        return items, total

    @staticmethod
    async def get_camera(camera_id: str) -> dict | None:
        db = get_db()
        col = db["link"]

        try:
            oid = ObjectId(camera_id)
        except Exception:
            return None

        doc = await col.find_one({"_id": oid}, {"name": 1, "url": 1, "location": 1})
        return _oid_to_str(doc) if doc else None
