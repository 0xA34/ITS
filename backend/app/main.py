from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.middleware import RequestIdLoggingMiddleware
from app.database.mongo import connect_mongo, close_mongo
from app.routes.camera_routes import router as camera_router

app = FastAPI()

# CORS (dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(RequestIdLoggingMiddleware)

@app.on_event("startup")
async def on_startup():
    await connect_mongo()

@app.on_event("shutdown")
async def on_shutdown():
    await close_mongo()

@app.get("/api/health")
def health():
    return {"ok": True}

app.include_router(camera_router, prefix=settings.API_PREFIX)
