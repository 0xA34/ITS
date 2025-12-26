from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.detection_service import VehicleDetector
from typing import List
import cv2
import time
import io

router = APIRouter(prefix="/detection", tags=["detection"])

router = APIRouter(prefix="/detection", tags=["detection"])

class DetectionConfig(BaseModel):
    source: str
    model_path: str = "yolov8n.pt" 
    line_coords: List[int] = [0, 500, 1920, 500] 


detector = VehicleDetector()
is_running = False
active_config: DetectionConfig | None = None

# Global state for counting
# Global state for counting
vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
counted_ids = set()
track_history = {}

def reset_counts():
    global vehicle_counts, counted_ids, track_history
    vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
    counted_ids = set()
    track_history = {}

def update_counts(new_labels):
    global vehicle_counts
    for label in new_labels:
        if label in vehicle_counts:
            vehicle_counts[label] += 1

def generate_frames():
    global is_running, active_config, counted_ids, track_history
    if not active_config:
        return

    source = active_config.source
    if source.isdigit():
        source = int(source)
    
    # No line coords needed for unique ID counting

    is_image_url = "ImageHandler.ashx" in str(source) or str(source).endswith((".jpg", ".png", ".jpeg"))
    
    if is_image_url:
        import requests
        import numpy as np
        import re
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://giaothong.hochiminhcity.gov.vn/"  
        }

        last_content = None
        last_frame_bytes = None

        while is_running:
            try:
                # 1. Update Timestamp
                current_time = str(int(time.time() * 1000))
                url_str = str(source)
                
                if "t=" in url_str:
                    url_str = re.sub(r't=\d+', f't={current_time}', url_str)
                else:
                    separator = "&" if "?" in url_str else "?"
                    url_str = f"{url_str}{separator}t={current_time}"

                # 2. Fetch
                resp = requests.get(url_str, headers=headers, timeout=5)
                
                if resp.status_code == 200:
                    current_content = resp.content
                    
                    # Optimization: If image is identical to last one, reuse previous result
                    if last_content is not None and current_content == last_content and last_frame_bytes is not None:
                        # Yield cached frame to keep stream alive
                         yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + last_frame_bytes + b'\r\n')
                         time.sleep(0.5)
                         continue
                    
                    last_content = current_content
                    arr = np.frombuffer(resp.content, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                         # Detect
                        detections = detector.process_frame(frame)
                        
                        # Count Unique IDs
                        new_counts, counted_ids, track_history = detector.count_unique_vehicles(
                            detections, counted_ids, track_history
                        )
                        update_counts(new_counts)
                        
                        # Annotate
                        frame = detector.annotate_frame(frame, detections)

                        # Encode
                        ret, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                        last_frame_bytes = frame_bytes
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.5) 
                
            except Exception as e:
                print(f"Error fetching image: {e}")
                time.sleep(1)
                continue
        return

    # Video/Stream logic
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while is_running and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Detect
        detections = detector.process_frame(frame)
        
        # Count Unique IDs
        new_counts, counted_ids, track_history = detector.count_unique_vehicles(
            detections, counted_ids, track_history
        )
        update_counts(new_counts)
        
        # Annotate
        frame = detector.annotate_frame(frame, detections)
        
        # No Line Drawing
        
        # Encode
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
    cap.release()

@router.post("/start")
async def start_detection(config: DetectionConfig):
    global is_running, active_config
    if is_running:
        return {"status": "already running"}
    
    try:
        reset_counts() # Reset counts on fresh start
        detector.load_model(config.model_path)
        is_running = True
        active_config = config
        return {"status": "started", "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_detection():
    global is_running, active_config
    if not is_running:
        return {"status": "not running"}
    
    is_running = False
    active_config = None 
    return {"status": "stopped"}

@router.get("/status")
async def get_status():
    return {
        "running": is_running, 
        "device": detector.device,
        "model_loaded": detector.model is not None,
        "source": active_config.source if active_config else None,
        "counts": vehicle_counts
    }

@router.get("/video_feed")
async def video_feed():
    if not is_running:
         # Return a placeholder image or 404? 
         # StreamingResponse with empty generator might block.
         # Better to raise error or return static image.
         raise HTTPException(status_code=400, detail="Detection not running")
         
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
@router.get("/snapshot")
async def get_snapshot(source: str):
    """
    Returns a single JPEG frame from the source (Image URL or Video/m3u8).
    Used for the Zone Editor.
    """
    if not source:
        raise HTTPException(status_code=400, detail="Source URL required")
    
    # 1. Handle Image URL (Proxy)
    is_image = "ImageHandler.ashx" in str(source) or str(source).endswith((".jpg", ".png", ".jpeg"))
    if is_image:
        import requests
        import re
        try:
             # Add timestamp to avoid cache
            current_time = str(int(time.time() * 1000))
            if "t=" in source:
                url_str = re.sub(r't=\d+', f't={current_time}', source)
            else:
                separator = "&" if "?" in source else "?"
                url_str = f"{source}{separator}t={current_time}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Referer": "https://giaothong.hochiminhcity.gov.vn/"  
            }
            resp = requests.get(url_str, headers=headers, timeout=5)
            if resp.status_code == 200:
                return StreamingResponse(io.BytesIO(resp.content), media_type="image/jpeg")
        except Exception as e:
            print(f"Snapshot proxy error: {e}")
            # Fallthrough to cv2 if requests fail? No, better error out or try cv2 as backup.
    
    # 2. Handle Video/Stream (Capture Frame)
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
             raise HTTPException(status_code=404, detail="Could not open video source")
        
        # Read a few frames to settle (optional, often first frame is black/keyframe issues)
        # m3u8 might be slow to start.
        success, frame = cap.read()
        cap.release()
        
        if success:
             ret, buffer = cv2.imencode('.jpg', frame)
             return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
        else:
             raise HTTPException(status_code=500, detail="Failed to capture frame")

    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
