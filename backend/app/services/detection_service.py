import cv2
import numpy as np
import torch
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tracker = None
        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def process_frame(self, frame, conf_threshold=0.25):
        if not self.model:
            return frame, []
        
        # imgsz=1088: Matches 1080p height (multiple of 32), reduces scaling artifacts
        # agnostic_nms=True: Prevent overlapping car/truck/bus boxes
        # iou=0.45: Standard NMS threshold
        # This is not optimised, since it varies on the users hardware especially the GPU
        results = self.model.track(
            frame, 
            persist=True, 
            conf=conf_threshold, 
            imgsz=1088, 
            agnostic_nms=True,
            iou=0.45,
            classes=list(self.class_names.keys())
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                
                label = self.class_names.get(cls, 'vehicle')
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'label': label,
                    'conf': conf,
                    'track_id': track_id
                })

        return detections

    def count_unique_vehicles(self, detections, counted_ids, track_history):
        """
        counts vehicles based on unique IDs that have persisted for a few frames.
        Returns:
            new_counts: list of labels that were just counted (for incrementing global counters)
            counted_ids: updated set of IDs that have been counted
            track_history: updated dictionary of {track_id: frames_seen_count}
        """
        new_counts = []
        
        for det in detections:
            tid = det['track_id']
            if tid == -1:
                continue
            
            # Update history
            if tid not in track_history:
                track_history[tid] = 0
            track_history[tid] += 1
            
            # Count if unseen and robust (seen > 5 frames)
            # This avoids ghost detections cluttering the stats
            if tid not in counted_ids and track_history[tid] > 5:
                counted_ids.add(tid)
                new_counts.append(det['label'])
        
        return new_counts, counted_ids, track_history

    def check_parking(self, detections, parking_zone, parking_timers):
        """
        Checks if vehicles are stationary/present in a parking zone for too long.
        Args:
            detections: List of detection dicts
            parking_zone: List of [x, y] points defining the polygon
            parking_timers: Dictionary of {track_id: start_time}
        Returns:
            violators: Set of track_ids that are violating
            parking_timers: Updated timers
        """
        current_time = time.time()
        violators = set()
        
        # Convert zone to numpy array for cv2
        poly_pts = np.array(parking_zone, np.int32)
        poly_pts = poly_pts.reshape((-1, 1, 2))
        
        present_ids = set()

        for det in detections:
            tid = det['track_id']
            if tid == -1:
                continue
            
            x1, y1, x2, y2 = det['bbox']
            # specific point: bottom center of the vehicle
            cx, cy = (x1 + x2) / 2, y2 
            
            # Check if point is inside polygon (measureDist=False)
            # returns +1 (inside), -1 (outside), 0 (edge)
            is_inside = cv2.pointPolygonTest(poly_pts, (cx, cy), False) >= 0
            
            if is_inside:
                present_ids.add(tid)
                if tid not in parking_timers:
                    parking_timers[tid] = current_time
                
                # Check duration (Threshold: 5 seconds for demo)
                duration = current_time - parking_timers[tid]
                if duration > 5:
                    violators.add(tid)
            else:
                # Vehicle detected but outside zone - reset timer if it was tracked
                if tid in parking_timers:
                    del parking_timers[tid]
        
        for tid in list(parking_timers.keys()):
            if tid not in [d['track_id'] for d in detections if d['track_id'] != -1]:
                 # Vehicle lost. Reset timer.
                 del parking_timers[tid]

        return violators, parking_timers

    def annotate_frame(self, frame, detections, parking_zone=None, violators=None):
        # Colors (BGR)
        colors = {
            'car': (255, 255, 0),
            'motorcycle': (0, 255, 255),
            'bus': (255, 0, 255),
            'truck': (0, 140, 255),
            'vehicle': (0, 255, 0) 
        }

        # Draw Parking Zone
        if parking_zone is not None:
             pts = np.array(parking_zone, np.int32)
             pts = pts.reshape((-1, 1, 2))
             overlay = frame.copy()
             cv2.fillPoly(overlay, [pts], (0, 0, 255)) # Red zone
             cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
             cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label_text = f"{det['label']} {det['conf']:.2f}"
            color = colors.get(det['label'], colors['vehicle'])
            
            # Highlight Violators
            if violators and det['track_id'] in violators:
                color = (0, 0, 255) # Red
                label_text = "NO PARKING!"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4) # Thicker box
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(frame, label_text, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        
        return frame

