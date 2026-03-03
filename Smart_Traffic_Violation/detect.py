import cv2
from ultralytics import YOLO
import numpy as np
import argparse

# ================== CONFIG ==================
VIDEO_PATH = "input.mp4"      # তোমার CCTV video path
OUTPUT_PATH = "output_detected.mp4"
STOP_LINE_Y = 400             # Stop line এর Y position (প্রথম frame দেখে adjust করো)
SPEED_CALIBRATION = 0.05      # km/h এ convert করার factor (তোমার camera অনুযায়ী adjust করো)
CONF_THRESHOLD = 0.4
# ===========================================

# Load models (CPU তে force করা)
vehicle_model = YOLO("yolov8n.pt")          # Vehicles + Traffic Light
helmet_model = YOLO("helmet.pt")            # Helmet detection

# Tracker enable
vehicle_model.to('cpu')

# Red color HSV range for traffic light
LOWER_RED1 = np.array([0, 100, 100])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 100, 100])
UPPER_RED2 = np.array([180, 255, 255])

def is_red_light(light_box, frame):
    x1, y1, x2, y2 = map(int, light_box)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = cv2.bitwise_or(mask1, mask2)
    return cv2.countNonZero(mask) > 50  # red pixels আছে কি না

# Video capture
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Tracking history for speed
track_history = {}

print("🚀 Processing শুরু... (CPU তে slow হবে, ধৈর্য ধরো)")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    
    # Vehicle + Traffic Light detection with tracking
    results = vehicle_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD, device='cpu')
    
    red_light_active = False
    violations = []
    
    for result in results:
        boxes = result.boxes
        if boxes is None: continue
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None
            
            label = result.names[cls]
            
            # Traffic Light check
            if label == "traffic light":
                if is_red_light([x1, y1, x2, y2], frame):
                    red_light_active = True
                    cv2.putText(frame, "RED LIGHT", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
            # Vehicle detection
            if label in ["car", "motorcycle", "bus", "truck", "bicycle"]:
                # Stop line cross check
                center_y = (y1 + y2) // 2
                if center_y > STOP_LINE_Y and red_light_active:
                    violations.append(f"RED LIGHT JUMP + STOP LINE CROSS (ID:{track_id})")
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Overspeeding (simple calculation)
                if track_id and track_id in track_history:
                    prev_x, prev_y = track_history[track_id]
                    dist = abs(center_y - prev_y)
                    speed = dist * SPEED_CALIBRATION * fps  # approximate km/h
                    if speed > 60:
                        violations.append(f"OVERSPEEDING {speed:.0f} km/h (ID:{track_id})")
                        cv2.putText(frame, f"OVERSPEED {speed:.0f}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                if track_id:
                    track_history[track_id] = ((x1+x2)//2, center_y)
                
                # Helmet check (motorcycle এর জন্য)
                if label == "motorcycle":
                    # Rider crop (approximate)
                    rider_roi = frame[max(0, y1-50):y2, x1:x2]
                    if rider_roi.size > 0:
                        helmet_results = helmet_model(rider_roi, conf=0.4)
                        for h_res in helmet_results:
                            for h_box in h_res.boxes:
                                h_cls = int(h_box.cls[0])
                                h_label = h_res.names[h_cls]
                                if "no helmet" in h_label.lower() or h_label == "no_helmet":
                                    violations.append(f"NO HELMET (ID:{track_id})")
                                    cv2.putText(frame, "NO HELMET!", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 3)
    
    # Violation text on top
    if violations:
        y_offset = 30
        for v in violations:
            cv2.putText(frame, v, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            y_offset += 30
    
    # Stop line draw
    cv2.line(frame, (0, STOP_LINE_Y), (width, STOP_LINE_Y), (255, 255, 0), 3)
    cv2.putText(frame, "STOP LINE", (width//2-50, STOP_LINE_Y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    
    out.write(frame)
    cv2.imshow("Processing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Done! Output saved: {OUTPUT_PATH}")