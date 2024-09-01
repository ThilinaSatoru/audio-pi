import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import os

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # (yolov8n.pt) Using YOLOv8 nano for better performance

# Initialize variables for fall detection
prev_bbox = None
fall_threshold = 0.5  # Adjust as needed
fall_counter = 0
fall_frames = 5  # Number of consecutive frames to confirm a fall

# Create directories to save captured images
fall_dir = "fall_detected"
normal_dir = "normal_captures"
os.makedirs(fall_dir, exist_ok=True)
os.makedirs(normal_dir, exist_ok=True)

# Variables for controlling capture frequency
last_capture_time = 0
capture_interval = 5  # Capture every 5 seconds for non-fall images

def detect_fall(current_bbox, prev_bbox):
    if prev_bbox is None:
        return False
    
    # Calculate the change in y-coordinate of the bottom of the bounding box
    y_change = current_bbox[3] - prev_bbox[3]
    
    # If the change is greater than the threshold, consider it a fall
    return y_change > fall_threshold * current_bbox[3]

while True:
    # Capture frame
    frame = picam2.capture_array()
    
    # Run YOLO detection
    results = model(frame, classes=[0, 1], conf=0.5)  # Detect only persons (0) and children (1)
    
    fall_detected = False
    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Fall detection
            current_bbox = (x1, y1, x2, y2)
            if detect_fall(current_bbox, prev_bbox):
                fall_counter += 1
                if fall_counter >= fall_frames:
                    print("Fall detected!")
                    # Save captured image
                    timestamp = int(time.time())
                    cv2.imwrite(f"{fall_dir}/fall_detected_{timestamp}.jpg", frame)
                    fall_detected = True
                    fall_counter = 0
            else:
                fall_counter = 0
            
            prev_bbox = current_bbox
    
    # Save non-fall captures at regular intervals
    current_time = time.time()
    if not fall_detected and (current_time - last_capture_time) >= capture_interval:
        timestamp = int(current_time)
        cv2.imwrite(f"{normal_dir}/normal_capture_{timestamp}.jpg", frame)
        last_capture_time = current_time
    
    # Display the resulting frame
    # cv2.imshow('Human/Child Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
picam2.stop()