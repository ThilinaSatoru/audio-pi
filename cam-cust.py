import cv2
from ultralytics import YOLO
from datetime import datetime
from picamera2 import Picamera2
import time
import os

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Initialize YOLO model
model = YOLO('falldetectionmodelv3.pt')

# Create directories for saving images
os.makedirs("fall_detected", exist_ok=True)
os.makedirs("captured_images", exist_ok=True)

# Initialize variables
frame_count = 0
save_interval = 30  # Save a non-fall image every 30 frames

while True:
    # Capture frame
    frame = picam2.capture_array()
    
    # Increment frame count
    frame_count += 1
    
    # Run YOLO detection
    results = model.track(frame, persist=True, conf=0.5)
    
    fall_detected = False
    
    for obj in results[0].boxes:
        class_id = int(obj.cls)
        if class_id in [0, 1]:  # 'Fall' or 'Not Fall'
            class_name = model.names[class_id]
            confidence = float(obj.conf)
            bbox = obj.xyxy[0].int().tolist()
            
            if confidence > 0.8 and class_name == 'Fall':
                fall_detected = True
                color = (0, 0, 255)  # Red color for 'Fall'
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, f'{class_name} {confidence:.2f}', (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Save fall detected image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"fall_detected/fall_{timestamp}.jpg", frame)
                
                print(f"Fall detected! Image saved as fall_{timestamp}.jpg")
    
    # Save non-fall images at regular intervals
    if not fall_detected and frame_count % save_interval == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"captured_images/frame_{timestamp}.jpg", frame)
    
    # Display the frame (optional, can be commented out for headless operation)
    # cv2.imshow("Fall Detection", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
picam2.stop()