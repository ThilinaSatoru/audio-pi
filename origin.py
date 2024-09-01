import cv2
from ultralytics import YOLO
from datetime import datetime
import os

# Open the video file
cap = cv2.VideoCapture("child play.mp4")

# Load the YOLO model
model = YOLO('falldetectionmodelv3.pt')

# Set up frame skipping
frame_skip = 3  # Process every 3rd frame

# Create a directory to save images if it doesn't exist
output_dir = "detected_falls"
os.makedirs(output_dir, exist_ok=True)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Resize frame to reduce resolution
    frame = cv2.resize(frame, (640, 480))  # Adjust size as needed

    # Frame skipping logic
    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if frame_count % frame_skip != 0:
        continue

    # YOLO detection
    results = model.track(frame, persist=True, conf=0.5)

    detected_fall = False

    for obj in results[0].boxes:
        class_id = int(obj.cls)
        class_name = model.names[class_id]
        bbox = obj.xyxy[0]
        confidence = float(obj.conf)
        x1, y1, x2, y2 = bbox.int().tolist()

        if confidence > 0.8:
            if class_name == 'Fall':
                color = (0, 0, 255)  # Red color for 'Fall'
                detected_fall = True
            else:
                color = (0, 255, 0)  # Green color for 'Not Fall'

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if detected_fall:
        # Save the annotated frame if a fall is detected
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"fall_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Detected Fall and saved frame to {filename}")
    else:
        print("No fall detected in this frame.")

cap.release()
cv2.destroyAllWindows()
