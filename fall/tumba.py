import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

increment = 0
max_increment = 5

# Define fall detection threshold
FALL_THRESHOLD = 40

# Define the code to be executed when a fall is detected
# FALL_SCRIPT = "sms_script.py"
# AR_SCRIPT = "ar.py"

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Detect objects in frame
    results = model.predict(frame)

    # Loop through detected objects
    for result in results:
        # Get object class and confidence score
        class_id = int(result.boxes.cls[0])
        confidence = result.boxes.conf[0]

        # Check if object is a person and confidence score is above threshold
        if class_id == 0 and confidence > 0.5:
            # Get bounding box coordinates
            x1, y1, x2, y2 = result.boxes.xyxy[0]

            # Draw bounding box and label on frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Get box height and width
            box_height = y2 - y1
            box_width = x2 - x1

            # Check for fall
            if box_width < box_height * 0.5:
                increment += 1
                if increment >= max_increment:
                    print("Hello ji")
                    increment = 0


    # Display frame
    cv2.imshow("Frame", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()