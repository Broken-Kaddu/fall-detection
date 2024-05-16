import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Define fall detection threshold
FALL_THRESHOLD = 40

# Define the code to be executed when a fall is detected
FALL_SCRIPT = "sms_script.py"

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

            # Check for fall
            head_x = int((x1 + x2) / 2)
            head_y = int(y1)
            torso_x = int((x1 + x2) / 2)
            torso_y = int((y1 + y2) / 2)
            feet_x = int((x1 + x2) / 2)
            feet_y = int(y2)

            head_to_torso_distance = ((head_y - torso_y) ** 2) ** 0.5
            torso_to_feet_distance = ((torso_y - feet_y) ** 2) ** 0.5

            if head_to_torso_distance > FALL_THRESHOLD and torso_to_feet_distance < FALL_THRESHOLD:
                # Trigger fall script
                os.system(FALL_SCRIPT)


    # Display frame
    cv2.imshow("Frame", frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()