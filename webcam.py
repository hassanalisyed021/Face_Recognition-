# File: run_yolov8_webcam.py

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('50.pt')  # Replace with your trained model path

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam; change if multiple webcams are connected

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit the webcam feed.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # YOLOv8 expects images in (640, 640) size but we maintain the aspect ratio
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB (YOLOv8 works with RGB)
    height, width, _ = frame.shape

    # Run YOLOv8 inference
    results = model(frame_rgb, conf=0.8)  # Lower confidence for real-time detection

    # Draw detections on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw bounding box
            color = (0, 255, 0)  # Green for all detections
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Webcam Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
