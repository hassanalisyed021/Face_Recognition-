# File: run_yolov8_image_detection.py

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('50.pt')  # Replace with your model path (e.g., 'best.pt' or 'last.pt')

# Configuration
input_image_path = 'Faces/WhatsApp Image 2024-12-31 at 03.21.21.jpeg'  # Path to your image

# Load the image
image = cv2.imread(input_image_path)
if image is None:
    print(f"Error: Could not load image from {input_image_path}.")
    exit()

# Resize the image to 640x640 (YOLOv8 default size)
image = cv2.resize(image, (640, 640))

# Run inference with saving enabled
results = model(image, conf=0.3, save=True)  # Automatically saves results in 'runs' folder

# Check detections and print information
if results:
    print("Detections:")
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            print(f"- {label} at [{x1}, {y1}, {x2}, {y2}]")
else:
    print("No detections found.")

print("Results saved in YOLOv8's default directory (e.g., runs/detect/).")
