from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os

# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best.pt")

model = YOLO(model_path)
cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 == 0:  # Only predict every 3rd frame
        results = model.predict(source=frame, show=True, verbose=False)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# face\Scripts\activate