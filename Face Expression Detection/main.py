from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best.pt")

model = YOLO(model_path)

results =model.predict(source ="0",show = True)


# face\Scripts\activate