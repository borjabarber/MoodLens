from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("best.pt")
results = model.predict(source="0", show=True)