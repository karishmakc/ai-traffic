from ultralytics import YOLO
import cv2

VIDEO_PATH = "data/videos/traffic.mp4"

model = YOLO("yolo11s.pt")

results = model.track(
    source=VIDEO_PATH,
    show=True,
    save=True,
    tracker="botsort.yaml",
    classes=[2, 5, 7],
    persist=True,
    conf=0.15,
    imgsz=1280,
    
)

print("Tracking completed!")
