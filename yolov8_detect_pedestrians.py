import os
from ultralytics import YOLO


class YOLOv8PedestrianDetector:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)
        self.classes = [0]

    def detect(self, image_path, conf_threshold=0.5):
        results = self.model(
            image_path,
            conf=conf_threshold,
            task="detect",
        )
        pedestrians = [ped for ped in results[0].boxes if ped.cls == 0]

        if len(pedestrians):
            return True
        else:
            return False


pedDetector = YOLOv8PedestrianDetector()
demo_folder = "demo_test_images"
for img in os.listdir(demo_folder):
    img_path = f"{demo_folder}/{img}"
    result = pedDetector.detect(img_path)
    if result:
        print(f"{img_path.upper()}: Pedestrian Detected")
    else:
        print(f"{img_path.upper()}: NO Pedestrian Detected")