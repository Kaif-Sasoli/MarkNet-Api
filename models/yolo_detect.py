from ultralytics import YOLO
from PIL import Image
import numpy as np

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_and_crop(self, img):
        """
        img: PIL.Image.Image or str (image path)
        Returns list of tuples: (cropped PIL image, class_id)
        """
        # Handle input type
        if isinstance(img, str):
            pil_img = Image.open(img).convert("RGB")
            results = self.model(img, conf=0.4, verbose=False)
        else:
            pil_img = img.convert("RGB")
            results = self.model(np.array(pil_img), conf=0.4, verbose=False)

        crops = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls.item())

            crop = pil_img.crop((x1, y1, x2, y2))
            crops.append((crop, cls))
        
        return crops
