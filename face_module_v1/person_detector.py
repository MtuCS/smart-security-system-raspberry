from __future__ import annotations

from typing import List, Dict, Any
import cv2
from ultralytics import YOLO


class PersonDetector:
    def __init__(
        self,
        model_path: str = "yolov8n_ncnn_model",
        conf: float = 0.4,
        imgsz: int = 640,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            classes=[0],      # COCO class 0 = person
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )

        persons: List[Dict[str, Any]] = []
        if not results:
            return persons

        r = results[0]
        if r.boxes is None:
            return persons

        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if cls_id != 0:
                continue

            persons.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
            })

        return persons