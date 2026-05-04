from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import cv2

import face_module_v1.config as config
from face_module_v1.cameras.base_worker import BaseCameraWorker
from face_module_v1.face_engine import FaceEngine


def pick_primary_person(persons: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not persons:
        return None
    return max(persons, key=lambda p: max(1, p["bbox"][2] - p["bbox"][0]) * max(1, p["bbox"][3] - p["bbox"][1]))


def clip_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def crop_upper_body_for_face(frame, person_box, top_ratio: float = 0.8, side_padding: float = 0.13):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = person_box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * side_padding)
    crop = clip_box([x1 - pad_x, y1, x2 + pad_x, y1 + int(bh * top_ratio)], w, h)
    if crop is None:
        return None, None
    cx1, cy1, cx2, cy2 = crop
    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return None, None
    return roi, crop


def scale_face_results(face_results: List[Dict[str, Any]], roi_box: List[int]) -> List[Dict[str, Any]]:
    rx1, ry1, _, _ = roi_box
    scaled = []
    for item in face_results:
        x1, y1, x2, y2 = item["bbox"]
        scaled.append({"bbox": [x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1], "name": item["name"], "distance": item["distance"]})
    return scaled


def select_best_face(face_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not face_results:
        return None
    knowns = [f for f in face_results if f.get("name") != "Unknown"]
    if knowns:
        return min(knowns, key=lambda f: f.get("distance", 999.0))
    return min(face_results, key=lambda f: f.get("distance", 999.0))


class DoorCameraWorker(BaseCameraWorker):
    def __init__(self, camera_cfg: dict):
        super().__init__(camera_cfg)
        self.face_engine = FaceEngine(
            det_size=(getattr(config, "MAX_FACE_SIZE", 320), getattr(config, "MAX_FACE_SIZE", 320)),
            model_name=getattr(config, "FACE_MODEL_NAME", getattr(config, "MODEL_NAME", "buffalo_s")),
            cache_path=getattr(config, "FACE_DB_CACHE_PATH", getattr(config, "FACE_DB_CACHE_FILE", None)),
        )
        self.face_engine.build_database(config.FACE_DB_DIR)

    def run(self):
        frame_count = 0
        prev_time = time.time()
        fps = 0.0
        last_persons: List[Dict[str, Any]] = []
        last_face_results: List[Dict[str, Any]] = []
        person_every = int(getattr(config, "PERSON_DETECT_EVERY_N_FRAMES", 3))
        face_every = int(getattr(config, "FACE_RECOG_EVERY_N_FRAMES", 2))
        recog_threshold = float(getattr(config, "RECOG_THRESHOLD", 0.9))

        while self.running:
            frame = self.stream.read()
            if frame is None:
                self.event_engine.flush_if_timeout(self.recorder)
                self.recorder.update(None)
                time.sleep(0.01)
                continue

            frame_count += 1
            self.recorder.update(frame)
            display = frame.copy()

            if frame_count % max(1, person_every) == 0:
                try:
                    last_persons = self.person_detector.detect(frame)
                except Exception as e:
                    print(f"[{self.camera_id}] person detect error: {e}")
                    last_persons = []

            primary = pick_primary_person(last_persons)
            if primary is not None and frame_count % max(1, face_every) == 0:
                roi, roi_box = crop_upper_body_for_face(frame, primary["bbox"])
                if roi is not None:
                    try:
                        results = self.face_engine.infer(roi, threshold=recog_threshold)
                        last_face_results = scale_face_results(results, roi_box)
                    except Exception as e:
                        print(f"[{self.camera_id}] face infer error: {e}")
                        last_face_results = []
            elif primary is None:
                last_face_results = []

            best_face = select_best_face(last_face_results)
            if best_face is not None:
                name = best_face.get("name", "Unknown")
                conf = max(0.0, 1.0 - float(best_face.get("distance", 1.0)))
            elif primary is not None:
                name = "Unknown"
                conf = float(primary.get("confidence", 0.0))
            else:
                name = "Unknown"
                conf = 0.0

            self.event_engine.process(
                person_present=primary is not None,
                person_name=name,
                confidence=conf,
                frame=frame,
                recorder=self.recorder,
                extra={"camera_id": self.camera_id, "person_count": len(last_persons), "stream_ok": self.stream.stream_ok},
            )
            self.event_engine.flush_if_timeout(self.recorder)

            for p in last_persons:
                x1, y1, x2, y2 = p["bbox"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(display, f"person {p.get('confidence', 0):.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            for f in last_face_results:
                x1, y1, x2, y2 = f["bbox"]
                face_name = f.get("name", "Unknown")
                dist = float(f.get("distance", 999.0))
                color = (0, 255, 0) if face_name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{face_name} | {dist:.2f}", (x1, max(20, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            self.draw_common_overlay(display, fps)
            self.show(display)
