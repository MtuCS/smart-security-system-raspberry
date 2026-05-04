from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2

import face_module_v1.config as config
from face_module_v1.event_engine import EventEngine
from face_module_v1.person_detector import PersonDetector
from face_module_v1.recorder import EventRecorder
from face_module_v1.stream_reader import StreamReader
from face_module_v1.tracking_logic import resize_keep_ratio


class BaseCameraWorker:
    def __init__(self, camera_cfg: dict):
        self.camera_cfg = camera_cfg
        self.camera_id = camera_cfg["camera_id"]
        self.name = camera_cfg.get("name", self.camera_id)
        self.rtsp_url = camera_cfg["rtsp_url"]
        self.window_name = camera_cfg.get("window_name", self.name)
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.stream = StreamReader(self.rtsp_url)
        self.person_detector = PersonDetector(
            model_path=getattr(config, "PERSON_MODEL_PATH", "yolov8n_ncnn_model"),
            conf=getattr(config, "PERSON_CONFIDENCE", 0.4),
            imgsz=getattr(config, "PERSON_INFER_WIDTH", 640),
        )
        self.recorder = EventRecorder(
            base_dir=getattr(config, "EVENTS_DIR", Path("data/events")),
            camera_id=self.camera_id,
            fps=getattr(config, "EVENT_VIDEO_FPS", 12),
            pre_roll_seconds=getattr(config, "EVENT_PRE_ROLL_SECONDS", 5.0),
            post_roll_seconds=getattr(config, "EVENT_POST_ROLL_SECONDS", 6.0),
        )
        self.event_engine = EventEngine(
            camera_id=self.camera_id,
            start_confirm_frames=getattr(config, "EVENT_START_CONFIRM_FRAMES", 2),
            identity_confirm_frames=getattr(config, "EVENT_IDENTITY_CONFIRM_FRAMES", 3),
            end_absence_seconds=getattr(config, "EVENT_END_ABSENCE_SECONDS", 2.0),
            start_cooldown_seconds=getattr(config, "EVENT_START_COOLDOWN_SECONDS", 1.5),
            min_update_interval_seconds=getattr(config, "EVENT_MIN_UPDATE_INTERVAL_SECONDS", 0.8),
        )

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.stream.start()
        self.thread = threading.Thread(target=self.run, name=f"{self.camera_id}-worker", daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        try:
            self.stream.stop()
        except Exception:
            pass
        try:
            self.recorder.force_close()
        except Exception:
            pass
        if self.thread:
            self.thread.join(timeout=2.0)

    def run(self):
        raise NotImplementedError

    def draw_common_overlay(self, frame, fps: float):
        if not getattr(config, "SHOW_FPS", True):
            return frame
        status = "STREAM OK" if self.stream.stream_ok else "RECONNECTING"
        cv2.putText(frame, f"{self.name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | {status}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, self.event_engine.current_overlay_text(), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        return frame

    def show(self, frame):
        if not getattr(config, "SHOW_WINDOWS", True):
            return
        frame = resize_keep_ratio(frame, getattr(config, "DISPLAY_WIDTH", 960))
        cv2.imshow(self.window_name, frame)
