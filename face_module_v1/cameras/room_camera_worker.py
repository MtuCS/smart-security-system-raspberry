from __future__ import annotations

import time
from typing import Dict, List

import cv2

import face_module_v1.config as config
from face_module_v1.cameras.base_worker import BaseCameraWorker
from face_module_v1.tracking_logic import LineCrossingDetector, OccupancySmoother, SimpleIoUTracker


class RoomCameraWorker(BaseCameraWorker):
    def __init__(self, camera_cfg: dict):
        super().__init__(camera_cfg)
        self.tracker = SimpleIoUTracker(
            iou_threshold=getattr(config, "TRACK_IOU_THRESHOLD", 0.25),
            max_missed=getattr(config, "TRACK_MAX_MISSED", 12),
        )
        self.occupancy = OccupancySmoother(
            window_size=getattr(config, "OCCUPANCY_SMOOTHING_WINDOW", 3),
            min_agree=getattr(config, "OCCUPANCY_SMOOTHING_MIN_AGREE", 2),
        )
        self.line_crossing = LineCrossingDetector(
            p1_ratio=getattr(config, "VIRTUAL_LINE_P1_RATIO", (0.20, 0.50)),
            p2_ratio=getattr(config, "VIRTUAL_LINE_P2_RATIO", (0.85, 0.50)),
            in_direction=getattr(config, "VIRTUAL_LINE_IN_DIRECTION", "negative_to_positive"),
            min_move_px=getattr(config, "VIRTUAL_LINE_MIN_MOVE_PX", 10),
            cooldown_seconds=getattr(config, "VIRTUAL_LINE_COOLDOWN_SECONDS", 1.0),
        )

    def run(self):
        frame_count = 0
        prev_time = time.time()
        fps = 0.0
        last_persons: List[Dict] = []
        last_tracks = []
        confirmed_count = 0
        person_every = int(getattr(config, "PERSON_DETECT_EVERY_N_FRAMES", 3))

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
            h, w = frame.shape[:2]
            detection_updated = False

            if frame_count % max(1, person_every) == 0:
                try:
                    last_persons = self.person_detector.detect(frame)
                    last_tracks = self.tracker.update(last_persons)
                    confirmed_count, changed = self.occupancy.update(len(last_tracks))
                    detection_updated = True
                    if changed:
                        print(f"[{self.camera_id}] OCCUPANCY = {confirmed_count}")
                except Exception as e:
                    print(f"[{self.camera_id}] person detect/tracking error: {e}")
                    last_persons = []
                    last_tracks = []

            if detection_updated:
                for tr in last_tracks:
                    crossing = self.line_crossing.check(tr, w, h)
                    if crossing:
                        print(f"[{self.camera_id}] LINE CROSSING {crossing} | track_id={tr.track_id}")

            self.event_engine.process(
                person_present=confirmed_count > 0,
                person_name="Unknown",
                confidence=float(confirmed_count),
                frame=frame,
                recorder=self.recorder,
                extra={"camera_id": self.camera_id, "occupancy": confirmed_count, "person_count": len(last_persons), "stream_ok": self.stream.stream_ok},
            )
            self.event_engine.flush_if_timeout(self.recorder)

            for p in last_persons:
                x1, y1, x2, y2 = p["bbox"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(display, f"person {p.get('confidence', 0):.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            for tr in last_tracks:
                x1, y1, x2, y2 = tr.bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 180, 0), 2)
                cv2.circle(display, tr.center, 4, (255, 180, 0), -1)
                cv2.putText(display, f"ID {tr.track_id}", (x1, min(h - 10, y2 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

            if getattr(config, "DRAW_VIRTUAL_LINE", True):
                self.line_crossing.draw(display)

            cv2.putText(display, f"Occupancy: {confirmed_count}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            self.draw_common_overlay(display, fps)
            self.show(display)
