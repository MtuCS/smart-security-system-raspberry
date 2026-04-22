from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

import face_module.config as config
from face_module.event_engine import Event
from face_module.stream_reader import BufferedFrame


class EventRecorder:
    def __init__(self):
        self.current_event: Optional[Event] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.clip_path: Optional[str] = None
        self.snapshot_path: Optional[str] = None
        self.last_event_seen_ts: float = 0.0
        self.frame_size = None

        Path(config.CLIP_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)

    def _make_clip_path(self, event: Event) -> str:
        safe_identity = event.identity.replace(" ", "_")
        return str(Path(config.CLIP_DIR) / f"{event.event_id}_{safe_identity}.mp4")

    def _make_snapshot_path(self, event: Event) -> str:
        safe_identity = event.identity.replace(" ", "_")
        return str(Path(config.SNAPSHOT_DIR) / f"{event.event_id}_{safe_identity}.jpg")

    def _create_writer(self, width: int, height: int, path: str) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        return cv2.VideoWriter(path, fourcc, config.RECORD_FPS, (width, height))

    def start_event(self, event: Event, pre_roll_frames: List[BufferedFrame], snapshot_frame: np.ndarray):
        if not config.ENABLE_RECORDING:
            return

        if snapshot_frame is None:
            return

        h, w = snapshot_frame.shape[:2]
        self.frame_size = (w, h)
        self.current_event = event
        self.last_event_seen_ts = time.time()
        self.clip_path = self._make_clip_path(event)
        self.snapshot_path = self._make_snapshot_path(event)
        self.writer = self._create_writer(w, h, self.clip_path)

        cv2.imwrite(self.snapshot_path, snapshot_frame)
        event.snapshot_path = self.snapshot_path
        event.clip_path = self.clip_path

        for item in pre_roll_frames:
            frm = item.frame
            if frm is None:
                continue
            if frm.shape[:2] != (h, w):
                frm = cv2.resize(frm, (w, h))
            self.writer.write(frm)

    def write_live_frame(self, frame: np.ndarray):
        if self.writer is None or self.current_event is None:
            return
        if frame is None:
            return
        h, w = frame.shape[:2]
        target_w, target_h = self.frame_size
        out = frame if (w, h) == (target_w, target_h) else cv2.resize(frame, (target_w, target_h))
        self.writer.write(out)

    def mark_seen(self):
        self.last_event_seen_ts = time.time()

    def maybe_finalize(self):
        if self.current_event is None or self.writer is None:
            return None

        if time.time() - self.last_event_seen_ts >= config.POST_ROLL_SEC:
            finished = self.current_event
            self.writer.release()
            self.writer = None
            self.current_event = None
            self.clip_path = None
            self.snapshot_path = None
            return finished
        return None

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.current_event = None
