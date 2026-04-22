from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

import face_module_v1.config as config


@dataclass
class BufferedFrame:
    ts: float
    frame: np.ndarray


class EventRecorder:
    def __init__(self):
        self.enabled = config.ENABLE_RECORDING
        self.events_dir = Path(config.EVENTS_DIR)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.buffer: Deque[BufferedFrame] = deque(maxlen=max(int(config.PRE_ROLL_SECONDS * config.VIDEO_FPS), 1))
        self.writer = None
        self.active_event_id: Optional[str] = None
        self.active_video_path: Optional[Path] = None
        self.active_snapshot_path: Optional[Path] = None
        self.active_record_until: float = 0.0
        self.frame_size: Optional[Tuple[int, int]] = None
        self.last_saved_paths = None

    def push_frame(self, frame: np.ndarray, ts: float):
        if frame is None:
            return
        self.buffer.append(BufferedFrame(ts=ts, frame=frame.copy()))
        if self.writer is not None and ts <= self.active_record_until:
            self.writer.write(frame)
        elif self.writer is not None and ts > self.active_record_until:
            self._close_writer()

    def start_event(self, event, trigger_frame: np.ndarray):
        if not self.enabled:
            return None
        if self.writer is not None:
            self.active_record_until = max(self.active_record_until, time.time() + config.POST_ROLL_SECONDS)
            return self.last_saved_paths

        event_dir = self.events_dir / time.strftime("%Y%m%d")
        event_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{time.strftime('%H%M%S')}_{event.event_id}_{event.event_type}_{event.person_name}"
        snapshot_path = event_dir / f"{stem}.jpg"
        video_path = event_dir / f"{stem}.mp4"

        h, w = trigger_frame.shape[:2]
        self.frame_size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*config.CODEC)
        self.writer = cv2.VideoWriter(str(video_path), fourcc, config.VIDEO_FPS, self.frame_size)
        self.active_event_id = event.event_id
        self.active_video_path = video_path
        self.active_snapshot_path = snapshot_path
        self.active_record_until = time.time() + config.POST_ROLL_SECONDS

        quality = [int(cv2.IMWRITE_JPEG_QUALITY), int(config.SNAPSHOT_JPEG_QUALITY)]
        cv2.imwrite(str(snapshot_path), trigger_frame, quality)

        for item in self.buffer:
            if item.frame.shape[1::-1] != self.frame_size:
                resized = cv2.resize(item.frame, self.frame_size)
                self.writer.write(resized)
            else:
                self.writer.write(item.frame)

        if trigger_frame.shape[1::-1] != self.frame_size:
            trigger_frame = cv2.resize(trigger_frame, self.frame_size)
        self.writer.write(trigger_frame)
        self.last_saved_paths = {
            "snapshot_path": str(snapshot_path),
            "video_path": str(video_path),
        }
        return self.last_saved_paths

    def finalize_if_needed(self, now: float):
        if self.writer is not None and now > self.active_record_until:
            self._close_writer()

    def _close_writer(self):
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.active_event_id = None
        self.active_video_path = None
        self.active_snapshot_path = None
        self.frame_size = None
