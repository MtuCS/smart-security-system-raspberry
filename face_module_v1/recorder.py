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
    def __init__(self, base_dir=None, fps=None, pre_roll_seconds=None, post_roll_seconds=None):
        self.enabled = getattr(config, "ENABLE_RECORDING", True)
        self.events_dir = Path(base_dir or getattr(config, "EVENTS_DIR", "data/events"))
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps or getattr(config, "EVENT_VIDEO_FPS", 12))
        self.pre_roll_seconds = float(pre_roll_seconds or getattr(config, "EVENT_PRE_ROLL_SECONDS", 5.0))
        self.post_roll_seconds = float(post_roll_seconds or getattr(config, "EVENT_POST_ROLL_SECONDS", 6.0))
        self.buffer: Deque[BufferedFrame] = deque(maxlen=max(int(self.pre_roll_seconds * self.fps), 1))
        self.writer = None
        self.active_event = None
        self.active_video_path: Optional[Path] = None
        self.active_snapshot_path: Optional[Path] = None
        self.active_record_until: float = 0.0
        self.frame_size: Optional[Tuple[int, int]] = None
        self.last_saved_paths = None

    def update(self, frame: Optional[np.ndarray]):
        now = time.time()
        if frame is not None:
            self.buffer.append(BufferedFrame(ts=now, frame=frame.copy()))
            if self.writer is not None and self.frame_size is not None:
                out_frame = frame if frame.shape[1::-1] == self.frame_size else cv2.resize(frame, self.frame_size)
                self.writer.write(out_frame)
        if self.writer is not None and now > self.active_record_until:
            self._close_writer()

    def start_event(self, event, trigger_frame: np.ndarray):
        if not self.enabled:
            return None
        if self.writer is not None:
            self.active_record_until = max(self.active_record_until, time.time() + self.post_roll_seconds)
            return self.last_saved_paths

        event_dir = self.events_dir / time.strftime("%Y%m%d")
        event_dir.mkdir(parents=True, exist_ok=True)
        safe_name = str(event.person_name).replace("/", "_").replace("\\", "_").replace(" ", "_")
        stem = f"{time.strftime('%H%M%S')}_{event.event_id}_{event.event_type}_{safe_name}"
        snapshot_path = event_dir / f"{stem}.jpg"
        video_path = event_dir / f"{stem}.mp4"

        h, w = trigger_frame.shape[:2]
        self.frame_size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*getattr(config, "CODEC", "mp4v"))
        self.writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, self.frame_size)
        self.active_event = event
        self.active_video_path = video_path
        self.active_snapshot_path = snapshot_path
        self.active_record_until = time.time() + self.post_roll_seconds

        quality = [int(cv2.IMWRITE_JPEG_QUALITY), int(getattr(config, "SNAPSHOT_JPEG_QUALITY", 90))]
        cv2.imwrite(str(snapshot_path), trigger_frame, quality)

        for item in self.buffer:
            out = item.frame if item.frame.shape[1::-1] == self.frame_size else cv2.resize(item.frame, self.frame_size)
            self.writer.write(out)

        trigger_out = trigger_frame if trigger_frame.shape[1::-1] == self.frame_size else cv2.resize(trigger_frame, self.frame_size)
        self.writer.write(trigger_out)

        self.last_saved_paths = {"snapshot_path": str(snapshot_path), "video_path": str(video_path)}
        return self.last_saved_paths

    def mark_update(self, event):
        self.active_event = event
        if self.writer is not None:
            self.active_record_until = max(self.active_record_until, time.time() + self.post_roll_seconds)

    def force_close(self):
        self._close_writer()

    def _close_writer(self):
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.active_event = None
        self.active_video_path = None
        self.active_snapshot_path = None
        self.frame_size = None
