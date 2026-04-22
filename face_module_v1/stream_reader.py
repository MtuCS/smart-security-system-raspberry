from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2

import face_module_v1.config as config


@dataclass
class SharedFrame:
    lock: threading.Lock = field(default_factory=threading.Lock)
    frame: Optional[any] = None
    running: bool = True
    stream_ok: bool = False
    frame_id: int = 0
    last_frame_ts: float = 0.0


class StreamReader:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.shared = SharedFrame()
        self._thread: Optional[threading.Thread] = None

    @property
    def stream_ok(self) -> bool:
        return self.shared.stream_ok

    def open_rtsp(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.shared.running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def read_latest(self):
        with self.shared.lock:
            frame = None if self.shared.frame is None else self.shared.frame.copy()
            frame_id = self.shared.frame_id
            stream_ok = self.shared.stream_ok
            last_frame_ts = self.shared.last_frame_ts
        return frame, frame_id, stream_ok, last_frame_ts

    def read(self):
        frame, _, _, _ = self.read_latest()
        return frame

    def _capture_loop(self):
        while self.shared.running:
            cap = self.open_rtsp()
            if not cap.isOpened():
                print("[WARN] Không mở được RTSP stream. Reconnect sau vài giây...")
                self.shared.stream_ok = False
                time.sleep(config.RECONNECT_DELAY)
                continue

            print("[INFO] RTSP connected.")
            self.shared.stream_ok = True

            while self.shared.running:
                for _ in range(max(getattr(config, "FRAME_GRAB_FLUSH_COUNT", 0), 0)):
                    cap.grab()
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[WARN] Mất frame. Đang reconnect...")
                    self.shared.stream_ok = False
                    cap.release()
                    time.sleep(config.RECONNECT_DELAY)
                    break
                with self.shared.lock:
                    self.shared.frame = frame
                    self.shared.frame_id += 1
                    self.shared.last_frame_ts = time.time()
            try:
                cap.release()
            except Exception:
                pass
