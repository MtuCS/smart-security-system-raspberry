from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque

import cv2
import numpy as np

import face_module.config as config


@dataclass
class BufferedFrame:
    frame: np.ndarray
    ts: float


class StreamReader:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.lock = threading.Lock()
        self.running = False
        self.stream_ok = False
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_ts: float = 0.0
        self.frame_buffer: Deque[BufferedFrame] = deque(maxlen=max(1, int(config.PRE_ROLL_SEC * config.RECORD_FPS * 1.5)))
        self.thread: Optional[threading.Thread] = None

    def open_rtsp(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAPTURE_BUFFER_SIZE)
        return cap

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_pre_roll_frames(self):
        with self.lock:
            return list(self.frame_buffer)

    def is_stale(self) -> bool:
        return time.time() - self.latest_ts > config.STREAM_STALE_SEC

    def _capture_loop(self):
        while self.running:
            cap = self.open_rtsp()
            if not cap.isOpened():
                print("[WARN] Không mở được RTSP stream. Reconnect sau vài giây...")
                self.stream_ok = False
                time.sleep(config.RECONNECT_DELAY)
                continue

            print("[INFO] RTSP connected.")
            self.stream_ok = True

            while self.running:
                for _ in range(max(0, config.CAPTURE_GRAB_FLUSH)):
                    cap.grab()

                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[WARN] Mất frame. Đang reconnect...")
                    self.stream_ok = False
                    cap.release()
                    time.sleep(config.RECONNECT_DELAY)
                    break

                ts = time.time()
                with self.lock:
                    self.latest_frame = frame
                    self.latest_ts = ts
                    self.frame_buffer.append(BufferedFrame(frame=frame.copy(), ts=ts))

            try:
                cap.release()
            except Exception:
                pass
