from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2


def bbox_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def bbox_iou(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = bbox_area(a) + bbox_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def center_of(b) -> Tuple[int, int]:
    return (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2))


@dataclass
class Track:
    track_id: int
    bbox: List[int]
    confidence: float = 0.0
    missed: int = 0
    created_ts: float = field(default_factory=time.time)
    updated_ts: float = field(default_factory=time.time)
    prev_center: Optional[Tuple[int, int]] = None

    @property
    def center(self) -> Tuple[int, int]:
        return center_of(self.bbox)


class SimpleIoUTracker:
    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 12):
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Dict]) -> List[Track]:
        unmatched_track_ids = set(self.tracks.keys())
        unmatched_det_ids = set(range(len(detections)))
        pairs = []
        for tid, tr in self.tracks.items():
            for didx, det in enumerate(detections):
                pairs.append((bbox_iou(tr.bbox, det["bbox"]), tid, didx))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for score, tid, didx in pairs:
            if score < self.iou_threshold:
                break
            if tid not in unmatched_track_ids or didx not in unmatched_det_ids:
                continue
            tr = self.tracks[tid]
            tr.prev_center = tr.center
            tr.bbox = list(map(int, detections[didx]["bbox"]))
            tr.confidence = float(detections[didx].get("confidence", 0.0))
            tr.missed = 0
            tr.updated_ts = time.time()
            unmatched_track_ids.remove(tid)
            unmatched_det_ids.remove(didx)

        for tid in list(unmatched_track_ids):
            self.tracks[tid].missed += 1
            if self.tracks[tid].missed > self.max_missed:
                del self.tracks[tid]

        for didx in unmatched_det_ids:
            det = detections[didx]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(
                track_id=tid,
                bbox=list(map(int, det["bbox"])),
                confidence=float(det.get("confidence", 0.0)),
            )

        return list(self.tracks.values())


class OccupancySmoother:
    def __init__(self, window_size: int = 3, min_agree: int = 2):
        self.values = deque(maxlen=max(1, int(window_size)))
        self.min_agree = max(1, int(min_agree))
        self.current: Optional[int] = None

    def update(self, value: int):
        self.values.append(int(value))
        counter = Counter(self.values)
        best_value, agree = counter.most_common(1)[0]
        changed = False
        if agree >= self.min_agree and best_value != self.current:
            self.current = best_value
            changed = True
        return self.current if self.current is not None else value, changed


class LineCrossingDetector:
    def __init__(self, p1_ratio, p2_ratio, in_direction="negative_to_positive", min_move_px=10, cooldown_seconds=1.0):
        self.p1_ratio = p1_ratio
        self.p2_ratio = p2_ratio
        self.in_direction = in_direction
        self.min_move_px = float(min_move_px)
        self.cooldown_seconds = float(cooldown_seconds)
        self.last_cross_ts: Dict[int, float] = {}
        self._last_line = None

    def _line_points(self, w, h):
        p1 = (int(self.p1_ratio[0] * w), int(self.p1_ratio[1] * h))
        p2 = (int(self.p2_ratio[0] * w), int(self.p2_ratio[1] * h))
        self._last_line = (p1, p2)
        return p1, p2

    def _side(self, p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    def check(self, track: Track, w: int, h: int) -> Optional[str]:
        if track.prev_center is None:
            return None
        p1, p2 = self._line_points(w, h)
        old = track.prev_center
        new = track.center
        move = ((new[0] - old[0]) ** 2 + (new[1] - old[1]) ** 2) ** 0.5
        if move < self.min_move_px:
            return None
        s1 = self._side(old, p1, p2)
        s2 = self._side(new, p1, p2)
        if s1 == 0 or s2 == 0 or (s1 > 0) == (s2 > 0):
            return None
        now = time.time()
        if now - self.last_cross_ts.get(track.track_id, 0) < self.cooldown_seconds:
            return None
        self.last_cross_ts[track.track_id] = now
        positive = s1 < 0 and s2 > 0
        if self.in_direction == "negative_to_positive":
            return "IN" if positive else "OUT"
        return "OUT" if positive else "IN"

    def draw(self, frame):
        h, w = frame.shape[:2]
        p1, p2 = self._line_points(w, h)
        cv2.line(frame, p1, p2, (0, 255, 255), 2)
        cv2.putText(frame, "virtual line", (p1[0], max(20, p1[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def resize_keep_ratio(frame, target_width: Optional[int]):
    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))
