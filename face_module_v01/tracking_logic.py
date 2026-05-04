from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
BBox = List[int]


@dataclass
class Track:
    track_id: int
    bbox: BBox
    center: Point
    prev_center: Optional[Point]
    confidence: float
    age: int = 1
    missed: int = 0
    last_seen_ts: float = 0.0


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter + 1e-6))


class SimpleIoUTracker:
    """Lightweight tracking-by-detection for Raspberry Pi demos.

    It matches current person detections to existing tracks by IoU. This is not as
    strong as ByteTrack/DeepSORT, but it has no extra dependencies and is enough
    for the first IN/OUT + identity-mapping prototype.
    """

    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 12):
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[Track]:
        now = time.time()
        dets = [dict(d) for d in detections]
        unmatched_det_idxs = set(range(len(dets)))
        unmatched_track_ids = set(self.tracks.keys())

        candidates: List[Tuple[float, int, int]] = []
        for tid, tr in self.tracks.items():
            for di, det in enumerate(dets):
                iou = bbox_iou(tr.bbox, det["bbox"])
                if iou >= self.iou_threshold:
                    candidates.append((iou, tid, di))
        candidates.sort(reverse=True, key=lambda x: x[0])

        for _, tid, di in candidates:
            if tid not in unmatched_track_ids or di not in unmatched_det_idxs:
                continue
            det = dets[di]
            tr = self.tracks[tid]
            tr.prev_center = tr.center
            tr.bbox = [int(v) for v in det["bbox"]]
            tr.center = bbox_center(tr.bbox)
            tr.confidence = float(det.get("confidence", 0.0))
            tr.age += 1
            tr.missed = 0
            tr.last_seen_ts = now
            unmatched_track_ids.remove(tid)
            unmatched_det_idxs.remove(di)

        for tid in list(unmatched_track_ids):
            tr = self.tracks[tid]
            tr.missed += 1
            if tr.missed > self.max_missed:
                del self.tracks[tid]

        for di in sorted(unmatched_det_idxs):
            det = dets[di]
            bbox = [int(v) for v in det["bbox"]]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(
                track_id=tid,
                bbox=bbox,
                center=bbox_center(bbox),
                prev_center=None,
                confidence=float(det.get("confidence", 0.0)),
                last_seen_ts=now,
            )

        return [tr for tr in self.tracks.values() if tr.missed == 0]


def point_side(point: Point, p1: Point, p2: Point) -> float:
    x, y = point
    x1, y1 = p1
    x2, y2 = p2
    return float((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))


class LineCrossingDetector:
    def __init__(
        self,
        p1_ratio: Tuple[float, float],
        p2_ratio: Tuple[float, float],
        in_direction: str = "negative_to_positive",
        min_move_px: int = 10,
        cooldown_seconds: float = 1.0,
    ):
        self.p1_ratio = p1_ratio
        self.p2_ratio = p2_ratio
        self.in_direction = in_direction
        self.min_move_px = int(min_move_px)
        self.cooldown_seconds = float(cooldown_seconds)
        self.last_cross_ts: Dict[int, float] = {}

    def line_points(self, width: int, height: int) -> Tuple[Point, Point]:
        p1 = (int(self.p1_ratio[0] * width), int(self.p1_ratio[1] * height))
        p2 = (int(self.p2_ratio[0] * width), int(self.p2_ratio[1] * height))
        return p1, p2

    def check(self, track: Track, width: int, height: int) -> Optional[str]:
        if track.prev_center is None:
            return None
        now = time.time()
        if now - self.last_cross_ts.get(track.track_id, 0.0) < self.cooldown_seconds:
            return None

        dx = track.center[0] - track.prev_center[0]
        dy = track.center[1] - track.prev_center[1]
        if (dx * dx + dy * dy) ** 0.5 < self.min_move_px:
            return None

        p1, p2 = self.line_points(width, height)
        s_prev = point_side(track.prev_center, p1, p2)
        s_curr = point_side(track.center, p1, p2)
        if s_prev == 0 or s_curr == 0 or s_prev * s_curr > 0:
            return None

        neg_to_pos = s_prev < 0 < s_curr
        if self.in_direction == "negative_to_positive":
            event = "IN" if neg_to_pos else "OUT"
        else:
            event = "OUT" if neg_to_pos else "IN"
        self.last_cross_ts[track.track_id] = now
        return event

    def draw(self, frame, color=(0, 255, 255)):
        h, w = frame.shape[:2]
        p1, p2 = self.line_points(w, h)
        cv2.line(frame, p1, p2, color, 2)
        cv2.putText(frame, "virtual line", (p1[0], max(20, p1[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def rect_from_ratio(zone_ratio: Tuple[float, float, float, float], width: int, height: int) -> BBox:
    x1, y1, x2, y2 = zone_ratio
    return [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]


def point_in_rect(point: Point, rect: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_rect(frame, rect: BBox, label: str, color=(255, 0, 255)):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class TrackIdentityManager:
    def __init__(self, max_age_seconds: float = 30.0):
        self.track_to_identity: Dict[int, Dict[str, Any]] = {}
        self.max_age_seconds = float(max_age_seconds)

    def assign_identity(self, track_id: int, name: str, confidence: float):
        now = time.time()
        current = self.track_to_identity.get(track_id)
        if current is None:
            self.track_to_identity[track_id] = {
                "name": name,
                "confidence": float(confidence),
                "assigned_ts": now,
                "last_seen": now,
            }
            print(f"[ASSIGN] track_id={track_id} -> {name} ({confidence:.2f})")
            return

        if confidence > float(current.get("confidence", 0.0)) or current.get("name") == "Unknown":
            current["name"] = name
            current["confidence"] = float(confidence)
        current["last_seen"] = now

    def touch(self, track_id: int):
        if track_id in self.track_to_identity:
            self.track_to_identity[track_id]["last_seen"] = time.time()

    def get_identity(self, track_id: int) -> Dict[str, Any]:
        return self.track_to_identity.get(track_id, {"name": "Unknown", "confidence": 0.0})

    def remove(self, track_id: int):
        self.track_to_identity.pop(track_id, None)

    def cleanup(self):
        now = time.time()
        for tid in list(self.track_to_identity.keys()):
            if now - float(self.track_to_identity[tid].get("last_seen", 0.0)) > self.max_age_seconds:
                del self.track_to_identity[tid]


class OccupancySmoother:
    def __init__(self, window_size: int = 3, min_agree: Optional[int] = None):
        self.window_size = max(1, int(window_size))
        self.min_agree = int(min_agree or self.window_size)
        self.values: Deque[int] = deque(maxlen=self.window_size)
        self.confirmed_count: Optional[int] = None

    def update(self, raw_count: int) -> Tuple[Optional[int], bool]:
        self.values.append(int(raw_count))
        if len(self.values) < self.window_size:
            return self.confirmed_count, False
        value, freq = Counter(self.values).most_common(1)[0]
        if freq >= self.min_agree and value != self.confirmed_count:
            old = self.confirmed_count
            self.confirmed_count = value
            print(f"[OCCUPANCY] {old} -> {value}")
            return value, True
        return self.confirmed_count, False
