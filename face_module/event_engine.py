from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import face_module.config as config


@dataclass
class Event:
    event_id: str
    event_type: str
    identity: str
    started_at: float
    last_seen_at: float
    best_distance: float = 999.0
    detections: List[Dict[str, Any]] = field(default_factory=list)
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None
    is_open: bool = True


class EventEngine:
    def __init__(self):
        self.active_event: Optional[Event] = None
        self.last_emit_by_key: Dict[str, float] = {}
        self.last_event_end_ts: float = 0.0
        self._seq = 0

    def _next_event_id(self) -> str:
        self._seq += 1
        return f"evt_{int(time.time())}_{self._seq:05d}"

    def _cooldown_for(self, identity: str) -> float:
        if identity == config.UNKNOWN_LABEL:
            return config.UNKNOWN_EVENT_COOLDOWN_SEC
        return config.EVENT_COOLDOWN_SEC

    def _event_key(self, identity: str) -> str:
        if identity == config.UNKNOWN_LABEL:
            return "unknown_face"
        return f"known:{identity}"

    def update(self, detections: List[Dict[str, Any]], now_ts: Optional[float] = None) -> Dict[str, Any]:
        now_ts = now_ts or time.time()
        output = {
            "event_started": None,
            "event_updated": None,
            "event_closed": None,
        }

        best_det = None
        if detections:
            best_det = min(detections, key=lambda d: d.get("distance", 999.0))

        if best_det is not None:
            identity = best_det["name"]
            key = self._event_key(identity)

            if self.active_event and self.active_event.identity == identity and self.active_event.is_open:
                self.active_event.last_seen_at = now_ts
                self.active_event.detections = detections
                self.active_event.best_distance = min(self.active_event.best_distance, best_det.get("distance", 999.0))
                output["event_updated"] = self.active_event
                return output

            last_emit = self.last_emit_by_key.get(key, 0.0)
            cooldown = self._cooldown_for(identity)
            if (now_ts - last_emit) >= cooldown and (now_ts - self.last_event_end_ts) >= config.MIN_EVENT_GAP_SEC:
                event = Event(
                    event_id=self._next_event_id(),
                    event_type="known_face_detected" if identity != config.UNKNOWN_LABEL else "unknown_face_detected",
                    identity=identity,
                    started_at=now_ts,
                    last_seen_at=now_ts,
                    best_distance=best_det.get("distance", 999.0),
                    detections=detections,
                )
                self.active_event = event
                self.last_emit_by_key[key] = now_ts
                output["event_started"] = event
                return output

        if self.active_event and self.active_event.is_open:
            if now_ts - self.active_event.last_seen_at >= config.EVENT_END_HOLD_SEC:
                self.active_event.is_open = False
                self.last_event_end_ts = now_ts
                output["event_closed"] = self.active_event
                self.active_event = None

        return output
