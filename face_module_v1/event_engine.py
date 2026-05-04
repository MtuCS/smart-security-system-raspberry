from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ActiveEvent:
    event_id: str
    camera_id: str
    event_type: str
    person_name: str
    start_ts: float
    last_seen_ts: float
    last_update_ts: float
    best_confidence: float = 0.0
    best_frame: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


class EventEngine:
    def __init__(
        self,
        camera_id: str = "camera_01",
        start_confirm_frames: int = 2,
        identity_confirm_frames: int = 3,
        end_absence_seconds: float = 2.0,
        start_cooldown_seconds: float = 1.5,
        min_update_interval_seconds: float = 0.8,
    ):
        self.camera_id = camera_id
        self.start_confirm_frames = max(1, int(start_confirm_frames))
        self.identity_confirm_frames = max(1, int(identity_confirm_frames))
        self.end_absence_seconds = float(end_absence_seconds)
        self.start_cooldown_seconds = float(start_cooldown_seconds)
        self.min_update_interval_seconds = float(min_update_interval_seconds)

        self.active_event: Optional[ActiveEvent] = None
        self.last_start_ts: float = 0.0
        self.presence_counter: int = 0
        self.identity_counter: Dict[str, int] = defaultdict(int)
        self._overlay_text: str = "EVENT: none"

    def current_overlay_text(self) -> str:
        return self._overlay_text

    def _event_type_for(self, person_name: str) -> str:
        if person_name == "Unknown":
            return "person_detected"
        return "known_face_detected"

    def _reset_pending(self):
        self.presence_counter = 0
        self.identity_counter.clear()

    def process(self, person_present: bool, person_name: str, confidence: float, frame, recorder, extra: Optional[Dict[str, Any]] = None):
        now = time.time()
        extra = extra or {}

        if person_present:
            self.presence_counter += 1
            self.identity_counter[person_name] += 1
        else:
            self._reset_pending()

        if self.active_event is None:
            if person_present and self.presence_counter >= self.start_confirm_frames and (now - self.last_start_ts) >= self.start_cooldown_seconds:
                stable_name = person_name
                event = ActiveEvent(
                    event_id=uuid.uuid4().hex[:12],
                    camera_id=self.camera_id,
                    event_type=self._event_type_for(stable_name),
                    person_name=stable_name,
                    start_ts=now,
                    last_seen_ts=now,
                    last_update_ts=now,
                    best_confidence=confidence,
                    best_frame=frame.copy() if frame is not None else None,
                    extra=dict(extra),
                )
                self.active_event = event
                self.last_start_ts = now
                saved = recorder.start_event(event, event.best_frame if event.best_frame is not None else frame)
                self._overlay_text = f"EVENT: {event.event_type} | {event.person_name}"
                print(f"[EVENT START] {event.event_type}: {event.person_name} | {saved}")
                self._reset_pending()
            return

        event = self.active_event
        if person_present:
            event.last_seen_ts = now
            if confidence >= event.best_confidence and frame is not None:
                event.best_confidence = confidence
                event.best_frame = frame.copy()
                event.extra = dict(extra)

            should_upgrade = (
                person_name != "Unknown"
                and person_name != event.person_name
                and self.identity_counter[person_name] >= self.identity_confirm_frames
                and (now - event.last_update_ts) >= self.min_update_interval_seconds
            )
            if should_upgrade:
                old_type = event.event_type
                old_name = event.person_name
                event.person_name = person_name
                event.event_type = self._event_type_for(person_name)
                event.last_update_ts = now
                recorder.mark_update(event)
                self._overlay_text = f"EVENT: {event.event_type} | {event.person_name}"
                print(f"[EVENT UPDATE] {old_type}:{old_name} -> {event.event_type}:{event.person_name}")
                self.identity_counter.clear()
            else:
                self._overlay_text = f"EVENT: {event.event_type} | {event.person_name}"
        else:
            self._overlay_text = f"EVENT: {event.event_type} | {event.person_name}"

    def flush_if_timeout(self, recorder):
        if self.active_event is None:
            return
        now = time.time()
        if (now - self.active_event.last_seen_ts) >= self.end_absence_seconds:
            print(f"[EVENT END] {self.active_event.event_type}: {self.active_event.person_name}")
            recorder.mark_update(self.active_event)
            self.active_event = None
            self._overlay_text = "EVENT: none"
            self._reset_pending()
