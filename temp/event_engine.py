from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import face_module_v1.config as config


@dataclass
class ActiveEvent:
    event_id: str
    event_type: str
    person_name: str
    start_ts: float
    last_seen_ts: float
    best_frame: Any = None
    best_face_result: Optional[Dict[str, Any]] = None
    best_person_result: Optional[Dict[str, Any]] = None
    snapshots_saved: bool = False
    recording_started: bool = False


class EventEngine:
    def __init__(self):
        self.active_event: Optional[ActiveEvent] = None
        self.last_trigger_by_key: Dict[str, float] = {}

    def _cooldown_for(self, person_name: str) -> float:
        return config.UNKNOWN_EVENT_COOLDOWN_SECONDS if person_name == "Unknown" else config.EVENT_COOLDOWN_SECONDS

    def _can_trigger(self, person_name: str, now: float) -> bool:
        key = f"face:{person_name}"
        last_ts = self.last_trigger_by_key.get(key, 0.0)
        return (now - last_ts) >= self._cooldown_for(person_name)

    def update(
        self,
        now: float,
        frame,
        person_result: Optional[Dict[str, Any]],
        face_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "triggered": False,
            "ended": False,
            "active_event": self.active_event,
        }

        detected = person_result is not None
        person_name = "Unknown"
        event_type = "person_detected"
        if face_result is not None:
            person_name = face_result.get("name", "Unknown")
            event_type = "known_face_detected" if person_name != "Unknown" else "unknown_face_detected"

        if detected:
            if self.active_event is None:
                if self._can_trigger(person_name, now):
                    event = ActiveEvent(
                        event_id=uuid.uuid4().hex[:12],
                        event_type=event_type,
                        person_name=person_name,
                        start_ts=now,
                        last_seen_ts=now,
                        best_frame=frame.copy(),
                        best_face_result=face_result,
                        best_person_result=person_result,
                    )
                    self.active_event = event
                    self.last_trigger_by_key[f"face:{person_name}"] = now
                    response["triggered"] = True
                    response["active_event"] = event
                    return response
            else:
                self.active_event.last_seen_ts = now
                if self._should_upgrade(self.active_event, face_result):
                    self.active_event.person_name = face_result.get("name", "Unknown")
                    self.active_event.event_type = (
                        "known_face_detected" if self.active_event.person_name != "Unknown" else "unknown_face_detected"
                    )
                    self.active_event.best_face_result = face_result
                    self.active_event.best_person_result = person_result
                    self.active_event.best_frame = frame.copy()
                elif self.active_event.best_person_result is None and person_result is not None:
                    self.active_event.best_person_result = person_result
                    self.active_event.best_frame = frame.copy()

                response["active_event"] = self.active_event
                return response

        if self.active_event is not None:
            idle_time = now - self.active_event.last_seen_ts
            if idle_time >= config.EVENT_HOLD_SECONDS:
                response["ended"] = True
                response["active_event"] = self.active_event
                self.active_event = None
                return response

        response["active_event"] = self.active_event
        return response

    def _should_upgrade(self, active: ActiveEvent, face_result: Optional[Dict[str, Any]]) -> bool:
        if face_result is None:
            return False
        if not config.PREFER_KNOWN_FACE:
            return False
        new_name = face_result.get("name", "Unknown")
        old_name = active.person_name
        if old_name == "Unknown" and new_name != "Unknown":
            return True
        if active.best_face_result is None:
            return True
        if new_name == old_name and face_result.get("distance", 999.0) < active.best_face_result.get("distance", 999.0):
            return True
        return False
