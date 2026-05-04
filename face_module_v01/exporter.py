from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import face_module_v1.config as config


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _day_parts(ts: Optional[float] = None) -> tuple[str, str, str]:
    dt = datetime.fromtimestamp(ts or time.time()).astimezone()
    return dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")


def _sanitize_rel_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return path.replace("\\", "/")


class LocalExporter:
    def __init__(self):
        self.event_queue_dir = Path(config.EVENT_JSON_QUEUE_DIR)
        self.health_queue_dir = Path(config.HEALTH_JSON_QUEUE_DIR)
        self.snapshot_queue_dir = Path(config.SNAPSHOT_QUEUE_DIR)
        self.video_queue_dir = Path(config.VIDEO_QUEUE_DIR)

    def _copy_to_queue(self, src_path: Optional[str | Path], base_dir: Path, ts: Optional[float] = None) -> Optional[str]:
        if not src_path:
            return None
        src = Path(src_path)
        if not src.exists() or not src.is_file():
            return None
        y, m, d = _day_parts(ts)
        dst_dir = base_dir / y / m / d
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return _sanitize_rel_path(str(dst.relative_to(config.EXPORT_QUEUE_DIR)))

    def export_event(self, event: Any, extra_payload: Optional[Dict[str, Any]] = None, recorder_paths: Optional[Dict[str, str]] = None) -> Path:
        payload: Dict[str, Any] = {}
        if is_dataclass(event):
            payload.update(asdict(event))
        elif isinstance(event, dict):
            payload.update(event)
        else:
            raise TypeError("event must be dataclass or dict")

        start_ts = float(payload.get("start_ts", time.time()))
        last_seen_ts = float(payload.get("last_seen_ts", start_ts))
        event_id = str(payload.get("event_id"))
        person_name = str(payload.get("person_name", "Unknown"))
        event_type = str(payload.get("event_type", "person_detected"))
        best_confidence = float(payload.get("best_confidence", 0.0))
        extra = payload.get("extra") or {}

        snapshot_path = None
        video_path = None
        if recorder_paths:
            snapshot_path = recorder_paths.get("snapshot_path")
            video_path = recorder_paths.get("video_path")

        snapshot_queue_rel = self._copy_to_queue(snapshot_path, self.snapshot_queue_dir, start_ts)
        video_queue_rel = None
        should_queue_video = bool(config.UPLOAD_VIDEO_ENABLED)
        if config.UPLOAD_VIDEO_FOR_UNKNOWN_ONLY:
            should_queue_video = should_queue_video and person_name == "Unknown"
        if should_queue_video:
            video_queue_rel = self._copy_to_queue(video_path, self.video_queue_dir, start_ts)

        y, m, d = _day_parts(start_ts)
        dst_dir = self.event_queue_dir / y / m / d
        dst_dir.mkdir(parents=True, exist_ok=True)
        json_path = dst_dir / f"{event_id}.json"

        event_time_iso = datetime.fromtimestamp(start_ts).astimezone().isoformat(timespec="seconds")
        created_at = _now_iso()
        payload_out = {
            "event_id": event_id,
            "event_time": event_time_iso,
            "device_id": getattr(config, "DEVICE_ID", "pi5-01"),
            "camera_id": payload.get("camera_id") or getattr(config, "CAMERA_ID", "camera_01"),
            "event_type": event_type,
            "person_name": person_name,
            "is_known": person_name != "Unknown",
            "person_count": int(extra.get("person_count", 1)),
            "best_confidence": best_confidence,
            "stream_ok": bool(extra.get("stream_ok", True)),
            "snapshot_path": snapshot_queue_rel,
            "video_path": video_queue_rel,
            "event_start_ts": start_ts,
            "event_last_seen_ts": last_seen_ts,
            "event_duration_sec": round(max(0.0, last_seen_ts - start_ts), 3),
            "created_at": created_at,
            "extra": extra,
        }
        if extra_payload:
            payload_out.update(extra_payload)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload_out, f, ensure_ascii=False, indent=2)
        return json_path

    def export_health(self, payload: Dict[str, Any]) -> Path:
        ts = float(payload.get("health_ts", time.time()))
        y, m, d = _day_parts(ts)
        dst_dir = self.health_queue_dir / y / m / d
        dst_dir.mkdir(parents=True, exist_ok=True)
        health_id = str(payload.get("health_id"))
        json_path = dst_dir / f"{health_id}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return json_path
