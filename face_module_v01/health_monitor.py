from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import face_module_v1.config as config
from face_module_v1.exporter import LocalExporter


class HealthMonitor:
    def __init__(self, exporter: LocalExporter, camera_id: str, device_id: str):
        self.exporter = exporter
        self.camera_id = camera_id
        self.device_id = device_id
        self.interval_seconds = float(getattr(config, "HEALTH_EXPORT_INTERVAL_SECONDS", 60.0))
        self.enabled = bool(getattr(config, "ENABLE_HEALTH_EXPORT", True))
        self.last_export_ts = 0.0
        self.last_fps = 0.0
        self.stream_ok = False

    def update_runtime(self, *, fps: float, stream_ok: bool):
        self.last_fps = float(fps)
        self.stream_ok = bool(stream_ok)

    def maybe_export(self):
        if not self.enabled:
            return None
        now = time.time()
        if (now - self.last_export_ts) < self.interval_seconds:
            return None
        payload = {
            "health_id": f"health_{uuid.uuid4().hex[:12]}",
            "health_time": datetime.fromtimestamp(now).astimezone().isoformat(timespec="seconds"),
            "health_ts": now,
            "device_id": self.device_id,
            "camera_id": self.camera_id,
            "fps": round(self.last_fps, 3),
            "cpu_percent": round(_read_cpu_percent(), 2),
            "ram_percent": round(_read_ram_percent(), 2),
            "temperature_c": round(_read_temperature_c(), 2),
            "stream_ok": self.stream_ok,
            "created_at": datetime.fromtimestamp(now).astimezone().isoformat(timespec="seconds"),
        }
        self.last_export_ts = now
        return self.exporter.export_health(payload)


def _read_temperature_c() -> float:
    thermal_zone = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return float(thermal_zone.read_text().strip()) / 1000.0
    except Exception:
        return 0.0


def _read_ram_percent() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            info = {}
            for line in f:
                key, value = line.split(":", 1)
                info[key.strip()] = int(value.strip().split()[0])
        total = float(info.get("MemTotal", 0))
        available = float(info.get("MemAvailable", 0))
        if total <= 0:
            return 0.0
        used = total - available
        return (used / total) * 100.0
    except Exception:
        return 0.0


def _read_cpu_percent(sample_seconds: float = 0.05) -> float:
    try:
        t1_idle, t1_total = _read_cpu_times()
        time.sleep(sample_seconds)
        t2_idle, t2_total = _read_cpu_times()
        total_delta = max(1.0, t2_total - t1_total)
        idle_delta = max(0.0, t2_idle - t1_idle)
        return (1.0 - idle_delta / total_delta) * 100.0
    except Exception:
        return 0.0


def _read_cpu_times() -> tuple[float, float]:
    with open("/proc/stat", "r", encoding="utf-8") as f:
        cpu = f.readline().strip().split()[1:]
    values = [float(x) for x in cpu]
    idle = values[3] + values[4]
    total = sum(values)
    return idle, total
