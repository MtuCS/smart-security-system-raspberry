from __future__ import annotations

import time

import cv2

import face_module_v1.config as config
from face_module_v1.cameras.camera_config import CAMERAS
from face_module_v1.cameras.door_camera_worker import DoorCameraWorker
from face_module_v1.cameras.room_camera_worker import RoomCameraWorker


def build_worker(camera_cfg: dict):
    role = camera_cfg.get("role")
    if role == "face_access":
        return DoorCameraWorker(camera_cfg)
    if role == "occupancy_monitoring":
        return RoomCameraWorker(camera_cfg)
    raise ValueError(f"Unknown camera role: {role}")


def main():
    workers = [build_worker(cfg) for cfg in CAMERAS.values()]

    print("[INFO] Starting multi-camera system...")
    for w in workers:
        print(f"[INFO] Start {w.camera_id}: {w.rtsp_url}")
        w.start()

    try:
        while True:
            # OpenCV window events should be pumped from main process too.
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received.")
    finally:
        print("[INFO] Stopping workers...")
        for w in workers:
            w.stop()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
