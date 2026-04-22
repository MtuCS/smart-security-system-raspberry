# from __future__ import annotations

# import shutil
# import threading
# import time
# from pathlib import Path
# from typing import Iterable

# import face_module_v1.config as config


# class BackgroundUploader:
#     def __init__(self):
#         self.enabled = bool(getattr(config, "UPLOAD_ENABLED", True))
#         self.poll_interval = float(getattr(config, "UPLOAD_POLL_INTERVAL_SECONDS", 5.0))
#         self.queue_dirs = {
#             Path(config.EVENT_JSON_QUEUE_DIR): Path(config.STORAGE_EVENTS_DIR),
#             Path(config.HEALTH_JSON_QUEUE_DIR): Path(config.STORAGE_HEALTH_DIR),
#             Path(config.SNAPSHOT_QUEUE_DIR): Path(config.STORAGE_SNAPSHOTS_DIR),
#         }
#         if bool(getattr(config, "UPLOAD_VIDEO_ENABLED", False)):
#             self.queue_dirs[Path(config.VIDEO_QUEUE_DIR)] = Path(config.STORAGE_VIDEOS_DIR)
#         self.sent_root = Path(config.EXPORT_SENT_DIR)
#         self.failed_root = Path(config.EXPORT_FAILED_DIR)
#         self._running = False
#         self._thread: threading.Thread | None = None

#     def start(self):
#         if not self.enabled or (self._thread and self._thread.is_alive()):
#             return
#         self._running = True
#         self._thread = threading.Thread(target=self._loop, name="background-uploader", daemon=True)
#         self._thread.start()

#     def stop(self):
#         self._running = False
#         if self._thread:
#             self._thread.join(timeout=2.0)

#     def _loop(self):
#         while self._running:
#             try:
#                 self.run_once()
#             except Exception as e:
#                 print(f"[WARN] uploader loop error: {e}")
#             time.sleep(self.poll_interval)

#     def run_once(self):
#         for queue_root, storage_root in self.queue_dirs.items():
#             for path in self._iter_files(queue_root):
#                 try:
#                     self._upload_file(path, queue_root, storage_root)
#                 except Exception as e:
#                     print(f"[WARN] upload failed for {path}: {e}")

#     def _iter_files(self, root: Path) -> Iterable[Path]:
#         if not root.exists():
#             return []
#         return sorted(p for p in root.rglob("*") if p.is_file())

#     def _upload_file(self, src: Path, queue_root: Path, storage_root: Path):
#         rel = src.relative_to(queue_root)
#         dst = storage_root / rel
#         dst.parent.mkdir(parents=True, exist_ok=True)
#         shutil.copy2(src, dst)

#         sent_path = self.sent_root / queue_root.name / rel
#         sent_path.parent.mkdir(parents=True, exist_ok=True)
#         if bool(getattr(config, "UPLOAD_DELETE_AFTER_SUCCESS", False)):
#             src.unlink(missing_ok=True)
#         else:
#             shutil.move(str(src), str(sent_path))
#         print(f"[UPLOAD OK] {src} -> {dst}")

import time
import shutil
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account

import face_module_v1.config as config


class GCSUploader:
    def __init__(self):
        credentials = service_account.Credentials.from_service_account_file(
            config.GCS_CREDENTIALS_FILE
        )
        self.client = storage.Client(credentials=credentials)
        self.bucket = self.client.bucket(config.GCS_BUCKET_NAME)

    def upload_file(self, local_path: Path, remote_path: str):
        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(str(local_path))


def get_gcs_path(file_path: Path):
    parts = file_path.parts

    if "events-json" in parts:
        prefix = config.GCS_EVENTS_PREFIX
    elif "health-json" in parts:
        prefix = config.GCS_HEALTH_PREFIX
    elif "snapshots" in parts:
        prefix = config.GCS_SNAPSHOTS_PREFIX
    elif "videos" in parts:
        prefix = config.GCS_VIDEOS_PREFIX
    else:
        return None

    # lấy phần path phía sau queue/
    idx = parts.index("queue")
    relative = Path(*parts[idx + 1:])

    return f"{prefix}/{relative}"


def move_file(src: Path, dst_root: Path):
    relative = src.relative_to(config.EXPORT_QUEUE_DIR)
    dst = dst_root / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def run_uploader():
    uploader = GCSUploader()
    retry_count = {}

    while True:
        try:
            files = list(config.EXPORT_QUEUE_DIR.rglob("*.*"))

            for file_path in files:
                if not file_path.is_file():
                    continue

                gcs_path = get_gcs_path(file_path)
                if not gcs_path:
                    continue

                try:
                    uploader.upload_file(file_path, gcs_path)
                    print(f"[UPLOAD OK] {file_path} -> {gcs_path}")

                    move_file(file_path, config.EXPORT_SENT_DIR)
                    retry_count.pop(file_path, None)

                except Exception as e:
                    print(f"[UPLOAD FAIL] {file_path} | {e}")

                    retry_count[file_path] = retry_count.get(file_path, 0) + 1

                    if retry_count[file_path] >= config.MAX_UPLOAD_RETRIES:
                        print(f"[MOVE FAILED] {file_path}")
                        move_file(file_path, config.EXPORT_FAILED_DIR)
                        retry_count.pop(file_path, None)

            time.sleep(config.UPLOAD_POLL_INTERVAL_SECONDS)

        except Exception as e:
            print(f"[UPLOADER ERROR] {e}")
            time.sleep(5)


if __name__ == "__main__":
    run_uploader()