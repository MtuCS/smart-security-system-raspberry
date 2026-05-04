from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVENTS_DIR = DATA_DIR / "events"
CACHE_DIR = DATA_DIR / "cache"
FACE_DB_DIR = str(BASE_DIR / "face_db")
RUNS_DIR = BASE_DIR / "runs"
TEMP_DIR = BASE_DIR / "temp"

# Export / upload pipeline
EXPORT_DIR = DATA_DIR / "export"
EXPORT_QUEUE_DIR = EXPORT_DIR / "queue"
EXPORT_SENT_DIR = EXPORT_DIR / "sent"
EXPORT_FAILED_DIR = EXPORT_DIR / "failed"
EVENT_JSON_QUEUE_DIR = EXPORT_QUEUE_DIR / "events-json"
HEALTH_JSON_QUEUE_DIR = EXPORT_QUEUE_DIR / "health-json"
SNAPSHOT_QUEUE_DIR = EXPORT_QUEUE_DIR / "snapshots"
VIDEO_QUEUE_DIR = EXPORT_QUEUE_DIR / "videos"

# Storage target for uploader.
# Giai đoạn đầu dùng local filesystem target để test end-to-end.
# Sau này có thể thay bằng mount cloud storage, rclone mount, S3 sync, ...
STORAGE_ROOT_DIR = DATA_DIR / "storage_mirror"
STORAGE_EVENTS_DIR = STORAGE_ROOT_DIR / "raw" / "events-json"
STORAGE_HEALTH_DIR = STORAGE_ROOT_DIR / "raw" / "health-json"
STORAGE_SNAPSHOTS_DIR = STORAGE_ROOT_DIR / "raw" / "snapshots"
STORAGE_VIDEOS_DIR = STORAGE_ROOT_DIR / "raw" / "videos"

# ===== GCS CONFIG =====
UPLOAD_BACKEND = "gcs"

GCS_BUCKET_NAME = "ai-camera-storage" 
GCS_BASE_PREFIX = "raw"

GCS_EVENTS_PREFIX = f"{GCS_BASE_PREFIX}/events-json"
GCS_HEALTH_PREFIX = f"{GCS_BASE_PREFIX}/health-json"
GCS_SNAPSHOTS_PREFIX = f"{GCS_BASE_PREFIX}/snapshots"
GCS_VIDEOS_PREFIX = f"{GCS_BASE_PREFIX}/videos"

GCS_CREDENTIALS_FILE = Path("/home/cs18tnt/credentials/ai-camera-cloud-7ca5c2802845.json")

DEVICE_ID = "pi5-01"
CAMERA_ID = "camera_01"
RTSP_URL = "rtsp://admin:12345678a@192.168.1.245:554/Streaming/Channels/101"
WINDOW_NAME = "Person + Face Security Pipeline"
SHOW_FPS = True
DISPLAY_WIDTH = 960
DRAW_CONFIDENCE = True
RECONNECT_DELAY = 2.0
FRAME_GRAB_FLUSH_COUNT = 2

# Person detection
PERSON_DETECTOR = "hog"
PERSON_DETECT_EVERY_N_FRAMES = 3
PERSON_CONFIDENCE = 0.4
PERSON_INFER_WIDTH = 640
PERSON_NMS_THRESHOLD = 0.35
PERSON_MIN_AREA_RATIO = 0.015
PERSON_PAD_RATIO = 0.08

# Face recognition
MODEL_NAME = "buffalo_s"
ENABLE_FACE_RECOGNITION = True
FACE_RECOG_EVERY_N_FRAMES = 2
MAX_FACE_SIZE = 320
RECOG_THRESHOLD = 0.9  # nhỏ hơn thì chặt hơn
FACE_MIN_SIZE = 36

# Event logic
EVENT_HOLD_SECONDS = 2.5
EVENT_COOLDOWN_SECONDS = 8.0
UNKNOWN_EVENT_COOLDOWN_SECONDS = 5.0
PREFER_KNOWN_FACE = True
EVENT_START_CONFIRM_FRAMES = 2
EVENT_IDENTITY_CONFIRM_FRAMES = 3
EVENT_END_ABSENCE_SECONDS = 2.0
EVENT_START_COOLDOWN_SECONDS = 1.5
EVENT_MIN_UPDATE_INTERVAL_SECONDS = 0.8

# Recording
ENABLE_RECORDING = True
EVENT_PRE_ROLL_SECONDS = 5
EVENT_POST_ROLL_SECONDS = 8
EVENT_VIDEO_FPS = 12
SNAPSHOT_JPEG_QUALITY = 90
CODEC = "mp4v"
UPLOAD_VIDEO_FOR_UNKNOWN_ONLY = False
UPLOAD_VIDEO_ENABLED = False  # để false ở phase đầu cho an toàn

# Health export / uploader
ENABLE_HEALTH_EXPORT = True
HEALTH_EXPORT_INTERVAL_SECONDS = 60.0
UPLOAD_ENABLED = True
UPLOAD_POLL_INTERVAL_SECONDS = 5.0
UPLOAD_COPY_FILES = True
UPLOAD_DELETE_AFTER_SUCCESS = False

# Face embedding cache
ENABLE_FACE_DB_CACHE = True
FACE_DB_CACHE_FILE = CACHE_DIR / "face_embeddings_cache.pkl"
FACE_DB_CACHE_PATH = FACE_DB_CACHE_FILE
FACE_MODEL_NAME = MODEL_NAME

# Logging / retention
MAX_UPLOAD_RETRIES = 20


for _dir in [
    DATA_DIR,
    EVENTS_DIR,
    CACHE_DIR,
    RUNS_DIR,
    TEMP_DIR,
    EXPORT_DIR,
    EXPORT_QUEUE_DIR,
    EXPORT_SENT_DIR,
    EXPORT_FAILED_DIR,
    EVENT_JSON_QUEUE_DIR,
    HEALTH_JSON_QUEUE_DIR,
    SNAPSHOT_QUEUE_DIR,
    VIDEO_QUEUE_DIR,
    STORAGE_EVENTS_DIR,
    STORAGE_HEALTH_DIR,
    STORAGE_SNAPSHOTS_DIR,
    STORAGE_VIDEOS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Tracking + occupancy + virtual line prototype
# ------------------------------------------------------------
ENABLE_TRACKING = True
TRACK_IOU_THRESHOLD = 0.25
TRACK_MAX_MISSED = 12
TRACK_IDENTITY_MAX_AGE_SECONDS = 30.0

# Smoothing s? ngu?i trong ph�ng
OCCUPANCY_SMOOTHING_WINDOW = 3
OCCUPANCY_SMOOTHING_MIN_AGREE = 3

# V�ng c?a d? uu ti�n nh?n di?n m?t v� g�n identity cho track_id.
# Format: (x1, y1, x2, y2) theo t? l? frame. C?n ch?nh theo g�c camera th?c t?.
DOOR_ZONE_RATIO = (0.25, 0.00, 0.75, 0.45)
DRAW_DOOR_ZONE = True

# Line ?o d�ng d? x�c d?nh IN/OUT tr�n cam t?ng qu�t.
# Format: (x_ratio, y_ratio). C?n ch?nh l?i sau khi xem frame th?c t?.
VIRTUAL_LINE_P1_RATIO = (0.20, 0.50)
VIRTUAL_LINE_P2_RATIO = (0.85, 0.50)
# N?u IN/OUT b? ngu?c, d?i th�nh "positive_to_negative".
VIRTUAL_LINE_IN_DIRECTION = "negative_to_positive"
VIRTUAL_LINE_MIN_MOVE_PX = 10
VIRTUAL_LINE_COOLDOWN_SECONDS = 1.0
DRAW_VIRTUAL_LINE = True

# Gi?i h?n s? track ch?y face recognition trong door zone m?i chu k? d? nh? Pi 5.
TRACK_RECOGNIZE_MAX_TRACKS = 1
