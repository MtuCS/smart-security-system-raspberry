from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# =========================
# Nguồn camera
# =========================
RTSP_URL = "rtsp://admin:12345678a@192.168.1.245:554/Streaming/Channels/101"
RECONNECT_DELAY = 2.0
CAPTURE_GRAB_FLUSH = 2
CAPTURE_BUFFER_SIZE = 1
STREAM_STALE_SEC = 3.0

# =========================
# Dữ liệu khuôn mặt
# =========================
FACE_DB_DIR = str(BASE_DIR / "face_db")
CACHE_DIR = str(BASE_DIR / "cache")
EMBEDDING_CACHE_FILE = str(Path(CACHE_DIR) / "face_embeddings_cache.npz")
EMBEDDING_CACHE_META_FILE = str(Path(CACHE_DIR) / "face_embeddings_cache_meta.json")

# =========================
# Nhận diện
# =========================
MODEL_NAME = "buffalo_s"
DETECTION_SIZE = 320               # det_size cho insightface
INFER_WIDTH = 640                  # resize trước khi infer để giảm tải CPU
RECOG_THRESHOLD = 0.90             # khoảng cách L2, càng nhỏ càng chặt
DETECT_EVERY_N_FRAMES = 4
UNKNOWN_LABEL = "Unknown"

# =========================
# Hiển thị
# =========================
SHOW_FPS = True
WINDOW_NAME = "Face Detect + Recognize"
DISPLAY_WIDTH = 960
DRAW_CONFIDENCE = True

# =========================
# Event engine
# =========================
EVENT_COOLDOWN_SEC = 10.0          # cùng 1 người không bắn event liên tục
UNKNOWN_EVENT_COOLDOWN_SEC = 8.0
MIN_EVENT_GAP_SEC = 1.0            # tránh spam do cùng cảnh trong nhiều frame
EVENT_END_HOLD_SEC = 2.0           # giữ event mở thêm chút sau lần detect cuối

# =========================
# Record clip theo event
# =========================
ENABLE_RECORDING = True
RECORD_DIR = str(BASE_DIR / "records")
SNAPSHOT_DIR = str(Path(RECORD_DIR) / "snapshots")
CLIP_DIR = str(Path(RECORD_DIR) / "clips")
PRE_ROLL_SEC = 5                   # số giây lưu sẵn trong RAM
POST_ROLL_SEC = 8                  # số giây ghi thêm sau khi hết detect
RECORD_FPS = 12.0                  # fps ghi file; không nhất thiết bằng fps nguồn
VIDEO_CODEC = "mp4v"

# =========================
# Logging trạng thái
# =========================
PRINT_EVENT_LOG = True
PRINT_DEBUG_STATUS = False
