from __future__ import annotations

import time
from typing import List, Dict, Any

import cv2

import face_module.config as config
from face_module.event_engine import EventEngine
from face_module.face_engine import FaceEngine
from face_module.recorder import EventRecorder
from face_module.stream_reader import StreamReader


def resize_keep_ratio(frame, target_width):
    if frame is None or target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h))


def scale_results(results: List[Dict[str, Any]], sx: float, sy: float) -> List[Dict[str, Any]]:
    scaled = []
    for item in results:
        x1, y1, x2, y2 = item["bbox"]
        scaled.append({
            "bbox": [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)],
            "name": item["name"],
            "distance": float(item["distance"]),
        })
    return scaled


def draw_results(display, cached_results: List[Dict[str, Any]]):
    for item in cached_results:
        x1, y1, x2, y2 = item["bbox"]
        name = item["name"]
        dist = item["distance"]

        color = (0, 255, 0) if name != config.UNKNOWN_LABEL else (0, 0, 255)
        label = f"{name} | {dist:.2f}" if config.DRAW_CONFIDENCE else name

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            display,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def log_event(prefix: str, event):
    if not config.PRINT_EVENT_LOG or event is None:
        return
    print(
        f"[{prefix}] id={event.event_id} type={event.event_type} "
        f"identity={event.identity} best_distance={event.best_distance:.3f} "
        f"snapshot={event.snapshot_path} clip={event.clip_path}"
    )


def main():
    engine = FaceEngine(
        det_size=(config.DETECTION_SIZE, config.DETECTION_SIZE),
        model_name=config.MODEL_NAME,
    )
    engine.build_database(config.FACE_DB_DIR, use_cache=True)

    stream = StreamReader(config.RTSP_URL)
    stream.start()

    event_engine = EventEngine()
    recorder = EventRecorder()

    frame_count = 0
    prev_time = time.time()
    fps = 0.0
    cached_results: List[Dict[str, Any]] = []

    try:
        while True:
            frame = stream.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            display = frame.copy()
            frame_count += 1

            if frame_count % config.DETECT_EVERY_N_FRAMES == 0:
                h, w = frame.shape[:2]
                infer_frame = resize_keep_ratio(frame, config.INFER_WIDTH)
                ih, iw = infer_frame.shape[:2]
                sx = w / iw
                sy = h / ih

                try:
                    results_small = engine.infer(infer_frame, threshold=config.RECOG_THRESHOLD)
                    cached_results = scale_results(results_small, sx, sy)
                except Exception as e:
                    print(f"[WARN] Lỗi infer: {e}")
                    cached_results = []

                event_updates = event_engine.update(cached_results, now_ts=time.time())
                if event_updates["event_started"] is not None:
                    event = event_updates["event_started"]
                    recorder.start_event(event, stream.get_pre_roll_frames(), frame)
                    log_event("EVENT_START", event)
                elif event_updates["event_updated"] is not None:
                    recorder.mark_seen()
                elif event_updates["event_closed"] is not None:
                    log_event("EVENT_CLOSE", event_updates["event_closed"])

            if recorder.current_event is not None:
                recorder.write_live_frame(frame)
                done = recorder.maybe_finalize()
                if done is not None:
                    log_event("CLIP_SAVED", done)

            draw_results(display, cached_results)

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = now

            if config.SHOW_FPS:
                cv2.putText(
                    display,
                    f"FPS: {fps:.1f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

                status = "STREAM OK" if stream.stream_ok else "RECONNECTING..."
                cv2.putText(
                    display,
                    status,
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255) if stream.stream_ok else (0, 0, 255),
                    2,
                )

                record_status = "REC ON" if recorder.current_event is not None else "REC OFF"
                cv2.putText(
                    display,
                    record_status,
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 0) if recorder.current_event is not None else (180, 180, 180),
                    2,
                )

            display = resize_keep_ratio(display, config.DISPLAY_WIDTH)
            cv2.imshow(config.WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        stream.stop()
        recorder.close()
        time.sleep(0.2)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
