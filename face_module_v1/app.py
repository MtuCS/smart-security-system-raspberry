from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import cv2

import face_module_v1.config as config
from face_module_v1.event_engine import EventEngine
from face_module_v1.exporter import LocalExporter
from face_module_v1.health_monitor import HealthMonitor
from face_module_v1.face_engine import FaceEngine
from face_module_v1.person_detector import PersonDetector
from face_module_v1.recorder import EventRecorder
from face_module_v1.stream_reader import StreamReader
from face_module_v1.uploader import BackgroundUploader


def resize_keep_ratio(frame, target_width: Optional[int]):
    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h))


def pick_primary_person(persons: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not persons:
        return None
    return max(
        persons,
        key=lambda p: max(1, (p["bbox"][2] - p["bbox"][0])) * max(1, (p["bbox"][3] - p["bbox"][1])),
    )


def clip_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def crop_upper_body_for_face(frame, person_box, top_ratio: float = 0.8, side_padding: float = 0.13):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = person_box
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * side_padding)
    crop = [x1 - pad_x, y1, x2 + pad_x, y1 + int(bh * top_ratio)]
    crop = clip_box(crop, w, h)
    if crop is None:
        return None, None

    cx1, cy1, cx2, cy2 = crop
    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return None, None
    return roi, crop


def select_best_face(face_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not face_results:
        return None
    knowns = [f for f in face_results if f.get("name") != "Unknown"]
    if knowns:
        return min(knowns, key=lambda f: f.get("distance", 999.0))
    return min(face_results, key=lambda f: f.get("distance", 999.0))


def main():
    camera_id = getattr(config, "CAMERA_ID", "camera_01")

    stream = StreamReader(rtsp_url=config.RTSP_URL)
    face_engine = FaceEngine(
        det_size=(getattr(config, "MAX_FACE_SIZE", 320), getattr(config, "MAX_FACE_SIZE", 320)),
        model_name=getattr(config, "FACE_MODEL_NAME", "buffalo_l"),
        cache_path=getattr(config, "FACE_DB_CACHE_PATH", None),
    )
    face_engine.build_database(config.FACE_DB_DIR)

    person_detector = PersonDetector()
    exporter = LocalExporter()
    uploader = BackgroundUploader()
    health_monitor = HealthMonitor(
        exporter=exporter,
        camera_id=camera_id,
        device_id=getattr(config, "DEVICE_ID", "pi5-01"),
    )

    recorder = EventRecorder(
        base_dir=getattr(config, "EVENTS_DIR", "data/events"),
        fps=getattr(config, "EVENT_VIDEO_FPS", 12),
        pre_roll_seconds=getattr(config, "EVENT_PRE_ROLL_SECONDS", 5.0),
        post_roll_seconds=getattr(config, "EVENT_POST_ROLL_SECONDS", 6.0),
    )
    event_engine = EventEngine(
        camera_id=camera_id,
        start_confirm_frames=getattr(config, "EVENT_START_CONFIRM_FRAMES", 2),
        identity_confirm_frames=getattr(config, "EVENT_IDENTITY_CONFIRM_FRAMES", 3),
        end_absence_seconds=getattr(config, "EVENT_END_ABSENCE_SECONDS", 2.0),
        start_cooldown_seconds=getattr(config, "EVENT_START_COOLDOWN_SECONDS", 1.5),
        min_update_interval_seconds=getattr(config, "EVENT_MIN_UPDATE_INTERVAL_SECONDS", 0.8),
        exporter=exporter,
    )

    uploader.start()
    stream.start()

    frame_count = 0
    prev_time = time.time()
    fps = 0.0

    last_persons: List[Dict[str, Any]] = []
    last_face_results: List[Dict[str, Any]] = []

    person_every = int(getattr(config, "PERSON_DETECT_EVERY_N_FRAMES", 3))
    face_every = int(getattr(config, "FACE_DETECT_EVERY_N_FRAMES", 2))
    recog_threshold = float(getattr(config, "RECOG_THRESHOLD", 0.9))

    try:
        while True:
            frame = stream.read()
            if frame is None:
                event_engine.flush_if_timeout(recorder=recorder)
                recorder.update(None)
                health_monitor.update_runtime(fps=fps, stream_ok=stream.stream_ok)
                health_monitor.maybe_export()
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                time.sleep(0.01)
                continue

            frame_count += 1
            recorder.update(frame)
            display = frame.copy()

            if frame_count % max(1, person_every) == 0:
                try:
                    last_persons = person_detector.detect(frame)
                except Exception as e:
                    print(f"[WARN] person detect error: {e}")
                    last_persons = []

            primary_person = pick_primary_person(last_persons)
            current_name = "Unknown"
            current_conf = 0.0

            if primary_person is not None and frame_count % max(1, face_every) == 0:
                roi, roi_box = crop_upper_body_for_face(frame, primary_person["bbox"])
                if roi is not None:
                    try:
                        results = face_engine.infer(roi, threshold=recog_threshold)
                        rx1, ry1, _, _ = roi_box
                        scaled_results = []
                        for item in results:
                            x1, y1, x2, y2 = item["bbox"]
                            scaled_results.append(
                                {
                                    "bbox": [x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1],
                                    "name": item["name"],
                                    "distance": item["distance"],
                                }
                            )
                        last_face_results = scaled_results
                    except Exception as e:
                        print(f"[WARN] face infer error: {e}")
                        last_face_results = []
                else:
                    last_face_results = []
            elif primary_person is None:
                last_face_results = []

            best_face = select_best_face(last_face_results)
            if best_face is not None:
                current_name = best_face.get("name", "Unknown")
                current_conf = max(0.0, 1.0 - float(best_face.get("distance", 1.0)))
            elif primary_person is not None:
                current_name = "Unknown"
                current_conf = float(primary_person.get("confidence", 0.0))

            event_engine.process(
                person_present=primary_person is not None,
                person_name=current_name,
                confidence=current_conf,
                frame=frame,
                recorder=recorder,
                extra={
                    "person_count": len(last_persons),
                    "stream_ok": stream.stream_ok,
                    "camera_id": camera_id,
                },
            )
            event_engine.flush_if_timeout(recorder=recorder)

            for person in last_persons:
                x1, y1, x2, y2 = person["bbox"]
                conf = person.get("confidence", 0.0)
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    display,
                    f"person {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                )

            for face in last_face_results:
                x1, y1, x2, y2 = face["bbox"]
                name = face["name"]
                dist = float(face.get("distance", 999.0))
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display,
                    f"{name} | {dist:.2f}",
                    (x1, max(20, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = now

            health_monitor.update_runtime(fps=fps, stream_ok=stream.stream_ok)
            health_monitor.maybe_export()

            if getattr(config, "SHOW_FPS", True):
                cv2.putText(display, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
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
                cv2.putText(
                    display,
                    event_engine.current_overlay_text(),
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            display = resize_keep_ratio(display, getattr(config, "DISPLAY_WIDTH", 960))
            cv2.imshow(getattr(config, "WINDOW_NAME", "Person + Face Security Pipeline"), display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        try:
            uploader.stop()
        except Exception:
            pass
        try:
            stream.stop()
        except Exception:
            pass
        try:
            recorder.force_close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
