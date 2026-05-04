from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import cv2

import face_module_v1.config as config
from face_module_v1.event_engine import EventEngine
from face_module_v1.face_engine import FaceEngine
from face_module_v1.person_detector import PersonDetector
from face_module_v1.recorder import EventRecorder
from face_module_v1.stream_reader import StreamReader
from face_module_v1.tracking_logic import (
    LineCrossingDetector,
    OccupancySmoother,
    SimpleIoUTracker,
    Track,
    TrackIdentityManager,
    draw_rect,
    point_in_rect,
    rect_from_ratio,
)


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


def scale_face_results(face_results: List[Dict[str, Any]], roi_box: List[int]) -> List[Dict[str, Any]]:
    rx1, ry1, _, _ = roi_box
    scaled_results = []
    for item in face_results:
        x1, y1, x2, y2 = item["bbox"]
        scaled_results.append(
            {
                "bbox": [x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1],
                "name": item["name"],
                "distance": item["distance"],
            }
        )
    return scaled_results


def choose_tracks_for_identity(tracks: List[Track], door_rect: List[int], max_tracks: int) -> List[Track]:
    in_zone = [t for t in tracks if point_in_rect(t.center, door_rect)]
    if not in_zone:
        return []
    cx = (door_rect[0] + door_rect[2]) / 2
    cy = (door_rect[1] + door_rect[3]) / 2
    in_zone.sort(key=lambda t: (t.center[0] - cx) ** 2 + (t.center[1] - cy) ** 2)
    return in_zone[: max(1, int(max_tracks))]


def main():
    camera_id = getattr(config, "CAMERA_ID", "camera_01")

    stream = StreamReader(rtsp_url=config.RTSP_URL)
    face_engine = FaceEngine(
        det_size=(getattr(config, "MAX_FACE_SIZE", 320), getattr(config, "MAX_FACE_SIZE", 320)),
        model_name=getattr(config, "FACE_MODEL_NAME", getattr(config, "MODEL_NAME", "buffalo_s")),
        cache_path=getattr(config, "FACE_DB_CACHE_PATH", getattr(config, "FACE_DB_CACHE_FILE", None)),
    )
    face_engine.build_database(config.FACE_DB_DIR)

    person_detector = PersonDetector(
        conf=getattr(config, "PERSON_CONFIDENCE", 0.4),
        imgsz=getattr(config, "PERSON_INFER_WIDTH", 640),
    )
    recorder = EventRecorder(
        base_dir=getattr(config, "EVENTS_DIR", "data/events"),
        fps=getattr(config, "EVENT_VIDEO_FPS", getattr(config, "VIDEO_FPS", 12)),
        pre_roll_seconds=getattr(config, "EVENT_PRE_ROLL_SECONDS", getattr(config, "PRE_ROLL_SECONDS", 5.0)),
        post_roll_seconds=getattr(config, "EVENT_POST_ROLL_SECONDS", getattr(config, "POST_ROLL_SECONDS", 6.0)),
    )
    event_engine = EventEngine(
        camera_id=camera_id,
        start_confirm_frames=getattr(config, "EVENT_START_CONFIRM_FRAMES", 2),
        identity_confirm_frames=getattr(config, "EVENT_IDENTITY_CONFIRM_FRAMES", 3),
        end_absence_seconds=getattr(config, "EVENT_END_ABSENCE_SECONDS", 2.0),
        start_cooldown_seconds=getattr(config, "EVENT_START_COOLDOWN_SECONDS", 1.5),
        min_update_interval_seconds=getattr(config, "EVENT_MIN_UPDATE_INTERVAL_SECONDS", 0.8),
    )

    enable_tracking = bool(getattr(config, "ENABLE_TRACKING", True))
    tracker = SimpleIoUTracker(
        iou_threshold=getattr(config, "TRACK_IOU_THRESHOLD", 0.25),
        max_missed=getattr(config, "TRACK_MAX_MISSED", 12),
    )
    identity_manager = TrackIdentityManager(
        max_age_seconds=getattr(config, "TRACK_IDENTITY_MAX_AGE_SECONDS", 30.0),
    )
    occupancy = OccupancySmoother(
        window_size=getattr(config, "OCCUPANCY_SMOOTHING_WINDOW", 3),
        min_agree=getattr(config, "OCCUPANCY_SMOOTHING_MIN_AGREE", 3),
    )
    line_crossing = LineCrossingDetector(
        p1_ratio=getattr(config, "VIRTUAL_LINE_P1_RATIO", (0.20, 0.50)),
        p2_ratio=getattr(config, "VIRTUAL_LINE_P2_RATIO", (0.85, 0.50)),
        in_direction=getattr(config, "VIRTUAL_LINE_IN_DIRECTION", "negative_to_positive"),
        min_move_px=getattr(config, "VIRTUAL_LINE_MIN_MOVE_PX", 10),
        cooldown_seconds=getattr(config, "VIRTUAL_LINE_COOLDOWN_SECONDS", 1.0),
    )

    stream.start()

    frame_count = 0
    prev_time = time.time()
    fps = 0.0

    last_persons: List[Dict[str, Any]] = []
    last_tracks: List[Track] = []
    last_face_results: List[Dict[str, Any]] = []
    last_occupancy = None

    person_every = int(getattr(config, "PERSON_DETECT_EVERY_N_FRAMES", 3))
    face_every = int(getattr(config, "FACE_DETECT_EVERY_N_FRAMES", getattr(config, "FACE_RECOG_EVERY_N_FRAMES", 2)))
    recog_threshold = float(getattr(config, "RECOG_THRESHOLD", 0.9))

    try:
        while True:
            frame = stream.read()
            if frame is None:
                event_engine.flush_if_timeout(recorder=recorder)
                recorder.update(None)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                time.sleep(0.01)
                continue

            frame_count += 1
            recorder.update(frame)
            display = frame.copy()
            h, w = frame.shape[:2]
            door_rect = rect_from_ratio(getattr(config, "DOOR_ZONE_RATIO", (0.25, 0.0, 0.75, 0.45)), w, h)

            detection_updated = False
            if frame_count % max(1, person_every) == 0:
                try:
                    last_persons = person_detector.detect(frame)
                    detection_updated = True
                    if enable_tracking:
                        last_tracks = tracker.update(last_persons)
                        identity_manager.cleanup()
                        confirmed_count, changed = occupancy.update(len(last_tracks))
                        if changed:
                            last_occupancy = confirmed_count
                except Exception as e:
                    print(f"[WARN] person detect/tracking error: {e}")
                    last_persons = []
                    last_tracks = []

            primary_person = pick_primary_person(last_persons)
            current_name = "Unknown"
            current_conf = 0.0

            # Face recognition ưu tiên track nằm trong vùng cửa để gán identity cho track_id.
            if enable_tracking and last_tracks and frame_count % max(1, face_every) == 0:
                last_face_results = []
                target_tracks = choose_tracks_for_identity(
                    last_tracks,
                    door_rect,
                    max_tracks=getattr(config, "TRACK_RECOGNIZE_MAX_TRACKS", 1),
                )
                for tr in target_tracks:
                    roi, roi_box = crop_upper_body_for_face(frame, tr.bbox)
                    if roi is None:
                        continue
                    try:
                        results = face_engine.infer(roi, threshold=recog_threshold)
                        scaled = scale_face_results(results, roi_box)
                        last_face_results.extend(scaled)
                        best = select_best_face(scaled)
                        if best is not None:
                            name = best.get("name", "Unknown")
                            conf = max(0.0, 1.0 - float(best.get("distance", 1.0)))
                            identity_manager.assign_identity(tr.track_id, name, conf)
                    except Exception as e:
                        print(f"[WARN] face infer for track error: {e}")
            elif primary_person is not None and frame_count % max(1, face_every) == 0:
                # Fallback giữ logic cũ nếu chưa bật tracking.
                roi, roi_box = crop_upper_body_for_face(frame, primary_person["bbox"])
                if roi is not None:
                    try:
                        results = face_engine.infer(roi, threshold=recog_threshold)
                        last_face_results = scale_face_results(results, roi_box)
                    except Exception as e:
                        print(f"[WARN] face infer error: {e}")
                        last_face_results = []
                else:
                    last_face_results = []
            elif primary_person is None:
                last_face_results = []

            # Line crossing chỉ kiểm tra sau khi tracker vừa update bằng detection mới.
            if enable_tracking and detection_updated:
                for tr in last_tracks:
                    identity_manager.touch(tr.track_id)
                    crossing = line_crossing.check(tr, w, h)
                    if crossing is None:
                        continue
                    identity = identity_manager.get_identity(tr.track_id)
                    name = identity.get("name", "Unknown")
                    if crossing == "IN":
                        print(f"[ROOM EVENT] IN | track_id={tr.track_id} | name={name}")
                    else:
                        print(f"[ROOM EVENT] OUT | track_id={tr.track_id} | name={name}")
                        identity_manager.remove(tr.track_id)

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
                    "tracked_count": len(last_tracks),
                    "occupancy": last_occupancy,
                    "stream_ok": stream.stream_ok,
                    "camera_id": camera_id,
                },
            )
            event_engine.flush_if_timeout(recorder=recorder)

            # Draw person detections.
            for person in last_persons:
                x1, y1, x2, y2 = person["bbox"]
                conf = person.get("confidence", 0.0)
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(display, f"person {conf:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            # Draw tracks + identity mapping.
            if enable_tracking:
                for tr in last_tracks:
                    x1, y1, x2, y2 = tr.bbox
                    identity = identity_manager.get_identity(tr.track_id)
                    label = f"ID {tr.track_id}: {identity.get('name', 'Unknown')}"
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 180, 0), 2)
                    cv2.circle(display, tr.center, 4, (255, 180, 0), -1)
                    cv2.putText(display, label, (x1, min(h - 10, y2 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

            for face in last_face_results:
                x1, y1, x2, y2 = face["bbox"]
                name = face["name"]
                dist = float(face.get("distance", 999.0))
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{name} | {dist:.2f}", (x1, max(20, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            if getattr(config, "DRAW_DOOR_ZONE", True):
                draw_rect(display, door_rect, "door zone", color=(255, 0, 255))
            if getattr(config, "DRAW_VIRTUAL_LINE", True):
                line_crossing.draw(display)

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = now

            if getattr(config, "SHOW_FPS", True):
                cv2.putText(display, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                status = "STREAM OK" if stream.stream_ok else "RECONNECTING..."
                cv2.putText(display, status, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if stream.stream_ok else (0, 0, 255), 2)
                cv2.putText(display, event_engine.current_overlay_text(), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Tracked: {len(last_tracks)} | Occupancy: {last_occupancy}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            display = resize_keep_ratio(display, getattr(config, "DISPLAY_WIDTH", 960))
            cv2.imshow(getattr(config, "WINDOW_NAME", "Person + Face Security Pipeline"), display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
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
