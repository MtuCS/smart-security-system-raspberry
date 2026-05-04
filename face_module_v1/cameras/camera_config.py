from __future__ import annotations

CAMERAS = {
    "door_cam": {
        "camera_id": "door_cam",
        "name": "Door Camera",
        "rtsp_url": "rtsp://admin:12345678a@192.168.1.245:554/Streaming/Channels/101",
        "role": "face_access",
        "enable_face": True,
        "enable_counting": False,
        "window_name": "Door Camera - Person + Face",
    },
    "room_cam": {
        "camera_id": "room_cam",
        "name": "Room Camera",
        "rtsp_url": "rtsp://admin:12345678a@192.168.1.248:554/Streaming/Channels/101",
        "role": "occupancy_monitoring",
        "enable_face": False,
        "enable_counting": True,
        "window_name": "Room Camera - Occupancy",
    },
}
