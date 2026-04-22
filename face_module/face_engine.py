from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import face_module.config as config
from face_module.face_db import FaceDatabase

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class FaceEngine:
    def __init__(self, det_size=(416, 416), model_name="buffalo_s"):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=-1, det_size=det_size)
        self.db = FaceDatabase()

    def _iter_image_files(self, db_dir: str):
        db_path = Path(db_dir)
        if not db_path.exists():
            raise FileNotFoundError(f"Face DB not found: {db_dir}")

        for person_dir in sorted(db_path.iterdir()):
            if not person_dir.is_dir():
                continue
            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() in VALID_EXTS:
                    yield person_dir.name, img_path

    def _compute_db_signature(self, db_dir: str) -> dict:
        files = list(self._iter_image_files(db_dir))
        items = []
        for person_name, img_path in files:
            st = img_path.stat()
            items.append({
                "person": person_name,
                "path": str(img_path.resolve()),
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            })

        return {
            "db_dir": str(Path(db_dir).resolve()),
            "file_count": len(items),
            "items": items,
            "model_name": config.MODEL_NAME,
            "det_size": int(config.DETECTION_SIZE),
        }

    def build_database(self, db_dir: str, use_cache: bool = True):
        signature = self._compute_db_signature(db_dir)

        if use_cache and self.db.load_cache(
            config.EMBEDDING_CACHE_FILE,
            config.EMBEDDING_CACHE_META_FILE,
            expected_signature=signature,
        ):
            print(f"[INFO] Đã load embedding cache: {len(self.db)} embeddings")
            return

        self.db.clear()
        total_imgs = 0
        total_faces = 0
        total_failed = 0

        db_path = Path(db_dir)
        if not db_path.exists():
            raise FileNotFoundError(f"Face DB not found: {db_dir}")

        for person_dir in sorted(db_path.iterdir()):
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            person_ok = 0
            person_fail = 0

            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in VALID_EXTS:
                    continue

                total_imgs += 1
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[WARN] Cannot read image: {img_path}")
                    total_failed += 1
                    person_fail += 1
                    continue

                faces = self.app.get(img)
                if len(faces) == 0:
                    print(f"[WARN] No face found: {img_path}")
                    total_failed += 1
                    person_fail += 1
                    continue

                face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                )
                self.db.add(person_name, face.embedding, source=str(img_path))
                total_faces += 1
                person_ok += 1

            print(f"[INFO] {person_name}: ok={person_ok}, fail={person_fail}")

        print(f"[INFO] Loaded DB: {total_faces} embeddings from {total_imgs} images. Failed: {total_failed}")
        if use_cache and not self.db.is_empty():
            self.db.save_cache(config.EMBEDDING_CACHE_FILE, config.EMBEDDING_CACHE_META_FILE, signature)
            print(f"[INFO] Đã lưu embedding cache -> {config.EMBEDDING_CACHE_FILE}")

    def infer(self, frame: np.ndarray, threshold: float = 0.45) -> List[Dict[str, Any]]:
        faces = self.app.get(frame)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            name, dist = self.db.match(face.embedding, threshold=threshold)
            results.append({
                "bbox": bbox,
                "name": name,
                "distance": dist,
            })

        return results
