from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from face_module_v1.face_db import FaceDatabase
import face_module_v1.config as config


class FaceEngine:
    def __init__(self, det_size=(416, 416), model_name="buffalo_l"):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=-1, det_size=det_size)
        self.db = FaceDatabase()

    def _collect_manifest(self, db_dir: str) -> List[Tuple[str, int, float]]:
        db_path = Path(db_dir)
        manifest: List[Tuple[str, int, float]] = []
        for img_path in sorted(db_path.rglob("*")):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            stat = img_path.stat()
            manifest.append((str(img_path), int(stat.st_size), float(stat.st_mtime)))
        return manifest

    def _manifest_hash(self, manifest: List[Tuple[str, int, float]]) -> str:
        raw = repr(manifest).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _load_cache(self, cache_file: Path, expected_hash: str) -> bool:
        if not cache_file.exists():
            return False
        try:
            with cache_file.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("manifest_hash") != expected_hash:
                return False
            self.db.clear()
            for name, emb in zip(payload["names"], payload["embeddings"]):
                self.db.add(name, emb)
            print(f"[INFO] Loaded face DB cache: {len(self.db.names)} embeddings")
            return True
        except Exception as e:
            print(f"[WARN] Cannot load face cache: {e}")
            return False

    def _save_cache(self, cache_file: Path, manifest_hash: str):
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "manifest_hash": manifest_hash,
                "names": self.db.names,
                "embeddings": self.db.embeddings,
            }
            with cache_file.open("wb") as f:
                pickle.dump(payload, f)
            print(f"[INFO] Saved face DB cache: {cache_file}")
        except Exception as e:
            print(f"[WARN] Cannot save face cache: {e}")

    def build_database(self, db_dir: str):
        db_path = Path(db_dir)
        if not db_path.exists():
            raise FileNotFoundError(f"Face DB not found: {db_dir}")

        manifest = self._collect_manifest(db_dir)
        manifest_hash = self._manifest_hash(manifest)
        if config.ENABLE_FACE_DB_CACHE and self._load_cache(Path(config.FACE_DB_CACHE_FILE), manifest_hash):
            return

        self.db.clear()
        total_imgs = 0
        total_faces = 0
        total_failed = 0

        for person_dir in sorted(db_path.iterdir()):
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            person_ok = 0
            person_fail = 0

            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                    continue

                total_imgs += 1
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[WARN] Cannot read image: {img_path}")
                    total_failed += 1
                    person_fail += 1
                    continue

                faces = self.app.get(img)
                if not faces:
                    print(f"[WARN] No face found: {img_path}")
                    total_failed += 1
                    person_fail += 1
                    continue

                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                self.db.add(person_name, face.embedding)
                total_faces += 1
                person_ok += 1

            print(f"[INFO] {person_name}: ok={person_ok}, fail={person_fail}")

        print(f"[INFO] Loaded DB: {total_faces} embeddings from {total_imgs} images. Failed: {total_failed}")
        if config.ENABLE_FACE_DB_CACHE:
            self._save_cache(Path(config.FACE_DB_CACHE_FILE), manifest_hash)

    def infer(self, frame: np.ndarray, threshold: float = 0.45) -> List[Dict[str, Any]]:
        faces = self.app.get(frame)
        results: List[Dict[str, Any]] = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            name, dist = self.db.match(face.embedding, threshold=threshold)
            results.append({
                "bbox": bbox,
                "name": name,
                "distance": dist,
            })
        return results
