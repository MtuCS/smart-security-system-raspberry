from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


class FaceDatabase:
    def __init__(self):
        self.names: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.sources: List[str] = []

    def add(self, name: str, embedding: np.ndarray, source: str = ""):
        self.names.append(name)
        self.embeddings.append(l2_normalize(embedding.astype(np.float32)))
        self.sources.append(source)

    def clear(self):
        self.names.clear()
        self.embeddings.clear()
        self.sources.clear()

    def is_empty(self) -> bool:
        return len(self.names) == 0

    def __len__(self) -> int:
        return len(self.names)

    def match(self, query_embedding: np.ndarray, threshold: float = 0.45) -> Tuple[str, float]:
        if self.is_empty():
            return "Unknown", 999.0

        query = l2_normalize(query_embedding.astype(np.float32))
        emb_matrix = np.stack(self.embeddings, axis=0)
        dists = np.linalg.norm(emb_matrix - query[None, :], axis=1)

        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        best_name = self.names[best_idx]

        if best_dist <= threshold:
            return best_name, best_dist
        return "Unknown", best_dist

    def save_cache(self, cache_file: str, meta_file: str, db_signature: dict):
        cache_path = Path(cache_file)
        meta_path = Path(meta_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        if self.is_empty():
            raise ValueError("Cannot save empty face database cache.")

        np.savez_compressed(
            cache_path,
            names=np.array(self.names, dtype=object),
            sources=np.array(self.sources, dtype=object),
            embeddings=np.stack(self.embeddings, axis=0).astype(np.float32),
        )
        meta_path.write_text(json.dumps(db_signature, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_cache(self, cache_file: str, meta_file: str, expected_signature: Optional[dict] = None) -> bool:
        cache_path = Path(cache_file)
        meta_path = Path(meta_file)
        if not cache_path.exists() or not meta_path.exists():
            return False

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if expected_signature is not None and meta != expected_signature:
                return False

            data = np.load(cache_path, allow_pickle=True)
            names = data["names"].tolist()
            sources = data["sources"].tolist() if "sources" in data else [""] * len(names)
            embeddings = data["embeddings"]

            self.clear()
            for name, source, embedding in zip(names, sources, embeddings):
                self.add(str(name), np.asarray(embedding, dtype=np.float32), str(source))
            return True
        except Exception as e:
            print(f"[WARN] Không load được embedding cache: {e}")
            return False
