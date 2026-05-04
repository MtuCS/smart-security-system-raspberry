from __future__ import annotations

from typing import List, Tuple
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

    def add(self, name: str, embedding: np.ndarray):
        self.names.append(name)
        self.embeddings.append(l2_normalize(embedding.astype(np.float32)))

    def is_empty(self) -> bool:
        return len(self.names) == 0

    def clear(self):
        self.names.clear()
        self.embeddings.clear()

    def match(self, query_embedding: np.ndarray, threshold: float = 0.45) -> Tuple[str, float]:
        if self.is_empty():
            return "Unknown", 999.0

        query = l2_normalize(query_embedding.astype(np.float32))
        dists = [float(np.linalg.norm(query - emb)) for emb in self.embeddings]
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        best_name = self.names[best_idx]

        if best_dist <= threshold:
            return best_name, best_dist
        return "Unknown", best_dist
