import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config


class FaceDatabase:
    """Lightweight embedding store backed by a pickle file.

    Supports multi-shot registration (several embeddings per identity).
    Identification compares the query against each identity's mean prototype
    using cosine similarity and returns the best match above a threshold.
    """

    def __init__(self, db_path: Path = config.DB_PATH):
        self._db_path = db_path
        # { name: [embedding_0, embedding_1, ...] }
        self._db: Dict[str, List[np.ndarray]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._db_path.exists():
            with open(self._db_path, "rb") as f:
                self._db = pickle.load(f)

    def save(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._db_path, "wb") as f:
            pickle.dump(self._db, f)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(self, name: str, embedding: np.ndarray) -> None:
        """Register an L2-normalised embedding under *name*."""
        self._db.setdefault(name, []).append(embedding.astype(np.float32))
        self.save()

    def remove(self, name: str) -> bool:
        """Delete all embeddings for *name*. Returns True if found."""
        if name in self._db:
            del self._db[name]
            self.save()
            return True
        return False

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def identify(
        self,
        query: np.ndarray,
        threshold: float = config.RECOGNITION_THRESHOLD,
    ) -> Tuple[Optional[str], float]:
        """Find the closest registered identity.

        Returns:
            (name, similarity) when best match >= threshold,
            (None,  similarity) otherwise.
        """
        if not self._db:
            return None, 0.0

        best_name: Optional[str] = None
        best_sim:  float         = -1.0

        for name, embeddings in self._db.items():
            prototype = np.mean(embeddings, axis=0)
            prototype /= (np.linalg.norm(prototype) + 1e-8)
            sim = float(np.dot(query, prototype))
            if sim > best_sim:
                best_sim  = sim
                best_name = name

        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_identities(self) -> List[str]:
        return list(self._db.keys())

    def __len__(self) -> int:
        return len(self._db)
