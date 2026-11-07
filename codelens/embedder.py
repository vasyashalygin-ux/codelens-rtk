"""Обёртка над sentence-transformers.

Изолирует выбор модели и тонкости префиксов (e5 требует "query:"/"passage:").
Ленивая загрузка модели — импорт пакета не тянет тяжёлый sentence-transformers,
пока он реально не понадобится (важно для быстрого старта score-only сценариев).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import config


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.DEFAULT_MODEL
        self._model = None
        self._e5 = config.needs_e5_prefix(self.model_name)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # ленивый импорт
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _prep(self, texts: list[str], kind: str) -> list[str]:
        # kind: "query" | "passage"
        if not self._e5:
            return texts
        prefix = "query: " if kind == "query" else "passage: "
        return [prefix + t for t in texts]

    def encode_passages(self, texts: list[str], batch_size: int = 32,
                        show_progress: bool = False) -> np.ndarray:
        prepped = self._prep(texts, "passage")
        return self.model.encode(
            prepped, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=show_progress, convert_to_numpy=True,
        )

    def encode_query(self, text: str) -> np.ndarray:
        prepped = self._prep([text], "query")
        return self.model.encode(
            prepped, normalize_embeddings=True, convert_to_numpy=True,
        )[0]


@lru_cache(maxsize=4)
def get_embedder(model_name: str | None = None) -> Embedder:
    """Кэшированная фабрика — одна модель на процесс (важно для Streamlit)."""
    return Embedder(model_name)
