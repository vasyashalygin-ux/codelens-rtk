"""Персистентное векторное хранилище на ChromaDB.

ChromaDB — встроенная векторная БД на Python, ищет ближайших соседей по
косинусному расстоянию, работает из коробки без сервера. Мы передаём
собственные эмбеддинги (а не используем встроенную embedding_function), чтобы
полностью контролировать модель и префиксы query/passage.

В метаданные коллекции пишется имя модели эмбеддингов — app.py читает его и
автоматически грузит ту же модель, что использовалась при индексировании.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .chunker import Chunk


class VectorStore:
    def __init__(self, persist_dir: Path | None = None,
                 collection_name: str | None = None):
        import chromadb
        self.persist_dir = Path(persist_dir or config.CHROMA_DIR)
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = None

    # --- запись ------------------------------------------------------------
    def reset(self, model_name: str):
        """Пересоздаёт коллекцию с нуля (idempotent re-index)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": model_name},
        )
        return self._collection

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray, batch: int = 256):
        col = self.collection
        n = len(chunks)
        for i in range(0, n, batch):
            sl = slice(i, i + batch)
            part = chunks[sl]
            col.add(
                ids=[c.chunk_id for c in part],
                embeddings=[e.tolist() for e in embeddings[sl]],
                documents=[c.code for c in part],
                metadatas=[{
                    "rel_path": c.rel_path,
                    "name": c.name,
                    "kind": c.kind,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "docstring": c.docstring,
                } for c in part],
            )

    # --- чтение -------------------------------------------------------------
    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_collection(self.collection_name)
        return self._collection

    @property
    def model_name(self) -> str:
        meta = self.collection.metadata or {}
        return meta.get("embedding_model", config.DEFAULT_MODEL)

    def count(self) -> int:
        return self.collection.count()

    def query(self, query_embedding: np.ndarray, n_results: int):
        res = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append({
                "chunk_id": cid,
                "code": doc,
                "meta": meta,
                "distance": dist,           # косинусное расстояние (0 = идентично)
                "similarity": 1.0 - dist,    # косинусная близость в [~0..1]
            })
        return out

    def all_documents(self):
        """Возвращает (ids, documents, metadatas) для построения BM25-индекса."""
        data = self.collection.get(include=["documents", "metadatas"])
        return data["ids"], data["documents"], data["metadatas"]

    def exists(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False
