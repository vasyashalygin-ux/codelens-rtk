"""Поиск: плотный (векторный) + опциональный гибрид с BM25 + опциональный реранк.

- Плотный поиск: эмбеддинг запроса → ближайшие соседи в ChromaDB (косинус).
- Гибрид: к плотному добавляется лексический BM25 (хорош для точных совпадений
  имён функций/переменных и для запросов про несуществующую функциональность).
  Итоговый скор = alpha * dense + (1 - alpha) * bm25_norm.
- Реранк (опц.): кросс-энкодер переоценивает top-кандидатов парами (запрос, код).
"""
from __future__ import annotations

import re
from functools import lru_cache

from . import config
from .embedder import get_embedder
from .store import VectorStore

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Токенизация с разбиением snake_case и camelCase на под-слова."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        raw_l = raw.lower()
        tokens.append(raw_l)
        # snake_case
        parts = raw.split("_")
        # camelCase
        cam = []
        for p in parts:
            cam.extend(_CAMEL_RE.sub(" ", p).split())
        for c in cam:
            cl = c.lower()
            if cl and cl != raw_l:
                tokens.append(cl)
    return tokens


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class Retriever:
    def __init__(self, store: VectorStore | None = None, model_name: str | None = None):
        self.store = store or VectorStore()
        # модель берём из метаданных коллекции, чтобы query совпал с passage
        self.model_name = model_name or self.store.model_name
        self.embedder = get_embedder(self.model_name)
        self._bm25 = None
        self._bm25_ids = None
        self._reranker = None

    # --- BM25 ---------------------------------------------------------------
    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi
        ids, docs, metas = self.store.all_documents()
        corpus_tokens = []
        for doc, meta in zip(docs, metas):
            name = meta.get("name", "")
            docstring = meta.get("docstring", "")
            corpus_tokens.append(tokenize(f"{name} {docstring} {doc}"))
        self._bm25 = BM25Okapi(corpus_tokens)
        self._bm25_ids = ids

    def _reranker_model(self):
        if self._reranker is None and config.RERANKER_MODEL:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(config.RERANKER_MODEL)
        return self._reranker

    # --- основной поиск -----------------------------------------------------
    def search(self, query: str, top_k: int = None, use_hybrid: bool = None,
               alpha: float = None, candidates: int = None, rerank: bool = False):
        top_k = top_k or config.TOP_K
        use_hybrid = config.USE_HYBRID if use_hybrid is None else use_hybrid
        alpha = config.HYBRID_ALPHA if alpha is None else alpha
        candidates = candidates or config.CANDIDATES

        # 1) плотный поиск
        q_emb = self.embedder.encode_query(query)
        dense_hits = self.store.query(q_emb, n_results=candidates)
        dense_sim = {h["chunk_id"]: h["similarity"] for h in dense_hits}
        info = {h["chunk_id"]: h for h in dense_hits}

        if not use_hybrid:
            fused = dense_sim
        else:
            # 2) BM25 по всему корпусу
            self._ensure_bm25()
            bm_scores_arr = self._bm25.get_scores(tokenize(query))
            bm_all = {cid: float(s) for cid, s in zip(self._bm25_ids, bm_scores_arr)}
            # объединяем кандидатов: плотные + топ BM25
            top_bm_ids = sorted(bm_all, key=bm_all.get, reverse=True)[:candidates]
            union = set(dense_sim) | set(top_bm_ids)
            dense_n = _minmax({c: dense_sim.get(c, 0.0) for c in union})
            bm_n = _minmax({c: bm_all.get(c, 0.0) for c in union})
            fused = {c: alpha * dense_n[c] + (1 - alpha) * bm_n[c] for c in union}
            # дотягиваем метаданные для BM25-only кандидатов
            missing = [c for c in union if c not in info]
            if missing:
                got = self.store.collection.get(
                    ids=missing, include=["documents", "metadatas"])
                for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                    info[cid] = {"chunk_id": cid, "code": doc, "meta": meta,
                                 "similarity": dense_sim.get(cid, 0.0)}

        ranked = sorted(fused, key=fused.get, reverse=True)[:max(top_k, candidates if rerank else top_k)]

        # 3) опциональный реранк кросс-энкодером
        if rerank and self._reranker_model() is not None:
            pairs = [(query, info[c]["code"]) for c in ranked]
            ce_scores = self._reranker_model().predict(pairs)
            scored = sorted(zip(ranked, ce_scores), key=lambda x: x[1], reverse=True)
            ranked = [c for c, _ in scored]

        results = []
        for cid in ranked[:top_k]:
            h = dict(info[cid])
            h["score"] = float(fused[cid])
            # процент релевантности для UI: используем косинусную близость, если есть
            sim = h.get("similarity")
            h["relevance_pct"] = round(max(0.0, min(1.0, sim if sim is not None else fused[cid])) * 100, 1)
            results.append(h)
        return results


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()
