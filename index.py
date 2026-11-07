#!/usr/bin/env python3
"""CodeLens — индексирующий скрипт.

Использование:
    python index.py <путь-к-папке-с-кодом> [--model NAME] [--reset]

Пример (после распаковки codebase_python.zip получится папка gymhero/):
    python index.py gymhero

ВАЖНО про chunk_id: путь к файлу в идентификаторе считается ОТНОСИТЕЛЬНО
переданной папки. Чтобы идентификаторы выглядели как «gymhero/security.py:...»,
передавайте ВНЕШНЮЮ папку `gymhero` (внутри неё лежит пакет gymhero/).

Три шага: (1) парсинг .py через ast на функции/классы/методы,
(2) генерация эмбеддингов через sentence-transformers,
(3) сохранение векторов и метаданных в ChromaDB.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from codelens import config
from codelens.chunker import chunk_repo
from codelens.embedder import get_embedder
from codelens.store import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="CodeLens indexer")
    parser.add_argument("directory", help="Папка с .py-файлами (корень для chunk_id)")
    parser.add_argument("--model", default=config.DEFAULT_MODEL,
                        help=f"Модель эмбеддингов (default: {config.DEFAULT_MODEL})")
    parser.add_argument("--reset", action="store_true", default=True,
                        help="Пересоздать коллекцию с нуля (по умолчанию да)")
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"ERROR: {root} — не директория", file=sys.stderr)
        return 1

    print(f"[1/3] Парсинг .py-файлов в {root} ...")
    t0 = time.time()
    chunks = chunk_repo(root)
    if not chunks:
        print("ERROR: не найдено ни одного фрагмента (.py-файлов нет?).", file=sys.stderr)
        return 1
    n_files = len({c.rel_path for c in chunks})
    print(f"      Файлов: {n_files} | фрагментов: {len(chunks)} "
          f"(функций/классов/методов) за {time.time()-t0:.1f}с")

    print(f"[2/3] Генерация эмбеддингов моделью {args.model} ...")
    t0 = time.time()
    embedder = get_embedder(args.model)
    embeddings = embedder.encode_passages(
        [c.embed_text for c in chunks], show_progress=True)
    print(f"      Размерность вектора: {embeddings.shape[1]} | "
          f"время: {time.time()-t0:.1f}с")

    print(f"[3/3] Сохранение в ChromaDB ({config.CHROMA_DIR}) ...")
    t0 = time.time()
    store = VectorStore()
    store.reset(model_name=args.model)
    store.add_chunks(chunks, embeddings)
    print(f"      Сохранено {store.count()} фрагментов за {time.time()-t0:.1f}с")
    print("\nГотово. Запустите интерфейс:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
