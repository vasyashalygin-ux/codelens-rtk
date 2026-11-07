#!/usr/bin/env python3
"""Прогон тестового набора и подсчёт Precision@5.

Использование:
    python evaluate.py [--questions data/eval_questions.json]
                       [--out results.json]
                       [--hybrid/--no-hybrid] [--alpha 0.75] [--rerank]

Делает три вещи:
  1) прогоняет каждый вопрос из eval_questions.json через ретривер,
  2) пишет results.json в формате скорера (question_id + top_5_chunks),
  3) запускает score.py и печатает Precision@5.

Метрику из строки `Total score:` и файл results.json предъявляют на защите.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from codelens.retriever import Retriever
from codelens.store import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="CodeLens evaluation")
    parser.add_argument("--questions", default="data/eval_questions.json")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--score-script", default="score.py")
    parser.add_argument("--hybrid", dest="hybrid", action="store_true", default=True)
    parser.add_argument("--no-hybrid", dest="hybrid", action="store_false")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--rerank", action="store_true", default=False)
    args = parser.parse_args()

    store = VectorStore()
    if not store.exists() or store.count() == 0:
        print("ERROR: индекс пуст. Сначала: python index.py gymhero", file=sys.stderr)
        return 1

    retriever = Retriever(store)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    results = []
    latencies = []
    for q in questions:
        t0 = time.time()
        hits = retriever.search(
            q["query"], top_k=5, use_hybrid=args.hybrid,
            alpha=args.alpha, rerank=args.rerank)
        latencies.append(time.time() - t0)
        results.append({
            "question_id": q["question_id"],
            "top_5_chunks": [h["chunk_id"] for h in hits],
        })

    Path(args.out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results.json сохранён ({len(results)} вопросов). "
          f"Средняя задержка поиска: {sum(latencies)/len(latencies):.3f}с "
          f"(макс {max(latencies):.3f}с)\n")

    # Запускаем официальный скорер
    score_script = Path(args.score_script)
    if score_script.exists():
        subprocess.run([sys.executable, str(score_script),
                        "--predictions", args.out,
                        "--questions", args.questions])
    else:
        print(f"(score.py не найден по пути {score_script} — запустите вручную)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
