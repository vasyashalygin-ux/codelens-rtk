"""Опциональная генерация связного ответа (полный RAG).

Найденные фрагменты + вопрос → LLM → человекочитаемое объяснение со ссылками
на файлы и строки. Бэкенд выбирается конфигом: локальная Ollama (mistral:7b)
или облачный OpenAI API. Если ничего не доступно — поиск работает и без LLM.
"""
from __future__ import annotations

from . import config

SYSTEM_PROMPT = (
    "Ты — ассистент по навигации в кодовой базе. Отвечай на вопрос разработчика, "
    "опираясь ТОЛЬКО на приведённые фрагменты кода. Кратко объясни, как это "
    "реализовано, и обязательно ссылайся на файлы и имена функций/классов. "
    "Если в контексте нет ответа — честно скажи, что в найденном коде этого нет. "
    "Отвечай на языке вопроса."
)


def _build_context(results: list[dict]) -> str:
    blocks = []
    for r in results:
        m = r["meta"]
        blocks.append(
            f"### {r['chunk_id']}\n"
            f"({m.get('kind')} {m.get('name')} в {m.get('rel_path')}, "
            f"строки {m.get('start_line')}-{m.get('end_line')})\n"
            f"```python\n{r['code']}\n```"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, results: list[dict]):
    context = _build_context(results)
    user = (
        f"Вопрос: {question}\n\n"
        f"Найденные фрагменты кода:\n{context}\n\n"
        f"Дай связный ответ со ссылками на конкретные файлы и функции."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_answer(question: str, results: list[dict],
                    backend: str | None = None) -> str:
    backend = backend or config.LLM_BACKEND
    messages = build_messages(question, results)
    if backend == "ollama":
        return _ollama(messages)
    if backend == "openai":
        return _openai(messages)
    return "(LLM-режим выключен. Включите Ollama или OpenAI в настройках.)"


def _ollama(messages) -> str:
    import requests
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={"model": config.OLLAMA_MODEL, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _openai(messages) -> str:
    from openai import OpenAI
    client = OpenAI()  # ключ берётся из переменной окружения OPENAI_API_KEY
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL, messages=messages, temperature=0.2,
    )
    return resp.choices[0].message.content


def llm_available(backend: str | None = None) -> bool:
    backend = backend or config.LLM_BACKEND
    if backend == "ollama":
        try:
            import requests
            requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=2)
            return True
        except Exception:
            return False
    if backend == "openai":
        import os
        return bool(os.getenv("OPENAI_API_KEY"))
    return False
