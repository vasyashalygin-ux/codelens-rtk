"""Централизованная конфигурация CodeLens.

Все параметры можно переопределить переменными окружения (префикс CODELENS_),
что удобно для Docker и для синхронизации модели между index.py и app.py.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Пути -------------------------------------------------------------------
# Корень проекта (папка, где лежит этот пакет)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Куда ChromaDB складывает персистентный индекс
CHROMA_DIR = Path(os.getenv("CODELENS_CHROMA_DIR", PROJECT_ROOT / "chroma_db"))

# Имя коллекции в ChromaDB
COLLECTION_NAME = os.getenv("CODELENS_COLLECTION", "codelens_chunks")

# --- Модель эмбеддингов -----------------------------------------------------
# Модель ДОЛЖНА совпадать при индексировании и при поиске. Имя модели, которой
# реально проиндексировали базу, сохраняется в метаданных коллекции, и app.py
# подхватывает его автоматически — это защищает от классической ошибки
# «проиндексировал одной моделью, ищу другой».
#
# Рекомендации по выбору (ru+en, кросс-язычный поиск по коду):
#   intfloat/multilingual-e5-large            — сильная, ~2.2 ГБ, лучший баланс качества (default)
#   BAAI/bge-m3                               — очень сильная многоязычная, ~2.3 ГБ
#   intfloat/multilingual-e5-base             — легче (~1.1 ГБ), чуть слабее
#   paraphrase-multilingual-MiniLM-L12-v2     — самая лёгкая (рекомендация ТЗ), 384-dim
DEFAULT_MODEL = os.getenv("CODELENS_MODEL", "intfloat/multilingual-e5-large")

# Модели семейства e5 требуют префиксов "query:"/"passage:". Определяем автоматически.
def needs_e5_prefix(model_name: str) -> bool:
    return "e5" in model_name.lower()

# --- Параметры поиска -------------------------------------------------------
TOP_K = int(os.getenv("CODELENS_TOP_K", "5"))          # сколько результатов показывать
CANDIDATES = int(os.getenv("CODELENS_CANDIDATES", "30"))  # сколько кандидатов тянуть до реранка/гибрида

# Вес плотного (векторного) поиска в гибриде: 1.0 = только вектор, 0.0 = только BM25
HYBRID_ALPHA = float(os.getenv("CODELENS_HYBRID_ALPHA", "0.75"))
USE_HYBRID = os.getenv("CODELENS_HYBRID", "1") == "1"

# Кросс-энкодер реранкер (опционально, тяжелее). Пусто = выключен.
RERANKER_MODEL = os.getenv("CODELENS_RERANKER", "")  # напр. "BAAI/bge-reranker-v2-m3"

# --- LLM (опционально) ------------------------------------------------------
LLM_BACKEND = os.getenv("CODELENS_LLM_BACKEND", "ollama")  # "ollama" | "openai" | "none"
OLLAMA_HOST = os.getenv("CODELENS_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("CODELENS_OLLAMA_MODEL", "mistral:7b")
OPENAI_MODEL = os.getenv("CODELENS_OPENAI_MODEL", "gpt-4o-mini")
