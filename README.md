# CodeLens — умный семантический поиск по кодовой базе

> Кейс 3 «CodeLens RAG» · Чемпионат ГК «Ростелеком», 2-й этап · Направление «Искусственный интеллект»

CodeLens отвечает на вопросы о коде на естественном языке (русском и английском) и
возвращает релевантные фрагменты, **даже если слова вопроса не совпадают с именами
функций и переменных**. Запрос «как здесь обрабатываются ошибки авторизации» находит
нужный код, потому что поиск идёт по смыслу (векторные эмбеддинги), а не по тексту (`grep`).

---

## TL;DR — запуск двумя командами

```bash
# 0) однократно: окружение и зависимости
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 0.1) распаковать кодовую базу (создаст папку gymhero/)
python -c "import zipfile; zipfile.ZipFile('data/codebase_python.zip').extractall('.')"

# 1) ИНДЕКСИРОВАНИЕ
python index.py gymhero

# 2) ВЕБ-ИНТЕРФЕЙС
streamlit run app.py
```

Открывается на `http://localhost:8501`. Первый запуск `index.py` скачивает модель
эмбеддингов (~2 ГБ для модели по умолчанию) — нужен интернет и ~5–10 минут.

> **Почему `python index.py gymhero`, а не `gymhero/gymhero`?** Идентификатор
> фрагмента считается *относительно переданной папки*. Эталоны в `eval_questions.json`
> выглядят как `gymhero/security.py:...`, поэтому корнем должна быть **внешняя** папка
> `gymhero` (внутри неё лежит пакет `gymhero/`).

---

## Архитектура системы

Всё на Python. Никакого Java, React/npm, отдельного фронтенда. Docker — опционально.

```
                          ┌──────────────────────── index.py ────────────────────────┐
                          │                                                            │
   gymhero/*.py  ───────► │  1. chunker (ast)      2. embedder            3. store     │ ───► ChromaDB
   (155 фрагментов)       │  функции/классы/   →   sentence-transformers  →  cosine    │      (персистентно
                          │  методы → chunk_id     (вектор 384–1024-dim)   HNSW-индекс  │       в ./chroma_db)
                          └────────────────────────────────────────────────────────────┘

                          ┌──────────────── app.py (Streamlit) / evaluate.py ──────────┐
   запрос ru/en  ───────► │  embedder(query)  →  retriever:                            │
                          │                       • dense  (косинус, ChromaDB)         │ ───► top-5 фрагментов
                          │                       • + BM25 (гибрид, опц.)              │      с подсветкой +
                          │                       • + cross-encoder rerank (опц.)      │      % релевантности
                          │                                              │             │
                          │                                              ▼ (опц.)      │
                          │                                       LLM (Ollama/OpenAI)  │ ───► связный ответ
                          └────────────────────────────────────────────────────────────┘
```

### Компоненты

| Слой | Реализация | Файл |
|------|------------|------|
| Парсинг | `ast` (встроенный) → функции/классы/методы | `codelens/chunker.py` |
| Эмбеддинги | `sentence-transformers`, модель многоязычная | `codelens/embedder.py` |
| Векторная БД | **ChromaDB**, косинусное расстояние, HNSW | `codelens/store.py` |
| Поиск | dense + BM25-гибрид + опц. реранк | `codelens/retriever.py` |
| UI | **Streamlit**: поиск, метрики, чат | `app.py` |
| LLM (опц.) | Ollama (`mistral:7b`) или OpenAI | `codelens/llm.py` |
| Оценка | прогон `eval_questions.json` → `results.json` → `score.py` | `evaluate.py` |

### Поток данных

**Индексирование.** `index.py` обходит папку, находит `.py`-файлы. Каждый режется
модулем `ast` на функции, классы и методы. Для каждого фрагмента строится
**обогащённый текст** (путь + тип + имя + сигнатура + docstring + код), который
переводится в нормированный вектор и сохраняется в ChromaDB вместе с метаданными
(путь, имя, тип, строки, docstring).

**Поиск.** Запрос той же моделью переводится в вектор; ChromaDB возвращает ближайших
по косинусу. Опционально результат смешивается с BM25 (лексическим поиском) и
переоценивается кросс-энкодером. Streamlit рисует карточки с подсветкой синтаксиса и
оценкой релевантности. В режиме чата фрагменты + вопрос уходят в LLM за связным ответом.

---

## Обоснование архитектурных решений

### Стратегия чанкования: **один chunk = одна функция / класс / метод**

Функция — это **минимальная самодостаточная семантическая единица** Python-кода: у неё
есть имя, сигнатура и часто docstring, которые на естественном языке описывают, *что*
она делает. Именно по этому описанию ищет пользователь. Альтернативы хуже:

- *по строкам/окнам фиксированного размера* — рвут логику посередине, теряют границы
  смысла, дают «обрывки» в выдаче;
- *по файлам целиком* — слишком крупно: один вектор на 200 строк размывает смысл, и
  нельзя точно показать, *где именно* реализован ответ.

Метод класса кодируется как `ClassName.method_name`, чтобы `create` (функция) и
`CRUDRepository.create` (метод) были **разными** фрагментами — это критично для
точного совпадения с эталонными `chunk_id`.

**Ключевой приём для точности:** мы эмбеддим не голый код, а *обогащённое
представление* — путь + тип + имя + сигнатуру + docstring + код. Имена и docstring
несут максимум естественно-языкового сигнала и заметно повышают Precision@5 на
запросах вроде «как создаётся токен доступа».

### Выбор модели эмбеддингов

По умолчанию — **`intfloat/multilingual-e5-large`**: сильная многоязычная модель,
которая хорошо строит *общее* пространство для русского запроса и английского кода
(кросс-язычный поиск — главная сложность этого датасета: 8 из 15 вопросов на русском,
а код на английском). Альтернативы переключаются флагом `--model` или env
`CODELENS_MODEL`:

| Модель | Размер | Когда брать |
|--------|--------|-------------|
| `intfloat/multilingual-e5-large` *(default)* | ~2.2 ГБ | лучший баланс качество/скорость |
| `BAAI/bge-m3` | ~2.3 ГБ | сильнейшая многоязычная, длинный контекст |
| `intfloat/multilingual-e5-base` | ~1.1 ГБ | слабее железо |
| `paraphrase-multilingual-MiniLM-L12-v2` | ~0.5 ГБ | рекомендация ТЗ, самая лёгкая |

> Имя модели сохраняется в метаданных коллекции ChromaDB, и `app.py` подхватывает его
> автоматически — это исключает классическую ошибку «проиндексировал одной моделью,
> ищу другой».

### Выбор векторной БД: **ChromaDB**

ChromaDB ставится одним `pip install`, работает из коробки без сервера, хранит вектор +
метаданные и ищет ближайших соседей по косинусу через HNSW. Для прототипа на ~150–200
фрагментов этого с запасом достаточно. FAISS быстрее на больших объёмах, но требует
вручную хранить метаданные рядом — для нашего размера абстракция ChromaDB удобнее и не
влияет на латентность (поиск по сотням векторов — единицы миллисекунд; основное время —
эмбеддинг запроса).

---

## 5 примеров запросов и ответов

Запросы выполняются по кодовой базе `gymhero`. Ниже — какой фрагмент система должна
поднять в топ и где он лежит.

1. **«как в проекте создаётся токен доступа и какой срок его жизни?»** (ru)
   → `gymhero/security.py:create_access_token` (формирование JWT) +
   `gymhero/config.py:Settings` (поле `ACCESS_TOKEN_EXPIRE_MINUTES`).

2. **“how does the project verify a JWT token from an incoming request?”** (en)
   → `gymhero/api/dependencies.py:get_token` → `get_current_user` +
   `gymhero/exceptions.py:_get_credential_exception` (исключение при невалидном токене).

3. **«где проверяется, является ли пользователь суперпользователем?»** (ru)
   → `gymhero/crud/user.py:UserCRUDRepository.is_super_user` +
   `gymhero/api/dependencies.py:get_current_superuser`.

4. **“what database session management strategy is used for FastAPI endpoints?”** (en)
   → `gymhero/database/db.py:get_db` (зависимость-генератор сессии) +
   `gymhero/database/session.py:get_local_session`.

5. **«как реализована пагинация запросов к базе и где параметры по умолчанию?»** (ru)
   → `gymhero/api/dependencies.py:get_pagination_params` +
   `gymhero/crud/base.py:CRUDRepository.get_many`.

20 запросов для демо (включая 3 про несуществующую функциональность — blockchain,
WebSocket, rate limiting) лежат в `data/sample_queries.txt`.

---

## Измерение качества (Precision@5)

```bash
# прогнать тестовый набор, собрать results.json и сразу посчитать метрику
python evaluate.py
```

Скрипт прогоняет 15 вопросов через ретривер, пишет `results.json` в формате скорера и
запускает официальный `score.py`. То же доступно в UI на странице **«Метрики
(Precision@5)»** с кнопкой скачать `results.json`.

Ручной прогон официального скорера:

```bash
python score.py --predictions results.json --questions data/eval_questions.json
```

Для подбора режима поиска:

```bash
python evaluate.py --no-hybrid                 # только вектор
python evaluate.py --hybrid --alpha 0.8        # гибрид с весом вектора 0.8
```

---

## Дополнительные возможности (плюс к оценке)

- **LLM-ответы** — страница «Чат (LLM)»: фрагменты + вопрос → связный ответ
  (Ollama `mistral:7b` локально или OpenAI). См. `codelens/llm.py`.
- **Гибридный поиск** — вектор + BM25 с настраиваемым весом α (слайдер в UI).
- **Метрики в UI** — Precision@5 считается и показывается на отдельной странице.
- **Реранкинг** кросс-энкодером — включается env `CODELENS_RERANKER=BAAI/bge-reranker-v2-m3`.
- **Docker Compose** — `docker compose up` распакует базу, проиндексирует и поднимет UI.

### Запуск LLM-режима

```bash
ollama serve &
ollama pull mistral:7b
CODELENS_LLM_BACKEND=ollama streamlit run app.py
```

### Запуск в Docker

```bash
docker compose up --build      # http://localhost:8501
```

---

## Структура проекта

```
codelens/
├── index.py              # CLI индексирования:  python index.py gymhero
├── app.py                # Streamlit UI:        streamlit run app.py
├── evaluate.py           # прогон + Precision@5 через score.py
├── score.py              # официальный скорер (из датасета)
├── codelens/
│   ├── chunker.py        # ast → функции/классы/методы → chunk_id
│   ├── embedder.py       # sentence-transformers (e5/bge-префиксы)
│   ├── store.py          # ChromaDB (косинус, метаданные, имя модели)
│   ├── retriever.py      # dense + BM25-гибрид + реранк
│   ├── llm.py            # Ollama / OpenAI (опц.)
│   └── config.py         # настройки + env-переопределения
├── data/
│   ├── codebase_python.zip
│   ├── eval_questions.json
│   └── sample_queries.txt
├── requirements.txt
├── Dockerfile · docker-compose.yml · docker-entrypoint.sh
└── README.md
```

---

## Технические требования к формату

- Идентификатор фрагмента: `{relative_path}:{name}:{start_line}`, прямые слэши,
  методы — `ClassName.method_name`, `start_line` = `node.lineno` из `ast`.
- При сравнении со скорером допускается отклонение `start_line` на ±2 строки.
- `results.json` — массив из 15 объектов `{question_id, top_5_chunks}` (ровно 5 id).

*Кодовая база `gymhero` распространяется по лицензии MIT (см. `gymhero/LICENSE`).*
