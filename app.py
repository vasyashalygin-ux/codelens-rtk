#!/usr/bin/env python3
"""CodeLens — веб-интерфейс (Streamlit).

Запуск:  streamlit run app.py
Перед этим база должна быть проиндексирована:  python index.py gymhero

Страницы:
  • Поиск      — семантический поиск с подсветкой кода и оценкой релевантности
  • Метрики    — живой подсчёт Precision@5 на eval_questions.json (бонус)
  • Чат (LLM)  — связный ответ на основе найденных фрагментов (бонус, опц.)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from codelens import config
from codelens.retriever import Retriever
from codelens.store import VectorStore

st.set_page_config(page_title="CodeLens — умный поиск по коду",
                   page_icon="🔎", layout="wide")


# --- встроенный скорер (та же формула ±2 строки, что и в score.py) ---------
def parse_chunk_id(cid: str):
    parts = cid.rsplit(":", 2)
    if len(parts) != 3:
        return None
    try:
        return parts[0], parts[1], int(parts[2])
    except ValueError:
        return None


def chunks_match(pred: str, ref: str, tol: int = 2) -> bool:
    p, r = parse_chunk_id(pred), parse_chunk_id(ref)
    if not p or not r:
        return False
    return p[0] == r[0] and p[1] == r[1] and abs(p[2] - r[2]) <= tol


def score_question(top5, correct) -> float:
    seen, used, matched = [], set(), 0
    for c in top5:
        if c not in seen:
            seen.append(c)
    for pred in seen:
        for i, ref in enumerate(correct):
            if i not in used and chunks_match(pred, ref):
                matched += 1
                used.add(i)
                break
    return matched / min(5, len(correct))


@st.cache_resource(show_spinner="Загружаю индекс и модель эмбеддингов…")
def load_retriever():
    store = VectorStore()
    if not store.exists() or store.count() == 0:
        return None
    return Retriever(store)


def render_result(rank: int, hit: dict):
    m = hit["meta"]
    with st.container(border=True):
        c1, c2 = st.columns([0.8, 0.2])
        with c1:
            st.markdown(
                f"**{rank}. `{m.get('rel_path')}`** — "
                f"{m.get('kind')} **{m.get('name')}** "
                f"(строки {m.get('start_line')}–{m.get('end_line')})")
            st.caption(f"`{hit['chunk_id']}`")
        with c2:
            st.metric("Релевантность", f"{hit['relevance_pct']}%")
        if m.get("docstring"):
            with st.expander("Docstring"):
                st.write(m["docstring"])
        st.code(hit["code"], language="python")  # подсветка синтаксиса


# ---------------------------------------------------------------------------
st.title("🔎 CodeLens — умный поиск по кодовой базе")
retriever = load_retriever()

with st.sidebar:
    st.header("Настройки")
    page = st.radio("Раздел", ["Поиск", "Метрики (Precision@5)", "Чат (LLM)"])
    st.divider()
    top_k = st.slider("Сколько результатов (top-K)", 1, 10, config.TOP_K)
    use_hybrid = st.toggle("Гибридный поиск (вектор + BM25)", value=config.USE_HYBRID)
    alpha = st.slider("Вес векторного поиска (α)", 0.0, 1.0, config.HYBRID_ALPHA, 0.05,
                      disabled=not use_hybrid,
                      help="1.0 — только вектор, 0.0 — только BM25")
    rerank = st.toggle("Реранкинг кросс-энкодером",
                       value=bool(config.RERANKER_MODEL),
                       disabled=not config.RERANKER_MODEL,
                       help="Доступно, если задан CODELENS_RERANKER")
    if retriever is not None:
        st.divider()
        st.caption(f"Модель: `{retriever.model_name}`")
        st.caption(f"Фрагментов в базе: **{retriever.store.count()}**")

if retriever is None:
    st.warning("База не проиндексирована. Выполните в терминале:\n\n"
               "```bash\npython index.py gymhero\n```")
    st.stop()


# === Страница: Поиск =======================================================
if page == "Поиск":
    st.write("Задайте вопрос о коде на русском или английском — "
             "поиск идёт по смыслу, а не по совпадению слов.")
    query = st.text_input(
        "Запрос",
        placeholder="как создаётся токен доступа и какой у него срок жизни?")
    cols = st.columns([0.15, 0.85])
    go = cols[0].button("Искать", type="primary")
    examples = [
        "где хранится секретный ключ для подписи токенов?",
        "how to create a new training plan?",
        "как реализована пагинация запросов к базе?",
    ]
    ex = cols[1].selectbox("…или выберите пример", [""] + examples,
                           label_visibility="collapsed")
    if ex:
        query, go = ex, True

    if go and query.strip():
        t0 = time.time()
        hits = retriever.search(query, top_k=top_k, use_hybrid=use_hybrid,
                                alpha=alpha, rerank=rerank)
        st.caption(f"Найдено {len(hits)} фрагментов за {time.time()-t0:.2f}с")
        for i, h in enumerate(hits, 1):
            render_result(i, h)


# === Страница: Метрики =====================================================
elif page == "Метрики (Precision@5)":
    st.subheader("Precision@5 на тестовом наборе eval_questions.json")
    qpath = Path("data/eval_questions.json")
    if not qpath.exists():
        st.error("Не найден data/eval_questions.json")
        st.stop()
    questions = json.loads(qpath.read_text(encoding="utf-8"))

    if st.button("Прогнать оценку", type="primary"):
        rows, latencies, results, total = [], [], [], 0.0
        bar = st.progress(0.0, text="Прогон вопросов…")
        for idx, q in enumerate(questions):
            t0 = time.time()
            hits = retriever.search(q["query"], top_k=5, use_hybrid=use_hybrid,
                                    alpha=alpha, rerank=rerank)
            latencies.append(time.time() - t0)
            pred = [h["chunk_id"] for h in hits]
            score = score_question(pred, q["correct_chunk_ids"])
            total += score
            rows.append({"question_id": q["question_id"], "lang": q["language"],
                         "difficulty": q["difficulty"], "score": round(score, 2),
                         "query": q["query"]})
            results.append({"question_id": q["question_id"], "top_5_chunks": pred})
            bar.progress((idx + 1) / len(questions), text=f"{idx+1}/{len(questions)}")

        mean = total / len(questions)
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Precision@5", f"{mean:.3f}")
        c2.metric("Средняя задержка", f"{sum(latencies)/len(latencies):.2f}с")
        c3.metric("Макс. задержка", f"{max(latencies):.2f}с")
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button("Скачать results.json",
                           data=json.dumps(results, ensure_ascii=False, indent=2),
                           file_name="results.json", mime="application/json")


# === Страница: Чат (LLM) ===================================================
elif page == "Чат (LLM)":
    st.subheader("Связный ответ на основе найденных фрагментов (полный RAG)")
    from codelens import llm
    backend = st.selectbox("LLM-бэкенд", ["ollama", "openai", "none"],
                           index=["ollama", "openai", "none"].index(config.LLM_BACKEND))
    if backend != "none" and not llm.llm_available(backend):
        st.info(f"Бэкенд `{backend}` недоступен. "
                f"Ollama: `ollama serve` + `ollama pull {config.OLLAMA_MODEL}`. "
                f"OpenAI: задайте OPENAI_API_KEY.")
    q = st.text_input("Вопрос о коде")
    if st.button("Спросить", type="primary") and q.strip():
        hits = retriever.search(q, top_k=top_k, use_hybrid=use_hybrid, alpha=alpha)
        with st.spinner("Генерирую ответ…"):
            try:
                answer = llm.generate_answer(q, hits, backend=backend)
            except Exception as e:
                answer = f"Не удалось получить ответ от LLM: {e}"
        st.markdown("### Ответ")
        st.markdown(answer)
        st.markdown("### Использованные фрагменты")
        for i, h in enumerate(hits, 1):
            render_result(i, h)
