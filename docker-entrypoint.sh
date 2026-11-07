#!/usr/bin/env bash
set -e

# 1) распаковать кодовую базу, если папка gymhero ещё не создана
if [ ! -d "gymhero" ] && [ -f "data/codebase_python.zip" ]; then
  echo "Распаковываю data/codebase_python.zip ..."
  python - <<'PY'
import zipfile
zipfile.ZipFile("data/codebase_python.zip").extractall(".")
PY
fi

# 2) проиндексировать, если индекс пуст
if [ ! -d "$CODELENS_CHROMA_DIR" ]; then
  echo "Индексирую кодовую базу ..."
  python index.py gymhero
fi

# 3) поднять Streamlit
exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501
