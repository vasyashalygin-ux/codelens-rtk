"""Нарезка Python-файлов на смысловые фрагменты (chunks) через модуль `ast`.

Стратегия чанкования: один chunk = одна функция, один класс или один метод
класса. Это минимальная самодостаточная семантическая единица Python-кода:
у неё есть имя, сигнатура и (часто) docstring, описывающие назначение на
естественном языке — именно то, по чему ищет пользователь.

Формат идентификатора фрагмента строго соответствует спецификации датасета:
    {relative_path}:{name}:{start_line}
где для метода класса name = "ClassName.method_name", а start_line — это
node.lineno (строка с `def`/`class`), как возвращает ast.parse.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str          # gymhero/security.py:create_access_token:12
    rel_path: str          # gymhero/security.py
    name: str              # create_access_token  или  CRUDRepository.get_many
    kind: str              # "function" | "class" | "method"
    start_line: int        # номер строки def/class (1-based, как в ast)
    end_line: int
    docstring: str
    code: str              # исходный текст фрагмента
    embed_text: str = field(default="")  # обогащённый текст, который реально эмбеддится


def _signature(node: ast.AST) -> str:
    """Восстанавливает строку-сигнатуру функции/метода для обогащения эмбеддинга."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        try:
            args = ast.unparse(node.args)
        except Exception:
            args = ""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({args})"
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    return ""


def _build_embed_text(rel_path: str, kind: str, qualname: str,
                      signature: str, docstring: str, code: str) -> str:
    """Собирает текст для эмбеддинга.

    Мы эмбеддим НЕ голый код, а обогащённое представление: путь + тип + имя +
    сигнатура + docstring + код. Имена функций (camelCase/snake_case) и docstring
    несут больше всего «естественно-языкового» сигнала и резко повышают
    попадание векторного поиска по запросам вроде «как создаётся токен доступа».
    """
    header = f"File: {rel_path}\n{kind.capitalize()}: {qualname}"
    parts = [header]
    if signature:
        parts.append(f"Signature: {signature}")
    if docstring:
        parts.append(f"Docstring: {docstring.strip()}")
    parts.append("Code:\n" + code)
    return "\n".join(parts)


def _src_segment(lines: list[str], node: ast.AST) -> str:
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    return "\n".join(lines[start:end])


def chunk_file(py_file: Path, repo_root: Path) -> list[Chunk]:
    """Возвращает список Chunk для одного .py-файла."""
    rel = py_file.relative_to(repo_root).as_posix()
    try:
        src = py_file.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []  # битый/несовместимый файл — пропускаем, индексирование не падает

    lines = src.splitlines()
    chunks: list[Chunk] = []

    def make_chunk(node, name, kind):
        code = _src_segment(lines, node)
        doc = ast.get_docstring(node) or ""
        sig = _signature(node)
        chunk_id = f"{rel}:{name}:{node.lineno}"
        chunks.append(Chunk(
            chunk_id=chunk_id, rel_path=rel, name=name, kind=kind,
            start_line=node.lineno, end_line=getattr(node, "end_lineno", node.lineno),
            docstring=doc, code=code,
            embed_text=_build_embed_text(rel, kind, name, sig, doc, code),
        ))

    # Обходим ТОЛЬКО верхний уровень модуля, чтобы методы не дублировались как
    # самостоятельные функции. Методы добавляем явно из тела класса.
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            make_chunk(node, node.name, "class")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    make_chunk(item, f"{node.name}.{item.name}", "method")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            make_chunk(node, node.name, "function")

    return chunks


def chunk_repo(repo_root: Path) -> list[Chunk]:
    """Обходит директорию рекурсивно и нарезает все .py-файлы."""
    repo_root = Path(repo_root)
    all_chunks: list[Chunk] = []
    for py_file in sorted(repo_root.rglob("*.py")):
        all_chunks.extend(chunk_file(py_file, repo_root))
    return all_chunks
