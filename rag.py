"""Project memory for Arke Code: find code by meaning, not by exact words.

    python rag.py index                  # embed every .py / .md file in the project
    python rag.py search "your question" # show the 5 closest pieces of code

Searching also refreshes the index: only files that changed since the last run are
embedded again. Arke Code uses the same functions through its search_code tool.
"""

import ast
import json
import math
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).parent
INDEX = ROOT / ".arke_index.json"
URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
SKIP = {".git", ".venv", "venv", "__pycache__", "logs", "data", "results", "evals"}
WINDOW = 40  # lines per chunk for files that are not Python


def embed(texts):
    r = requests.post(URL, json={"model": MODEL, "input": texts}, timeout=300)
    r.raise_for_status()
    return r.json()["embeddings"]


def normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def chunk_python(text):
    """One chunk per function, one per method (with its class name in front),
    and the remaining module code (imports, constants) in runs of lines."""
    lines = text.splitlines()
    chunks, used = [], set()

    def add(start, end, prefix=""):
        chunks.append((start + 1, end, prefix + "\n".join(lines[start:end])))
        used.update(range(start, end))

    for node in ast.parse(text).body:
        first = node.lineno - 1 - len(getattr(node, "decorator_list", []))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(first, node.end_lineno)
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            header_end = (methods[0].lineno - 1 - len(methods[0].decorator_list)) if methods else node.end_lineno
            add(first, header_end)
            for m in methods:
                add(m.lineno - 1 - len(m.decorator_list), m.end_lineno, f"class {node.name}:\n")

    # leftover module-level code (docstring, imports, constants), in runs of lines
    run = []
    for i in range(len(lines) + 1):
        if i < len(lines) and i not in used and len(run) < WINDOW:
            run.append(i)
            continue
        body = "\n".join(lines[run[0]:run[-1] + 1]).strip() if run else ""
        if any(l.strip() and not l.strip().startswith("#") for l in body.splitlines()):
            chunks.append((run[0] + 1, run[-1] + 1, body))
        run = [i] if i < len(lines) and i not in used else []
    return chunks


def chunk_text(text):
    lines = text.splitlines()
    return [(i + 1, min(i + WINDOW, len(lines)), "\n".join(lines[i:i + WINDOW]))
            for i in range(0, len(lines), WINDOW)]


def project_files(root):
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and p.suffix in (".py", ".md") and not any(part in SKIP for part in rel.parts):
            yield p, rel.as_posix()


def chunk_file(path):
    text = path.read_text(encoding="utf-8-sig", errors="replace")  # -sig drops a BOM
    try:
        return chunk_python(text) if path.suffix == ".py" else chunk_text(text)
    except SyntaxError:
        return chunk_text(text)


def update_index(root=ROOT, verbose=False):
    """Bring root/.arke_index.json up to date and return its chunks.
    Files whose size and modification time did not change keep their old vectors,
    so only new or edited files are embedded again."""
    root = Path(root).resolve()
    index_path = root / INDEX.name
    old = {}
    if index_path.exists():
        saved = json.loads(index_path.read_text(encoding="utf-8"))
        if saved.get("model") == MODEL:
            for c in saved["chunks"]:
                old.setdefault(c["path"], []).append(c)

    chunks, todo, reused = [], [], 0
    for path, rel in project_files(root):
        stat = path.stat()
        stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
        previous = old.get(rel, [])
        if previous and previous[0].get("stamp") == stamp:
            chunks += previous
            reused += 1
            continue
        for start, end, body in chunk_file(path):
            c = {"path": rel, "start": start, "end": end, "text": body, "stamp": stamp}
            chunks.append(c)
            todo.append(c)

    for i in range(0, len(todo), 32):
        batch = todo[i:i + 32]
        docs = [f"search_document: {c['path']} lines {c['start']}-{c['end']}\n{c['text'][:2000]}"
                for c in batch]
        for c, v in zip(batch, embed(docs)):
            c["vector"] = normalize(v)

    if todo or len(chunks) != sum(len(v) for v in old.values()):
        index_path.write_text(json.dumps({"model": MODEL, "chunks": chunks}), encoding="utf-8")
    if verbose:
        print(f"{len(chunks)} chunks: {len(todo)} embedded now, {reused} files unchanged -> {index_path.name}")
    return chunks


def search(query, k=5, root=ROOT):
    """Return [(score, chunk), ...] for the k chunks closest in meaning to the query."""
    chunks = update_index(root)
    if not chunks:
        return []
    q = normalize(embed(["search_query: " + query])[0])
    # both vectors have length 1, so the dot product is the cosine similarity
    scored = sorted(((sum(a * b for a, b in zip(q, c["vector"])), c) for c in chunks),
                    key=lambda pair: pair[0], reverse=True)
    return scored[:k]


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "index":
        update_index(verbose=True)
    elif len(sys.argv) >= 3 and sys.argv[1] == "search":
        for score, c in search(" ".join(sys.argv[2:])):
            first = c["text"].strip().splitlines()[0][:70]
            print(f"{score:.3f}  {c['path']}:{c['start']}-{c['end']}  {first}")
    else:
        print(__doc__)
