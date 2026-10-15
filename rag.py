"""Project memory for Arke Code: find code by meaning, not by exact words.

    python rag.py index                  # embed every .py / .md file in the project
    python rag.py search "your question" # show the 5 closest pieces of code
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


def build_index():
    chunks = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if any(part in SKIP for part in rel.parts) or p.suffix not in (".py", ".md"):
            continue
        text = p.read_text(encoding="utf-8")
        try:
            pieces = chunk_python(text) if p.suffix == ".py" else chunk_text(text)
        except SyntaxError:
            pieces = chunk_text(text)
        for start, end, body in pieces:
            chunks.append({"path": rel.as_posix(), "start": start, "end": end, "text": body})

    print(f"Embedding {len(chunks)} chunks from the project...")
    for i in range(0, len(chunks), 32):
        batch = chunks[i:i + 32]
        docs = [f"search_document: {c['path']} lines {c['start']}-{c['end']}\n{c['text'][:2000]}"
                for c in batch]
        for c, v in zip(batch, embed(docs)):
            c["vector"] = normalize(v)
    INDEX.write_text(json.dumps({"model": MODEL, "chunks": chunks}), encoding="utf-8")
    print(f"Saved {INDEX.name}")


def search(query, k=5):
    if not INDEX.exists():
        sys.exit("No index yet. Run: python rag.py index")
    chunks = json.loads(INDEX.read_text(encoding="utf-8"))["chunks"]
    q = normalize(embed(["search_query: " + query])[0])
    # both vectors have length 1, so the dot product is the cosine similarity
    scored = sorted(((sum(a * b for a, b in zip(q, c["vector"])), c) for c in chunks),
                    key=lambda pair: pair[0], reverse=True)
    return scored[:k]


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "index":
        build_index()
    elif len(sys.argv) >= 3 and sys.argv[1] == "search":
        for score, c in search(" ".join(sys.argv[2:])):
            first = c["text"].strip().splitlines()[0][:70]
            print(f"{score:.3f}  {c['path']}:{c['start']}-{c['end']}  {first}")
    else:
        print(__doc__)
