"""Tools Arke Code can call through Ollama's tool calling.

Two groups:

- Read-only (run immediately): list_dir, read_file, grep, git_diff.
- Changing or executing (need the user's yes first): edit_file, write_file,
  run_python, run_tests. Before asking, the user sees exactly what will happen:
  a diff for file changes, the command for runs.

Every path is resolved inside the workspace root, so the model can neither read nor
write outside the folder the assistant was started in.
"""

import difflib
import os
import re
import subprocess
import sys
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"}
PROTECTED = {".git", ".venv", "venv"}  # never written to
MAX_ENTRIES = 200
MAX_LINES = 400
MAX_CHARS = 30_000
MAX_OUTPUT = 8_000
MAX_MATCHES = 100
RUN_TIMEOUT = 60


class ToolError(Exception):
    """An error the model should see and recover from (bad path, missing file...)."""


def ask_user(summary, preview):
    """Default confirmation: show what will happen, accept only an explicit yes."""
    print(f"\n  [confirm] {summary}")
    if preview:
        print("\n".join("    " + line for line in preview.splitlines()))
    answer = input("  Allow? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def read_keep_eol(path):
    """Read text with '\n' line endings, remembering the file's own (LF or CRLF)."""
    with open(path, encoding="utf-8", newline="") as f:
        raw = f.read()
    return raw.replace("\r\n", "\n"), ("\r\n" if "\r\n" in raw else "\n")


def write_keep_eol(path, text, eol="\n"):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\r\n", "\n").replace("\n", eol))
    # A same-size edit within the same second can leave a stale .pyc that Python
    # still trusts, so the next run would test the old code. Drop it.
    for pyc in (Path(path).parent / "__pycache__").glob(f"{Path(path).stem}.*.pyc"):
        pyc.unlink(missing_ok=True)


def clip(text, limit=MAX_OUTPUT):
    if len(text) <= limit:
        return text
    return text[: limit // 2] + f"\n... [{len(text) - limit} characters cut] ...\n" + text[-limit // 2:]


class Workspace:
    def __init__(self, root, confirm=ask_user):
        self.root = Path(root).resolve()
        self.confirm = confirm

    # ---------------------------------------------------------------- helpers

    def resolve(self, path):
        target = (self.root / (path or ".")).resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"'{path}' is outside the workspace; use paths relative to the project root")
        return target

    def show(self, target):
        rel = target.relative_to(self.root).as_posix()
        return rel if rel != "." else "./"

    def writable(self, path):
        target = self.resolve(path)
        parts = target.relative_to(self.root).parts
        if parts and parts[0] in PROTECTED:
            raise ToolError(f"'{path}' is inside a protected folder ({parts[0]})")
        if target.is_dir():
            raise ToolError(f"'{path}' is a directory")
        return target

    def text_files(self, base):
        for p in sorted(base.rglob("*")):
            if any(part in SKIP_DIRS for part in p.relative_to(self.root).parts):
                continue
            if p.is_file() and p.stat().st_size < 2_000_000:
                yield p

    def diff(self, target, old, new):
        name = self.show(target)
        lines = difflib.unified_diff(old.splitlines(), new.splitlines(),
                                     f"a/{name}", f"b/{name}", lineterm="")
        return clip("\n".join(lines), 4_000)

    def run(self, cmd, summary):
        if not self.confirm(summary, "$ " + " ".join(cmd)):
            return "The user declined to run this."
        try:
            proc = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True,
                                  timeout=RUN_TIMEOUT,
                                  env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        except subprocess.TimeoutExpired:
            return f"Timed out after {RUN_TIMEOUT}s."
        out = (proc.stdout + ("\n" + proc.stderr if proc.stderr else "")).strip()
        return f"Exit code {proc.returncode}\n{clip(out) or '(no output)'}"

    # ------------------------------------------------------- read-only tools

    def list_dir(self, path="."):
        target = self.resolve(path)
        if not target.is_dir():
            raise ToolError(f"'{path}' is not a directory")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        lines = []
        for p in entries:
            if p.name in SKIP_DIRS:
                continue
            if p.is_dir():
                lines.append(f"{p.name}/")
            else:
                lines.append(f"{p.name}  ({p.stat().st_size:,} bytes)")
        if not lines:
            return f"{self.show(target)} is empty"
        extra = len(lines) - MAX_ENTRIES
        lines = lines[:MAX_ENTRIES] + ([f"... and {extra} more"] if extra > 0 else [])
        return f"Contents of {self.show(target)}:\n" + "\n".join(lines)

    def read_file(self, path, start_line=1, end_line=None):
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"'{path}' is not a file (use list_dir to see what exists)")
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"'{path}' is not a UTF-8 text file") from None
        lines = text.splitlines()
        start = max(1, int(start_line or 1))
        end = min(len(lines), int(end_line) if end_line else start + MAX_LINES - 1)
        end = min(end, start + MAX_LINES - 1)
        body, size = [], 0
        for n in range(start, end + 1):
            row = f"{n:>5}| {lines[n - 1]}"
            size += len(row) + 1
            if size > MAX_CHARS:
                end = n - 1
                break
            body.append(row)
        header = f"{self.show(target)} (lines {start}-{end} of {len(lines)})"
        more = (f"\n[truncated: call read_file with start_line={end + 1} to continue]"
                if end < len(lines) else "")
        return header + "\n" + "\n".join(body) + more

    def grep(self, pattern, path="."):
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"invalid regex: {e}") from None
        base = self.resolve(path)
        files = [base] if base.is_file() else self.text_files(base)
        hits = []
        for f in files:
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for n, line in enumerate(lines, 1):
                if rx.search(line):
                    hits.append(f"{self.show(f)}:{n}: {line.strip()[:200]}")
                    if len(hits) >= MAX_MATCHES:
                        return "\n".join(hits) + f"\n[stopped at {MAX_MATCHES} matches]"
        return "\n".join(hits) if hits else f"No matches for {pattern!r}"

    def git_diff(self, path=""):
        cmd = ["git", "diff", "--stat", "-p"] + ([str(self.resolve(path))] if path else [])
        try:
            proc = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=20)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise ToolError(f"git diff failed: {e}") from None
        if proc.returncode != 0:
            raise ToolError(proc.stderr.strip() or "not a git repository")
        return clip(proc.stdout.strip()) or "No uncommitted changes."

    # -------------------------------------------- tools that need permission

    def edit_file(self, path, old_text, new_text):
        target = self.writable(path)
        if not target.is_file():
            raise ToolError(f"'{path}' does not exist; use write_file to create it")
        try:
            text, eol = read_keep_eol(target)
        except UnicodeDecodeError:
            raise ToolError(f"'{path}' is not a UTF-8 text file") from None
        old_text, new_text = old_text.replace("\r\n", "\n"), new_text.replace("\r\n", "\n")
        count = text.count(old_text) if old_text else 0
        if count == 0:
            raise ToolError("old_text was not found; read_file the file and copy the text exactly")
        if count > 1:
            raise ToolError(f"old_text appears {count} times; include more surrounding lines")
        new = text.replace(old_text, new_text, 1)
        if not self.confirm(f"edit {self.show(target)}", self.diff(target, text, new)):
            return "The user declined this edit. Ask what they would like instead."
        write_keep_eol(target, new, eol)
        return f"Edited {self.show(target)}."

    def write_file(self, path, content):
        target = self.writable(path)
        old, eol = read_keep_eol(target) if target.is_file() else (None, "\n")
        if old is None:
            preview = clip(content, 2_000)
            summary = f"create {self.show(target)} ({len(content.splitlines())} lines)"
        else:
            preview = self.diff(target, old, content)
            summary = f"overwrite {self.show(target)}"
        if not self.confirm(summary, preview):
            return "The user declined this write. Ask what they would like instead."
        target.parent.mkdir(parents=True, exist_ok=True)
        write_keep_eol(target, content, eol)
        return f"Wrote {self.show(target)} ({len(content.splitlines())} lines)."

    def run_python(self, path, args=None):
        target = self.resolve(path)
        if not target.is_file() or target.suffix != ".py":
            raise ToolError(f"'{path}' is not a Python file")
        extra = [str(a) for a in (args or [])]
        return self.run([sys.executable, str(target.relative_to(self.root))] + extra,
                        f"run {self.show(target)}")

    def run_tests(self, path=""):
        target = self.resolve(path) if path else self.root
        rel = str(target.relative_to(self.root)) if target != self.root else "."
        try:
            import pytest  # noqa: F401
            cmd = [sys.executable, "-m", "pytest", "-q", rel]
        except ImportError:
            if target.is_file():
                cmd = [sys.executable, rel]
            else:
                cmd = [sys.executable, "-m", "unittest", "discover", "-s", rel]
        return self.run(cmd, f"run tests in {self.show(target)}")

    # --------------------------------------------------------------- dispatch

    def call(self, name, args):
        """Run a tool by name. Errors come back as text so the model can react."""
        funcs = {
            "list_dir": self.list_dir, "read_file": self.read_file, "grep": self.grep,
            "git_diff": self.git_diff, "edit_file": self.edit_file, "write_file": self.write_file,
            "run_python": self.run_python, "run_tests": self.run_tests,
        }
        if name not in funcs:
            return f"Error: unknown tool '{name}'"
        try:
            return funcs[name](**(args or {}))
        except ToolError as e:
            return f"Error: {e}"
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"


def _spec(name, description, properties, required=()):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(required)}}}


TOOL_SPECS = [
    _spec("list_dir",
          "List files and folders in the user's project. Use it to find out what exists before "
          "reading files. Paths are relative to the project root.",
          {"path": {"type": "string", "description": "Folder to list, default '.'"}}),
    _spec("read_file",
          "Read a text file from the user's project, with line numbers. Use it whenever the user "
          "asks about their code instead of guessing. Always read a file before editing it.",
          {"path": {"type": "string", "description": "File path relative to the project root"},
           "start_line": {"type": "integer", "description": "First line to read, default 1"},
           "end_line": {"type": "integer", "description": "Last line to read (optional)"}},
          ["path"]),
    _spec("grep",
          "Search the project for a regular expression. Returns file:line: text for each match. "
          "Use it to find where a function, class or string is used.",
          {"pattern": {"type": "string", "description": "Python regular expression"},
           "path": {"type": "string", "description": "File or folder to search, default '.'"}},
          ["pattern"]),
    _spec("git_diff",
          "Show uncommitted changes in the project (git diff).",
          {"path": {"type": "string", "description": "Limit to this file or folder (optional)"}}),
    _spec("edit_file",
          "Replace one exact piece of text in an existing file. Preferred for small changes. "
          "old_text must match the file exactly once, including indentation; read the file first. "
          "The user sees the diff and must approve it.",
          {"path": {"type": "string"},
           "old_text": {"type": "string", "description": "Exact text to replace"},
           "new_text": {"type": "string", "description": "Replacement text"}},
          ["path", "old_text", "new_text"]),
    _spec("write_file",
          "Create a new file, or overwrite a whole file. For small changes to an existing file "
          "use edit_file instead. The user must approve it.",
          {"path": {"type": "string"},
           "content": {"type": "string", "description": "Full file content"}},
          ["path", "content"]),
    _spec("run_python",
          "Run a Python file from the project and return its output. The user must approve it.",
          {"path": {"type": "string"},
           "args": {"type": "array", "items": {"type": "string"},
                    "description": "Command-line arguments (optional)"}},
          ["path"]),
    _spec("run_tests",
          "Run the project's tests (pytest if installed, otherwise unittest) and return the "
          "output. Use it after changing code. The user must approve it.",
          {"path": {"type": "string", "description": "Test file or folder (optional)"}}),
]
