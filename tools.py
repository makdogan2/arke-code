"""Tools Arke Code can call through Ollama's tool calling.

Phase 2 starts read-only: the model can look around the project but cannot change
anything. Every path is resolved inside the workspace root, so the model cannot read
files outside the folder the assistant was started in.
"""

from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"}
MAX_ENTRIES = 200
MAX_LINES = 400
MAX_CHARS = 30_000


class ToolError(Exception):
    """An error the model should see and recover from (bad path, missing file...)."""


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def resolve(self, path):
        target = (self.root / (path or ".")).resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"'{path}' is outside the workspace; use paths relative to the project root")
        return target

    def show(self, target):
        rel = target.relative_to(self.root).as_posix()
        return rel if rel != "." else "./"

    # ------------------------------------------------------------------ tools

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

    # --------------------------------------------------------------- dispatch

    def call(self, name, args):
        """Run a tool by name. Errors come back as text so the model can react."""
        funcs = {"list_dir": self.list_dir, "read_file": self.read_file}
        if name not in funcs:
            return f"Error: unknown tool '{name}'"
        try:
            return funcs[name](**(args or {}))
        except ToolError as e:
            return f"Error: {e}"
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and folders in the user's project. Use it to find out what "
                           "exists before reading files. Paths are relative to the project root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Folder to list, default '.'"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the user's project, with line numbers. Use it "
                           "whenever the user asks about their code instead of guessing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the project root"},
                    "start_line": {"type": "integer", "description": "First line to read, default 1"},
                    "end_line": {"type": "integer", "description": "Last line to read (optional)"},
                },
                "required": ["path"],
            },
        },
    },
]
