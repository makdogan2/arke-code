"""Score Arke Code as an agent: can it fix a real bug on its own?

Each task in evals/agent_tasks/<id>/ is a tiny project with one planted bug and a
failing test. The agent gets a fresh copy, the same tools as assistant.py (every
action auto-approved, inside a throwaway folder) and one instruction. Afterwards the
task counts as solved only if:

  1. the visible tests pass,
  2. hidden tests it never saw also pass (so the fix is real, not test-shaped),
  3. it did not edit the test files.

Usage:
    python agent_eval.py --check              # prove every task is broken and fixable
    python agent_eval.py                      # run all tasks on arke-code
    python agent_eval.py --only median slugify --tag prompt-v2
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests

from tools import TOOL_SPECS, Workspace

ROOT = Path(__file__).parent
TASKS_DIR = ROOT / "evals" / "agent_tasks"
RESULTS = ROOT / "results"
OLLAMA_URL = "http://localhost:11434"
MAX_ROUNDS = 15
TASK_TIMEOUT = 600  # seconds of wall clock per task
INSTRUCTION = ("The tests in this project are failing. Find the bug and fix it. "
               "Do not modify the test files. When you are done, say briefly what you changed.")


# ---------------------------------------------------------------------- tasks

def load_tasks(only=None):
    tasks = []
    for d in sorted(TASKS_DIR.iterdir()):
        meta = d / "task.json"
        if meta.exists() and (not only or d.name in only):
            tasks.append({**json.loads(meta.read_text(encoding="utf-8")), "dir": d})
    return tasks


def copy_project(task, dest):
    shutil.copytree(task["dir"] / "project", dest, dirs_exist_ok=True)


def test_hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).glob("test*.py"))}


def run_unittest(folder, module=None):
    cmd = [sys.executable, "-B", "-m", "unittest"] + ([module] if module else ["discover", "-s", "."])
    try:
        proc = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    last = (proc.stderr.strip().splitlines() or ["?"])[-1]
    return proc.returncode == 0, last


def score(task, folder, original_hashes):
    if test_hashes(folder) != original_hashes:
        return False, "edited the test files"
    ok, last = run_unittest(folder)
    if not ok:
        return False, f"visible tests fail ({last})"
    shutil.copy(task["dir"] / "hidden_test.py", Path(folder) / "hidden_test.py")
    ok, last = run_unittest(folder, "hidden_test")
    if not ok:
        return False, f"hidden tests fail ({last})"
    return True, ""


# ---------------------------------------------------------------------- agent

def run_agent(model, folder, num_ctx):
    """The assistant.py tool loop, without printing or asking. Returns a log dict."""
    ws = Workspace(folder, confirm=lambda summary, preview: True)
    messages = [{"role": "user", "content": INSTRUCTION}]
    calls, started = [], time.time()
    for _ in range(MAX_ROUNDS):
        if time.time() - started > TASK_TIMEOUT:
            return {"stopped": "timeout", "calls": calls, "messages": messages}
        r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=TASK_TIMEOUT, json={
            "model": model, "messages": messages, "tools": TOOL_SPECS, "stream": False,
            "options": {"temperature": 0, "seed": 42, "num_ctx": num_ctx, "num_predict": 4096}})
        r.raise_for_status()
        msg = r.json()["message"]
        tool_calls = msg.get("tool_calls") or []
        entry = {"role": "assistant", "content": msg.get("content", "")}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        messages.append(entry)
        if not tool_calls:
            return {"stopped": "answered", "calls": calls, "messages": messages}
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {}
            result = ws.call(name, args)
            calls.append(name)
            messages.append({"role": "tool", "tool_name": name, "content": result})
    return {"stopped": "too many rounds", "calls": calls, "messages": messages}


# ----------------------------------------------------------------------- runs

def check():
    """Every task must fail as shipped and pass (visible + hidden) with its reference fix."""
    bad = 0
    for task in load_tasks():
        with tempfile.TemporaryDirectory() as tmp:
            copy_project(task, tmp)
            broken, _ = run_unittest(tmp)
            for f in task["fix"]:
                p = Path(tmp) / f["file"]
                text = p.read_text(encoding="utf-8")
                assert text.count(f["old"]) == 1, f"{task['id']}: fix does not apply"
                p.write_text(text.replace(f["old"], f["new"]), encoding="utf-8")
            fixed, why = score(task, tmp, test_hashes(task["dir"] / "project"))
        ok = (not broken) and fixed
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {task['id']:<14} "
              f"{'' if not broken else 'tests pass before the fix! '}{why}")
    print(f"\n{len(load_tasks()) - bad}/{len(load_tasks())} tasks are sound.")
    return bad == 0


def evaluate(model, only, num_ctx, tag):
    tasks = load_tasks(only)
    print(f"Agent eval: {model} on {len(tasks)} tasks\n")
    records = []
    for n, task in enumerate(tasks, 1):
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            copy_project(task, tmp)
            hashes = test_hashes(tmp)
            log = run_agent(model, tmp, num_ctx)
            ok, why = score(task, tmp, hashes)
        secs = time.time() - t0
        records.append({"id": task["id"], "solved": ok, "detail": why, "seconds": round(secs, 1),
                        "tool_calls": len(log["calls"]), "tools": log["calls"],
                        "stopped": log["stopped"], "messages": log["messages"]})
        print(f"[{n}/{len(tasks)}] {'PASS' if ok else 'FAIL'} {task['id']:<14} {secs:5.1f}s  "
              f"{len(log['calls']):>2} tool calls  {why}")

    solved = sum(r["solved"] for r in records)
    avg_calls = sum(r["tool_calls"] for r in records) / len(records) if records else 0
    label = f"{model}@{tag}" if tag else model
    print(f"\n{label}: {solved}/{len(records)} bugs fixed, {avg_calls:.1f} tool calls per task")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    safe = label.replace(":", "_").replace("/", "_").replace("@", "_")
    (RESULTS / f"agent_{safe}_{stamp}.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    scores = RESULTS / "agent_scores.csv"
    new = not scores.exists()
    with scores.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "model", "solved", "total", "avg_tool_calls"])
        w.writerow([stamp, label, solved, len(records), f"{avg_calls:.1f}"])
    print(f"Saved results/agent_{safe}_{stamp}.json and appended to results/agent_scores.csv")


def main():
    ap = argparse.ArgumentParser(description="End-to-end bug-fixing eval for Arke Code.")
    ap.add_argument("--model", default="arke-code")
    ap.add_argument("--only", nargs="*", help="task ids to run (default: all)")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--tag", default="")
    ap.add_argument("--check", action="store_true", help="validate the tasks, no model needed")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check() else 1)
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    except requests.exceptions.RequestException:
        sys.exit("Ollama is not running. Open the Ollama app and try again.")
    evaluate(args.model, args.only, args.num_ctx, args.tag)


if __name__ == "__main__":
    main()
