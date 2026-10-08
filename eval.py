"""Measure Arke Code with numbers instead of impressions.

Usage:
    python eval.py --check                          # prove the eval set itself is correct
    python eval.py                                  # run the custom suite on arke-code
    python eval.py --suite humaneval --limit 40     # first 40 HumanEval problems
    python eval.py --model qwen3-coder --suite all  # compare against another Ollama model
    python eval.py --tag prompt-v2                  # label a round in scores.csv

Every run is saved to results/<model>_<suite>_<time>.json, and one summary line is
appended to results/scores.csv, so progress across training rounds stays visible.

Generated code runs on this machine in a separate Python process with a timeout.
It is model output: only run models you trust.
"""

import argparse
import csv
import gzip
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests

from evals.custom_tasks import TASKS

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OLLAMA_URL = "http://localhost:11434"
HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
HUMANEVAL_CACHE = ROOT / "evals" / "HumanEval.jsonl.gz"
RUN_TIMEOUT = 15  # seconds per program

CODE_ONLY = ("\n\nReply with a single ```python code block containing the complete code. "
             "No explanation, no example usage.")
TESTS_ONLY = ("\n\nReply with a single ```python code block containing only the test code. "
              "No explanation.")
STYLE_SMELL = re.compile(r"[=!]=\s*(True|False)\b")


# ------------------------------------------------------------------ model I/O

def ask_model(model, prompt, num_ctx):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": num_ctx, "num_predict": 2048},
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    seconds = data.get("eval_duration", 0) / 1e9
    tok_per_s = data.get("eval_count", 0) / seconds if seconds else 0.0
    return data["message"]["content"], tok_per_s


def extract_code(reply):
    """Take the longest fenced code block; fall back to the whole reply."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", reply, re.DOTALL)
    return max(blocks, key=len) if blocks else reply


def run_program(source):
    """Run a Python program in a fresh process. Returns (ok, short error text)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "program.py"
        path.write_text(source, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(path)], cwd=tmp, capture_output=True,
                                  text=True, timeout=RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            return False, f"timeout after {RUN_TIMEOUT}s"
    if proc.returncode == 0:
        return True, ""
    return False, describe_error(proc.stderr) or f"exit code {proc.returncode}"


def describe_error(stderr):
    """Last error line plus the program line that raised it, e.g.
    'AssertionError at: assert flatten(["ab"]) == ["ab"]'."""
    lines = stderr.strip().splitlines()
    if not lines:
        return ""
    culprit = ""
    for i, line in enumerate(lines[:-1]):
        if 'program.py", line' in line and i + 1 < len(lines):
            culprit = lines[i + 1].strip()
    error = lines[-1].strip()
    return f"{error} at: {culprit}"[:240] if culprit else error[:240]


# --------------------------------------------------------------- task scoring

def score_custom(task, code):
    """Return (passed, detail) for one custom task given the model's code."""
    if task["kind"] == "implement":
        return run_program(code + "\n\n" + task["tests"])

    # write_tests: must pass on the reference, fail on every mutant, avoid `== True`
    if STYLE_SMELL.search(code):
        return False, "style: compares with == True / == False"
    name = re.search(r"def (\w+)", task["reference"]).group(1)
    if re.search(rf"^\s*def {name}\b", code, re.MULTILINE):
        return False, f"redefines {name}(), so the planted bugs are never tested"
    ok, err = run_program(task["reference"] + "\n\n" + code)
    if not ok:
        return False, f"fails on correct code: {err}"
    for i, mutant in enumerate(task["mutants"], 1):
        survived, _ = run_program(mutant + "\n\n" + code)
        if survived:
            return False, f"misses bug #{i}: {task['bug_notes'][i - 1]}"
    return True, ""


def load_humaneval():
    if not HUMANEVAL_CACHE.exists():
        print("Downloading HumanEval (one time)...")
        r = requests.get(HUMANEVAL_URL, timeout=60)
        r.raise_for_status()
        HUMANEVAL_CACHE.write_bytes(r.content)
    with gzip.open(HUMANEVAL_CACHE, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def humaneval_prompt(problem):
    return ("Complete this Python function. Keep the same signature.\n\n```python\n"
            + problem["prompt"] + "```")


def score_humaneval(problem, code):
    # The original prompt goes first so its imports and helper functions exist;
    # `pass` makes it valid on its own, and the model's full function redefines it.
    program = (problem["prompt"] + "    pass\n\n" + code + "\n\n" + problem["test"]
               + f"\n\ncheck({problem['entry_point']})\n")
    return run_program(program)


# ---------------------------------------------------------------------- runs

def check_eval_set():
    """Run every reference against its own tests (and mutants) to prove the set is sound."""
    bad = 0
    for task in TASKS:
        if task["kind"] == "implement":
            ok, err = score_custom(task, task["reference"])
        else:
            ok, err = score_custom(task, task["example_tests"])
        print(f"  {'ok  ' if ok else 'FAIL'} {task['id']} {err}")
        bad += not ok
    print(f"\n{len(TASKS) - bad}/{len(TASKS)} tasks are sound.")
    return bad == 0


def build_jobs(suite, limit):
    jobs = []
    if suite in ("custom", "all"):
        for t in TASKS:
            suffix = TESTS_ONLY if t["kind"] == "write_tests" else CODE_ONLY
            jobs.append(("custom", t["id"], t["prompt"] + suffix,
                         lambda code, t=t: score_custom(t, code)))
    if suite in ("humaneval", "all"):
        for p in load_humaneval()[:limit]:
            jobs.append(("humaneval", p["task_id"], humaneval_prompt(p) + CODE_ONLY,
                         lambda code, p=p: score_humaneval(p, code)))
    return jobs


def run_eval(model, suite, limit, num_ctx, tag=""):
    jobs = build_jobs(suite, limit)
    print(f"Evaluating {model} on {len(jobs)} tasks ({suite})\n")
    records, started = [], time.time()
    for n, (source, task_id, prompt, scorer) in enumerate(jobs, 1):
        t0 = time.time()
        reply, tps = ask_model(model, prompt, num_ctx)
        code = extract_code(reply)
        ok, detail = scorer(code)
        records.append({"suite": source, "id": task_id, "passed": ok, "detail": detail,
                        "seconds": round(time.time() - t0, 1), "tok_per_s": round(tps, 1),
                        "reply": reply})
        mark = "PASS" if ok else "FAIL"
        print(f"[{n:>3}/{len(jobs)}] {mark} {task_id:<22} {time.time() - t0:5.1f}s  {detail}")

    passed = sum(r["passed"] for r in records)
    pct = 100 * passed / len(records) if records else 0.0
    speed = sum(r["tok_per_s"] for r in records) / len(records) if records else 0.0
    minutes = (time.time() - started) / 60
    print(f"\n{model}: {passed}/{len(records)} passed ({pct:.1f}%), "
          f"{speed:.0f} tok/s average, {minutes:.1f} min")
    for s in sorted({r["suite"] for r in records}):
        sub = [r for r in records if r["suite"] == s]
        print(f"  {s}: {sum(r['passed'] for r in sub)}/{len(sub)}")

    save(f"{model}@{tag}" if tag else model, suite, records, passed, pct, speed)


def save(model, suite, records, passed, pct, speed):
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    safe = model.replace(":", "_").replace("/", "_").replace("@", "_")
    detail_path = RESULTS / f"{safe}_{suite}_{stamp}.json"
    detail_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    scores = RESULTS / "scores.csv"
    new_file = not scores.exists()
    with scores.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "model", "suite", "passed", "total", "percent", "tok_per_s"])
        w.writerow([stamp, model, suite, passed, len(records), f"{pct:.1f}", f"{speed:.0f}"])
    print(f"\nSaved results/{detail_path.name} and appended to results/scores.csv")


def main():
    ap = argparse.ArgumentParser(description="Evaluate an Ollama model on coding tasks.")
    ap.add_argument("--model", default="arke-code")
    ap.add_argument("--suite", choices=["custom", "humaneval", "all"], default="custom")
    ap.add_argument("--limit", type=int, default=40, help="HumanEval problems to use (max 164)")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--check", action="store_true", help="validate the eval set, no model needed")
    ap.add_argument("--tag", default="", help="label for this round in scores.csv, e.g. prompt-v2")
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check_eval_set() else 1)
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    except requests.exceptions.RequestException:
        sys.exit("Ollama is not running. Open the Ollama app and try again.")
    run_eval(args.model, args.suite, args.limit, args.num_ctx, args.tag)


if __name__ == "__main__":
    main()
