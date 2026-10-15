import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

from tools import TOOL_SPECS, Workspace

# === CONFIG ===
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "arke-code"  # change here if the model is renamed

conversation_history = []
workspace = Workspace(".")

# Every exchange is appended to logs/ so good ones can become training data (Phase 5).
# Mark the last answer with /good or /bad; make_dataset.py keeps only /good ones.
LOG_DIR = Path(__file__).parent / "logs"
logging_on = True
last_log_id = None


def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
    except requests.exceptions.RequestException:
        print("Ollama is not running. Open the Ollama app and try again.")
        sys.exit(1)
    if not any(m.split(":")[0] == MODEL_NAME for m in models):
        print(f"Model '{MODEL_NAME}' not found. Available: {models}")
        print(f"Create it with: ollama create {MODEL_NAME} -f Modelfile")
        sys.exit(1)


MAX_TOOL_ROUNDS = 8


def trimmed_history(limit=30):
    """Last messages, never starting in the middle of a tool exchange."""
    msgs = conversation_history[-limit:]
    while msgs and msgs[0]["role"] != "user":
        msgs = msgs[1:]
    return msgs


class OllamaError(Exception):
    pass


def stream_reply(max_tokens, temperature):
    """One model call. Streams text to the screen, returns (text, tool_calls)."""
    payload = {
        "model": MODEL_NAME,
        "messages": trimmed_history(),
        "tools": TOOL_SPECS,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    text, tool_calls = "", []
    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True) as r:
        if r.status_code != 200:
            # Ollama explains what went wrong in the body, e.g. "model requires more system memory"
            try:
                detail = r.json().get("error", r.text)
            except ValueError:
                detail = r.text
            raise OllamaError(f"Ollama returned {r.status_code}: {detail}")
        for line in r.iter_lines():
            if not line:
                continue
            msg = json.loads(line).get("message", {})
            chunk = msg.get("content", "")
            if chunk:
                print(chunk, end="", flush=True)
                text += chunk
            tool_calls.extend(msg.get("tool_calls") or [])
    return text, tool_calls


def short(value, limit=60):
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def log_path():
    return LOG_DIR / f"conversations-{datetime.now():%Y-%m}.jsonl"


def write_log(record):
    if not logging_on:
        return
    LOG_DIR.mkdir(exist_ok=True)
    with log_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_exchange(temperature):
    """Save the conversation as the model saw it, ending with this turn's answer."""
    global last_log_id
    last_log_id = f"{datetime.now():%Y%m%d-%H%M%S}-{len(conversation_history)}"
    write_log({"id": last_log_id, "time": datetime.now().isoformat(timespec="seconds"),
               "model": MODEL_NAME, "temperature": temperature,
               "project": str(workspace.root), "messages": trimmed_history()})


def rate_last(rating, note=""):
    if not last_log_id:
        print("  Nothing to rate yet.")
    elif not logging_on:
        print("  Logging is off (--no-log), so there is nothing to rate.")
    else:
        write_log({"rate": last_log_id, "rating": rating, "note": note,
                   "time": datetime.now().isoformat(timespec="seconds")})
        print(f"  Marked the last answer as {rating}.")


def chat(user_message, max_tokens, temperature):
    conversation_history.append({"role": "user", "content": user_message})
    for _ in range(MAX_TOOL_ROUNDS):
        text, tool_calls = stream_reply(max_tokens, temperature)
        message = {"role": "assistant", "content": text}
        if tool_calls:
            message["tool_calls"] = tool_calls
        conversation_history.append(message)
        if not tool_calls:
            print()
            log_exchange(temperature)
            return
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            shown = ", ".join(f"{k}={short(v)}" for k, v in args.items())
            print(f"\n  [tool] {name}({shown})", flush=True)
            result = workspace.call(name, args)
            conversation_history.append({"role": "tool", "tool_name": name, "content": result})
    print("\n  (stopped: too many tool calls in a row)")
    log_exchange(temperature)


def read_pasted_code():
    print("  Paste your code, then press Enter on an empty line:")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    print(f"  ({len(lines)} lines received)")
    return "Review this code:\n```\n" + "\n".join(lines) + "\n```"


def main():
    global workspace, logging_on
    ap = argparse.ArgumentParser(description="Arke Code, a local coding assistant.")
    ap.add_argument("--root", default=".", help="project folder the assistant may read (default: here)")
    ap.add_argument("--no-log", action="store_true", help="do not save conversations to logs/")
    args = ap.parse_args()
    workspace = Workspace(args.root)
    logging_on = not args.no_log
    check_ollama()

    print("=" * 60)
    print("  ARKE CODE - local code assistant")
    print("=" * 60)
    print(f"Project: {workspace.root}")
    print("Tools: list_dir, read_file, grep, search_code, git_diff | ask first: edit_file, write_file, "
          "run_python, run_tests")
    print("Commands: /quit  /clear  /temp 0.5  /long  /short  /paste  /settings  /good  /bad")
    if logging_on:
        print(f"Logging to {log_path().relative_to(Path(__file__).parent)} (--no-log to turn off)")

    max_tokens = 2048
    temp = 0.5

    while True:
        print()
        user_input = input("You > ").strip()
        if not user_input:
            continue

        if user_input == "/quit":
            print("See you!")
            break
        if user_input == "/clear":
            conversation_history.clear()
            print("  History cleared.")
            continue
        if user_input.startswith("/temp "):
            try:
                temp = max(0.1, min(2.0, float(user_input.split()[1])))
                print(f"  Temperature: {temp}")
            except ValueError:
                print("  Usage: /temp 0.5")
            continue
        if user_input == "/long":
            max_tokens = 4096
            print("  Max tokens: 4096")
            continue
        if user_input == "/short":
            max_tokens = 512
            print("  Max tokens: 512")
            continue
        if user_input == "/settings":
            print(f"  Model: {MODEL_NAME} | Temp: {temp} | Max tokens: {max_tokens} "
                  f"| History: {len(conversation_history)} messages | Project: {workspace.root}")
            continue
        if user_input.split()[0] in ("/good", "/bad"):
            rate_last(user_input.split()[0][1:], user_input[len(user_input.split()[0]):].strip())
            continue
        if user_input == "/paste":
            user_input = read_pasted_code()

        print("\nArke Code > ", end="", flush=True)
        try:
            chat(user_input, max_tokens, temp)
        except (OllamaError, requests.exceptions.RequestException) as e:
            # keep the session alive; drop the unanswered question so history stays valid
            while conversation_history and conversation_history[-1]["role"] != "user":
                conversation_history.pop()
            if conversation_history:
                conversation_history.pop()
            print(f"\n  [error] {e}")
            print("  Check 'ollama ps' and the Ollama app, then ask again.")


if __name__ == "__main__":
    main()
