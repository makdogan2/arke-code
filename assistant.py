import argparse
import json
import sys

import requests

from tools import TOOL_SPECS, Workspace

# === CONFIG ===
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "arke-code"  # change here if the model is renamed

conversation_history = []
workspace = Workspace(".")


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
        r.raise_for_status()
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
    global workspace
    ap = argparse.ArgumentParser(description="Arke Code, a local coding assistant.")
    ap.add_argument("--root", default=".", help="project folder the assistant may read (default: here)")
    workspace = Workspace(ap.parse_args().root)
    check_ollama()

    print("=" * 60)
    print("  ARKE CODE - local code assistant")
    print("=" * 60)
    print(f"Project: {workspace.root}")
    print("Tools: list_dir, read_file, grep, git_diff | ask first: edit_file, write_file, "
          "run_python, run_tests")
    print("Commands: /quit  /clear  /temp 0.5  /long  /short  /paste  /settings")

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
        if user_input == "/paste":
            user_input = read_pasted_code()

        print("\nArke Code > ", end="", flush=True)
        chat(user_input, max_tokens, temp)


if __name__ == "__main__":
    main()
