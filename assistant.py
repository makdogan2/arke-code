import json
import sys

import requests

# === CONFIG ===
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "arke-code"  # change here if the model is renamed

conversation_history = []


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


def chat(user_message, max_tokens, temperature):
    conversation_history.append({"role": "user", "content": user_message})
    payload = {
        "model": MODEL_NAME,
        "messages": conversation_history[-20:],
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    full_response = ""
    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True) as r:
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line).get("message", {}).get("content", "")
            print(chunk, end="", flush=True)
            full_response += chunk
    print()

    conversation_history.append({"role": "assistant", "content": full_response})


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
    check_ollama()

    print("=" * 60)
    print(f"  ARKE CODE - local code assistant")
    print("=" * 60)
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
                  f"| History: {len(conversation_history)} messages")
            continue
        if user_input == "/paste":
            user_input = read_pasted_code()

        print(f"\nArke Code > ", end="", flush=True)
        chat(user_input, max_tokens, temp)


if __name__ == "__main__":
    main()
