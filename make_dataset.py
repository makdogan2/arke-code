"""Turn Arke Code's own good work into fine-tuning data (Phase 5).

Sources:
  - results/agent_*.json  : agent eval runs; only tasks that were solved
                            (visible + hidden tests passed, tests untouched)
  - logs/*.jsonl          : assistant.py conversations you marked with /good

Output: data/sft.jsonl, one chat example per line in the format most trainers
(Unsloth, TRL, Axolotl) accept for tool calling:

  {"messages": [{"role": "system", ...}, {"role": "user", ...},
                {"role": "assistant", "content": "...", "tool_calls": [...]},
                {"role": "tool", "name": "read_file", "content": "..."}, ...],
   "tools": [...], "source": "agent:median"}

The system prompt is taken from the Modelfile, because the fine-tuned model will
run with that same prompt.

Usage:
    python make_dataset.py              # build data/sft.jsonl and print a summary
    python make_dataset.py --show 2     # also print two examples in readable form
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from tools import TOOL_SPECS

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "sft.jsonl"

# Leftovers of the chat template that sometimes leak into plain text replies.
ARTIFACTS = re.compile(r"</?tool_call>|<\|im_(start|end)\|>")


def system_prompt():
    text = (ROOT / "Modelfile").read_text(encoding="utf-8")
    m = re.search(r'SYSTEM\s+"""(.*?)"""', text, re.DOTALL)
    return m.group(1).strip() if m else None


def clean_message(m):
    """Convert one Ollama-style message to the trainer format, dropping artifacts."""
    role = m["role"]
    if role == "tool":
        return {"role": "tool", "name": m.get("tool_name") or m.get("name", ""),
                "content": m.get("content", "")}
    out = {"role": role, "content": ARTIFACTS.sub("", m.get("content") or "").strip()}
    calls = []
    for c in m.get("tool_calls") or []:
        fn = c.get("function", {})
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                continue
        calls.append({"type": "function", "function": {"name": fn.get("name", ""), "arguments": args}})
    if calls:
        out["tool_calls"] = calls
    return out


def valid(messages):
    """A usable example starts with a user turn and ends with a plain assistant answer,
    and every tool result follows a tool call."""
    if not messages or messages[0]["role"] != "user":
        return False
    last = messages[-1]
    if last["role"] != "assistant" or last.get("tool_calls") or not last["content"]:
        return False
    for prev, cur in zip(messages, messages[1:]):
        if cur["role"] == "tool" and not (prev["role"] == "tool" or prev.get("tool_calls")):
            return False
    return True


def from_agent_runs():
    examples = []
    for path in sorted((ROOT / "results").glob("agent_*.json")):
        for rec in json.loads(path.read_text(encoding="utf-8")):
            if rec.get("solved"):
                msgs = [clean_message(m) for m in rec["messages"]]
                examples.append((f"agent:{rec['id']}", msgs))
    return examples


def from_logs(with_context=False):
    records, ratings = {}, {}
    for path in sorted((ROOT / "logs").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if "rate" in item:
                ratings[item["rate"]] = item["rating"]  # the last rating wins
            elif "id" in item:
                records[item["id"]] = item
    examples = []
    for log_id, rating in ratings.items():
        if rating == "good" and log_id in records:
            msgs = [clean_message(m) for m in records[log_id]["messages"]]
            if not with_context:
                # Keep only the rated turn: earlier answers in the same chat were not
                # rated and may be bad, and a trainer would learn from them too.
                last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
                msgs = msgs[last_user:]
            examples.append((f"chat:{log_id}", msgs))
    return examples, len(records), sum(r == "bad" for r in ratings.values())


def main():
    ap = argparse.ArgumentParser(description="Build Phase 5 fine-tuning data from Arke Code's own work.")
    ap.add_argument("--show", type=int, default=0, help="print this many examples")
    ap.add_argument("--with-context", action="store_true",
                    help="keep earlier turns of a rated chat (they get trained on too)")
    args = ap.parse_args()

    system = system_prompt()
    agent = from_agent_runs()
    chats, n_logged, n_bad = from_logs(args.with_context)

    seen, rows, skipped = set(), [], 0
    for source, msgs in agent + chats:
        if not valid(msgs):
            skipped += 1
            continue
        key = hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        full = ([{"role": "system", "content": system}] if system else []) + msgs
        rows.append({"messages": full, "tools": TOOL_SPECS, "source": source})

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_agent = sum(r["source"].startswith("agent:") for r in rows)
    calls = sum(len(m.get("tool_calls", [])) for r in rows for m in r["messages"])
    print(f"Agent eval runs : {len(agent)} solved trajectories")
    print(f"Chat logs       : {n_logged} logged exchanges, {len(chats)} marked /good, {n_bad} marked /bad")
    print(f"Written         : {len(rows)} examples ({n_agent} agent, {len(rows) - n_agent} chat), "
          f"{calls} tool calls, {skipped} skipped as malformed, duplicates removed")
    print(f"Output          : {OUT.relative_to(ROOT)}")

    for row in rows[: args.show]:
        print("\n" + "=" * 70 + f"\n{row['source']}")
        for m in row["messages"][1:]:
            if m["role"] == "tool":
                print(f"  [tool result: {m['name']}] {m['content'][:80]!r}...")
            else:
                text = m["content"][:160].replace("\n", " ")
                calls = ", ".join(c["function"]["name"] for c in m.get("tool_calls", []))
                print(f"  {m['role']}: {text}" + (f"  -> calls {calls}" if calls else ""))


if __name__ == "__main__":
    main()
