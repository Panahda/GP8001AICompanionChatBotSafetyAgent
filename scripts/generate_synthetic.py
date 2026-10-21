"""Step 1 - generate synthetic training messages with an LLM.

    python scripts/generate_synthetic.py --provider openai                # full run (~170 calls)
    python scripts/generate_synthetic.py --dry-run --max-calls 2         # print two prompts, no API call
    python scripts/generate_synthetic.py --provider gemini --only nssi   # one subtype

Output is appended to data/raw/synthetic_raw.jsonl, so an interrupted run can be resumed:
cells that already have enough rows are skipped.

Use a DIFFERENT provider for scripts/relabel_check.py than the one used here.
"""
from __future__ import annotations

import argparse
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import _common  # noqa: F401  (sets sys.path)
import yaml
from _common import FULL, ROOT, append_jsonl, read_jsonl

from safety.llm_providers import get_provider

SYSTEM = """You write realistic but fictional chat messages that will be used to train a SAFETY CLASSIFIER for CompanionAI, a practice and companionship chatbot for verbal autistic teenagers and adults in Singapore. The classifier protects users by noticing tension, overload and crisis. Your messages are never shown to real users.

Hard rules:
- Never include specific methods, means, places, amounts or instructions for self-harm or suicide. Describe thoughts, intent, plans or past attempts only in general words ("I have a plan", "I tried before").
- No real people or real organisations. Invented first names are fine.
- Write what the USER would type in a chat: 1 to 40 words each.
- Make the messages varied: different openings, lengths, wording and grammar. Do not copy the examples.
- Do not stereotype autistic people. Many write directly and literally, and that is normal.
Return JSON only: {"items": [{"context": "<the companion's previous line>", "text": "<the user's message>"}]}"""

USER_TMPL = """Write {n} different user messages.

Label: level {level} ({level_name}) / {subtype}
Meaning: {desc}
Setting: {scenario_desc}. For "context", write the companion's previous line that fits this setting{ctx_note}.
User style: {style}
{extra}
Examples of this label (for meaning only; do not copy them):
{examples}"""

EXTRA = {
    "idiom_negative": "Every message must use a figure of speech with words like kill, die, dead, death or dying that is clearly NOT about harm.",
    "blunt_literal": "Blunt or literal but calm. No distress.",
    "special_interest": "Long, detailed, enthusiastic messages about one interest. No distress.",
    "tense_explicit": "Mild tension only. No mention of death, suicide or self-harm.",
    "tense_hesitant": "Hesitation or reassurance-seeking only. No mention of death, suicide or self-harm.",
    "overwhelmed_overload": "Overload only. No mention of death, suicide or self-harm.",
    "overwhelmed_shutdown": "Wanting to stop or shutting down only. No mention of death, suicide or self-harm.",
}
CRISIS_ROLEPLAY = ("In about half of the messages the user first answers the companion's role-play line, "
                   "then says something real about their own life.")

ITEMS_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {"context": {"type": "string"}, "text": {"type": "string"}},
        "required": ["context", "text"]}}},
    "required": ["items"],
}

# Generated crisis items must stay non-instructional: drop any that names a means.
MEANS = re.compile(
    r"\b(pills?|tablets?|overdos\w*|rope|noose|hang(ing)?|bridge|ledge|roof|jump(ing)? (off|from)|blade|razor|knife|knives|"
    r"gun|poison\w*|bleach|tracks?|charcoal|lethal|mg|milligrams?|wrists?)\b", re.IGNORECASE)


def load_taxonomy():
    tax = yaml.safe_load((ROOT / "data/taxonomy.yaml").read_text(encoding="utf-8"))
    out = {}
    for lvl, info in tax["levels"].items():
        for st, d in info["subtypes"].items():
            desc = d["desc"] if isinstance(d, dict) else d
            cssrs = d.get("cssrs") if isinstance(d, dict) else None
            out[st] = {"level": int(lvl), "level_name": info["name"], "desc": desc, "cssrs": cssrs}
    return out


def build_calls(matrix, tax, seeds, only=None, rng=None, per_subtype=None):
    rng = rng or random.Random(0)
    n = matrix["items_per_call"]
    personas = matrix["personas"]
    calls = []
    for st, spec in matrix["subtypes"].items():
        if only and st not in only:
            continue
        pool = [s for s in seeds if s["subtype"] == st]
        n_calls = math.ceil(spec["target"] / n)
        for k in range(min(n_calls, per_subtype) if per_subtype else n_calls):
            scen = spec["scenarios"][k % len(spec["scenarios"])]
            persona = personas[(k * 7 + len(st)) % len(personas)]
            ex = rng.sample(pool, min(4, len(pool)))
            info = tax[st]
            extra = EXTRA.get(st, CRISIS_ROLEPLAY if info["level"] == 3 and scen != "free_chat" else "")
            user = USER_TMPL.format(
                n=n, level=info["level"], level_name=info["level_name"], subtype=st, desc=info["desc"],
                scenario_desc=matrix["scenarios"][scen],
                ctx_note=" (it may be an empty string)" if scen == "free_chat" else "",
                style=persona["style"], extra=(extra + "\n") if extra else "",
                examples="\n".join(f"- {e['text']}" for e in ex),
            )
            calls.append({"subtype": st, "k": k, "scenario": scen, "persona": persona["id"], "user": user, "info": info})
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=str(FULL / "data/raw/synthetic_raw.jsonl"))
    ap.add_argument("--only", nargs="*", help="subtypes to generate")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--calls-per-subtype", type=int, default=None,
                    help="cap calls per subtype (1 = one small batch of every label, for a quick end-to-end run)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--timeout", type=float, default=None,
                    help="seconds to wait for one call (default: 900 for lmstudio, 120 otherwise)")
    ap.add_argument("--items-per-call", type=int, default=None,
                    help="messages per call (default 20 from gen_matrix.yaml); keep the same value when resuming")
    args = ap.parse_args()

    matrix = yaml.safe_load((ROOT / "data/gen_matrix.yaml").read_text(encoding="utf-8"))
    if args.items_per_call:
        matrix["items_per_call"] = args.items_per_call
    tax = load_taxonomy()
    seeds = read_jsonl(ROOT / "data/seeds.jsonl")
    calls = build_calls(matrix, tax, seeds, args.only, random.Random(args.seed), args.calls_per_subtype)
    if args.max_calls:
        calls = calls[: args.max_calls]

    if args.dry_run:
        for c in calls:
            print("=" * 80, f"\n{c['subtype']} | {c['scenario']} | {c['persona']}\n", "-" * 80)
            print(SYSTEM, "\n", "-" * 80, "\n", c["user"])
        print(f"\n{len(calls)} calls planned")
        return

    out = Path(args.out)
    done = {(r["subtype"], r["call_k"]) for r in read_jsonl(out)} if out.exists() else set()
    timeout = args.timeout or (900 if args.provider == "lmstudio" else 120)
    llm = get_provider(args.provider, args.model, timeout_s=timeout)
    print(f"generator: {llm.name}/{llm.model}, {matrix['items_per_call']} messages per call, timeout {timeout:.0f}s")
    stats = Counter()
    for i, c in enumerate(calls):
        if (c["subtype"], c["k"]) in done:
            stats["skipped_done"] += 1
            continue
        print(f"[{i + 1}/{len(calls)}] {c['subtype']:<24} {c['scenario']:<14} generating ...", flush=True)
        t0 = time.time()
        items = []
        for attempt in range(2):
            try:
                raw = llm.complete_json(SYSTEM, c["user"], temperature=matrix.get("temperature", 1.0), max_tokens=3000,
                                        schema=ITEMS_SCHEMA)
                items = raw.get("items", [])
                if items:
                    break
            except Exception as e:
                print(f"  attempt {attempt + 1} failed after {time.time() - t0:.0f}s: {type(e).__name__}: {str(e)[:120]}")
                if "Timeout" in type(e).__name__:
                    print(f"  The model did not finish within {timeout:.0f}s. Try a larger --timeout, "
                          "a smaller --items-per-call, or a smaller/faster model in LM Studio.")
                    break  # retrying a timeout just waits the same time again
                time.sleep(2)
        if not items:
            stats["empty_or_refused"] += 1
            continue
        rows = []
        for j, it in enumerate(items):
            text = str(it.get("text", "")).strip()
            if not text or len(text.split()) > 60:
                stats["bad_length"] += 1
                continue
            if c["info"]["level"] >= 2 and MEANS.search(text):
                stats["dropped_means"] += 1
                continue
            rows.append({
                "id": f"syn_{c['subtype']}_{c['k']:03d}_{j:02d}", "text": text,
                "context": str(it.get("context", "") or "").strip(), "scenario": c["scenario"],
                "level": c["info"]["level"], "subtype": c["subtype"], "cssrs": c["info"]["cssrs"],
                "persona": c["persona"], "call_k": c["k"], "generator": f"{llm.name}/{llm.model}", "source": "synthetic",
            })
        append_jsonl(out, rows)
        stats["kept"] += len(rows)
        print(f"    kept {len(rows)} in {time.time() - t0:.0f}s")
    print(dict(stats))


if __name__ == "__main__":
    main()
