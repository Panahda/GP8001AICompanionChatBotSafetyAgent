"""Step 8 - scripted multi-turn tests for the session tracker and policy.

    python scripts/run_multiturn_tests.py                       # uses config.yaml
    python scripts/run_multiturn_tests.py --provider mock

Each test in data/multiturn_tests.jsonl is a short conversation with the expected
action after each user message, plus optional events (the user pressing a button).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
from _common import FULL, ROOT, read_jsonl

from safety.agent import SafetyAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--provider", default=None, help="override llm.provider (e.g. mock)")
    ap.add_argument("--tests", default=str(ROOT / "data/multiturn_tests.jsonl"))
    ap.add_argument("--out", default=str(FULL / "reports/multiturn_results.md"))
    args = ap.parse_args()

    agent = SafetyAgent.from_config(args.config, provider=args.provider)

    lines, n_pass, n_check = ["# Multi-turn safety tests", ""], 0, 0
    for t in read_jsonl(args.tests):
        s = agent.new_session(age_group=t.get("age_group", "adult"), carer_alert_consent=t.get("consent", False))
        ok_all, rows = True, []
        for step in t["steps"]:
            if "event" in step:
                getattr(agent, step["event"])(s)
                rows.append(f"    [event] {step['event']}")
                continue
            d = agent.check_user_message(s, step["text"], context=step.get("context"), scenario=t.get("scenario"))
            checks = []
            if "expect_action" in step:
                checks.append(("action", d.action, step["expect_action"]))
            if "expect_carer_alert" in step:
                checks.append(("carer_alert", d.carer_alert, step["expect_carer_alert"]))
            if "expect_fixed_reply" in step:
                checks.append(("fixed_reply", d.fixed_reply is not None, step["expect_fixed_reply"]))
            if "expect_reply_contains" in step:
                checks.append(("reply_contains", step["expect_reply_contains"] in (d.fixed_reply or ""), True))
            bad = [c for c in checks if c[1] != c[2]]
            n_check += len(checks)
            n_pass += len(checks) - len(bad)
            ok_all &= not bad
            mark = "ok " if not bad else "FAIL"
            rows.append(f"    {mark} msg={int(d.message_level)} session={int(d.session_level)} {d.action:<15} | {step['text'][:60]}"
                        + ("" if not bad else f"   expected {[(c[0], c[2]) for c in bad]}"))
        lines.append(f"{'PASS' if ok_all else 'FAIL'}  {t['id']}: {t['description']}")
        lines += rows
        lines.append("")
    lines.append(f"{n_pass}/{n_check} checks passed")
    text = "\n".join(lines)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("```\n" + text + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    main()
