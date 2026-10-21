"""Step 3 - check the synthetic labels with a second, different LLM.

The generator was told which label to write, but it does not always follow it.
A second model (different family) labels every item blind with the same prompt
the safety agent uses at run time. Then:

    gap >= 2 levels                   -> dropped
    gap == 1, or crisis -> non-crisis -> kept, listed in runs/full/data/review/to_review.csv for the team
    agree                             -> kept

    python scripts/relabel_check.py --provider gemini --workers 8
    python scripts/relabel_check.py --apply-review runs/full/data/review/to_review.csv   # after the team fills in 'final_level'
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import _common  # noqa: F401
from _common import FULL, ROOT, read_jsonl, write_jsonl
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from safety.llm_check import SYSTEM_PROMPT, VERDICT_SCHEMA, build_user_prompt, parse_verdict
from safety.llm_providers import get_provider


def label_one(llm, r):
    try:
        raw = llm.complete_json(SYSTEM_PROMPT, build_user_prompt(r["text"], r.get("context"), r.get("scenario")),
                                temperature=0.0, max_tokens=300, schema=VERDICT_SCHEMA)
        level, cat, conf, why = parse_verdict(raw)
        return int(level), cat, why
    except Exception as e:
        return None, None, f"error: {type(e).__name__}"


def apply_review(path_review, path_in, path_out):
    decisions = {}
    with open(path_review, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fl = (row.get("final_level") or "").strip()
            if fl == "drop":
                decisions[row["id"]] = None
            elif fl != "":
                decisions[row["id"]] = int(fl)
    rows, dropped, changed = [], 0, 0
    for r in read_jsonl(path_in):
        if r["id"] in decisions:
            if decisions[r["id"]] is None:
                dropped += 1
                continue
            if decisions[r["id"]] != r["level"]:
                r["level"], r["label_note"] = decisions[r["id"]], "changed in team review"
                changed += 1
        rows.append(r)
    write_jsonl(path_out, rows)
    print(f"applied review: {changed} relabelled, {dropped} dropped -> {path_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--inp", default=str(FULL / "data/interim/clean.jsonl"))
    ap.add_argument("--out", default=str(FULL / "data/interim/relabeled.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample", type=int, default=None, help="only check N random items (cheap trial run)")
    ap.add_argument("--apply-review", default=None)
    ap.add_argument("--review-dir", default=str(FULL / "data/review"))
    ap.add_argument("--reports", default=str(FULL / "reports"))
    ap.add_argument("--timeout", type=float, default=None,
                    help="seconds per call (default: 300 for lmstudio, 30 otherwise)")
    args = ap.parse_args()

    if args.apply_review:
        apply_review(args.apply_review, args.out, args.out)
        return

    rows = read_jsonl(args.inp)
    if args.sample:
        import random

        rows = random.Random(0).sample(rows, min(args.sample, len(rows)))
    llm = get_provider(args.provider, args.model,
                       timeout_s=args.timeout or (300 if args.provider == "lmstudio" else 30))
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda r: label_one(llm, r), rows))

    kept, review, dropped, errors = [], [], [], 0
    gold, pred = [], []
    for r, (lvl, cat, why) in zip(rows, results):
        r["relabel_level"], r["relabel_category"], r["relabel_note"] = lvl, cat, why
        if lvl is None:
            errors += 1
            review.append(r)
            kept.append(r)
            continue
        gold.append(r["level"])
        pred.append(lvl)
        gap = abs(lvl - r["level"])
        if gap >= 2:
            dropped.append(r)
            continue
        if gap == 1 or (r["level"] == 3 and lvl < 3):
            review.append(r)
        kept.append(r)

    write_jsonl(args.out, kept)
    review_dir, reports = Path(args.review_dir), Path(args.reports)
    review_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    with open(review_dir / "to_review.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "context", "subtype", "level", "relabel_level", "relabel_note", "final_level"])
        for r in review:
            w.writerow([r["id"], r["text"], r.get("context", ""), r["subtype"], r["level"],
                        r["relabel_level"], r["relabel_note"], ""])
    write_jsonl(review_dir / "dropped_by_relabel.jsonl", dropped)

    report = {
        "checked": len(rows), "errors": errors, "kept": len(kept), "to_review": len(review), "dropped_gap2": len(dropped),
        "agreement": sum(g == p for g, p in zip(gold, pred)) / max(1, len(gold)),
        "cohen_kappa": cohen_kappa_score(gold, pred) if gold else None,
        "cohen_kappa_quadratic": cohen_kappa_score(gold, pred, weights="quadratic") if gold else None,
        "confusion_gold_rows_x_relabel_cols": confusion_matrix(gold, pred, labels=[0, 1, 2, 3]).tolist() if gold else None,
        "disagreement_by_subtype": dict(Counter(r["subtype"] for r in review + dropped)),
        "relabeler": f"{llm.name}/{llm.model}",
    }
    (reports / "relabel_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Team: open runs/full/data/review/to_review.csv, fill 'final_level' (0-3 or 'drop'), then run with --apply-review.")


if __name__ == "__main__":
    main()
