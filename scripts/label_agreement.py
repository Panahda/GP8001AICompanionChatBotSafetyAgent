"""Inter-annotator agreement for the test set (do this BEFORE freezing data/test_set.jsonl).

Two team members label the same draft independently (a CSV with columns id,level each):

    python scripts/label_agreement.py annot_siming.csv annot_jeff.csv

Prints Cohen's kappa (plain and quadratic-weighted) and writes the disagreements
to data/review/test_disagreements.csv so the pair can settle a final label together.
"""
from __future__ import annotations

import argparse
import csv

import _common  # noqa: F401
from _common import ROOT, read_jsonl
from sklearn.metrics import cohen_kappa_score


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["id"]: int(r["level"]) for r in csv.DictReader(f) if r.get("level", "").strip() != ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--draft", default=str(ROOT / "data/test_set_draft.jsonl"))
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    ids = sorted(set(A) & set(B))
    a, b = [A[i] for i in ids], [B[i] for i in ids]
    print(f"items labelled by both: {len(ids)}")
    print(f"raw agreement:            {sum(x == y for x, y in zip(a, b)) / len(ids):.3f}")
    print(f"Cohen's kappa:            {cohen_kappa_score(a, b):.3f}")
    print(f"quadratic-weighted kappa: {cohen_kappa_score(a, b, weights='quadratic'):.3f}")
    text = {r["id"]: r["text"] for r in read_jsonl(args.draft)}
    out = ROOT / "data/review/test_disagreements.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "level_a", "level_b", "final_level"])
        for i in ids:
            if A[i] != B[i]:
                w.writerow([i, text.get(i, ""), A[i], B[i], ""])
    print(f"disagreements written to {out}")


if __name__ == "__main__":
    main()
