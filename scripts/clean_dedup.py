"""Step 2 - clean, de-duplicate, and remove anything too close to the test set.

    python scripts/clean_dedup.py --inputs data/raw/synthetic_raw.jsonl --include-seeds

1. drop empty / too long messages
2. exact duplicates (after normalisation)
3. near duplicates within a subtype (TF-IDF char 3-5 grams, cosine >= --near-dup)
4. leakage: any training item with cosine >= --leak to a test item is removed,
   so the test score is not inflated by near copies.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import _common  # noqa: F401
import numpy as np
from _common import ROOT, read_jsonl, write_jsonl
from sklearn.feature_extraction.text import TfidfVectorizer

from safety.text import normalize_for_rules


def near_dup_mask(texts: list[str], thresh: float) -> list[bool]:
    """Greedy: keep an item unless it is too similar to an item already kept."""
    if len(texts) < 2:
        return [True] * len(texts)
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts)
    keep = [True] * len(texts)
    S = (X @ X.T).toarray()
    for i in range(len(texts)):
        if not keep[i]:
            continue
        dup = np.where(S[i, i + 1:] >= thresh)[0] + i + 1
        for j in dup:
            keep[j] = False
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=[str(ROOT / "data/raw/synthetic_raw.jsonl")])
    ap.add_argument("--include-seeds", action="store_true", help="add data/seeds.jsonl to the pool")
    ap.add_argument("--test", default=str(ROOT / "data/test_set.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/interim/clean.jsonl"))
    ap.add_argument("--near-dup", type=float, default=0.90)
    ap.add_argument("--leak", type=float, default=0.80)
    ap.add_argument("--max-words", type=int, default=60)
    args = ap.parse_args()

    rows = []
    for p in args.inputs:
        rows += read_jsonl(p)
    if args.include_seeds:
        rows += read_jsonl(ROOT / "data/seeds.jsonl")
    stats = Counter(total=len(rows))

    # 1. length
    rows = [r for r in rows if 1 <= len(r["text"].split()) <= args.max_words]
    stats["after_length"] = len(rows)

    # 2. exact duplicates
    seen, uniq = set(), []
    for r in rows:
        key = normalize_for_rules(r["text"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    rows = uniq
    stats["after_exact_dedup"] = len(rows)

    # 3. near duplicates per subtype
    by_sub = defaultdict(list)
    for r in rows:
        by_sub[r["subtype"]].append(r)
    kept = []
    for st, group in by_sub.items():
        mask = near_dup_mask([normalize_for_rules(r["text"]) for r in group], args.near_dup)
        kept += [r for r, k in zip(group, mask) if k]
    rows = kept
    stats["after_near_dedup"] = len(rows)

    # 4. leakage against the test set
    leaked = []
    try:
        test = read_jsonl(args.test)
    except FileNotFoundError:
        test = read_jsonl(ROOT / "data/test_set_draft.jsonl")
        print("note: data/test_set.jsonl not found, using data/test_set_draft.jsonl for the leakage check")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    vec.fit([normalize_for_rules(r["text"]) for r in rows + test])
    A = vec.transform([normalize_for_rules(r["text"]) for r in rows])
    B = vec.transform([normalize_for_rules(t["text"]) for t in test])
    sim = (A @ B.T).toarray().max(axis=1) if len(test) else np.zeros(len(rows))
    clean = []
    for r, s in zip(rows, sim):
        (leaked if s >= args.leak else clean).append(r)
    stats["removed_test_leakage"] = len(leaked)
    stats["final"] = len(clean)

    write_jsonl(args.out, clean)
    report = {"stats": dict(stats), "per_level": dict(Counter(r["level"] for r in clean)),
              "per_subtype": dict(Counter(r["subtype"] for r in clean)),
              "leaked_examples": [r["text"] for r in leaked[:20]]}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/cleaning_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["stats"], indent=2))
    print("per level:", report["per_level"])


if __name__ == "__main__":
    main()
