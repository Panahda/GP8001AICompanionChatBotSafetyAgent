"""Step 4 - stratified train / validation split (+ light noise augmentation on train only).

    python scripts/split.py --inp data/interim/relabeled.jsonl --val-frac 0.15 --augment-rate 0.15

The TEST set is never made here: it is written by the team (data/test_set.jsonl)
and never used for training, threshold tuning or prompt tuning.
"""
from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict

import _common  # noqa: F401
from _common import ROOT, read_jsonl, write_jsonl


def noisy(text: str, rng: random.Random) -> str:
    """Phone-typing style noise: lower case, dropped punctuation, one swapped letter pair."""
    t = text.lower() if rng.random() < 0.6 else text
    if rng.random() < 0.5:
        t = re.sub(r"[.,!?;:]", "", t)
    words = t.split()
    cands = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if cands and rng.random() < 0.6:
        i = rng.choice(cands)
        w = words[i]
        j = rng.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(ROOT / "data/interim/relabeled.jsonl"))
    ap.add_argument("--outdir", default=str(ROOT / "data/processed"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--augment-rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = read_jsonl(args.inp)
    by = defaultdict(list)
    for r in rows:
        by[(r["level"], r["subtype"])].append(r)
    train, val = [], []
    for key, group in sorted(by.items()):
        rng.shuffle(group)
        n_val = max(1, round(len(group) * args.val_frac)) if len(group) > 3 else 0
        val += group[:n_val]
        train += group[n_val:]

    aug = []
    for r in train:
        if rng.random() < args.augment_rate:
            t2 = noisy(r["text"], rng)
            if t2 != r["text"]:
                aug.append({**r, "id": r["id"] + "_aug", "text": t2, "source": r.get("source", "") + "+aug"})
    train += aug
    rng.shuffle(train)

    write_jsonl(f"{args.outdir}/train.jsonl", train)
    write_jsonl(f"{args.outdir}/val.jsonl", val)
    print(f"train {len(train)} (incl. {len(aug)} augmented)  {dict(sorted(Counter(r['level'] for r in train).items()))}")
    print(f"val   {len(val)}  {dict(sorted(Counter(r['level'] for r in val).items()))}")


if __name__ == "__main__":
    main()
