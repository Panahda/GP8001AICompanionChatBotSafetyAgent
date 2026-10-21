"""Step 6 - temperature scaling + crisis thresholds, fitted on the VALIDATION set.

    python scripts/calibrate.py --model-dir models/transformer
    python scripts/calibrate.py --model-dir models/baseline

1. Temperature T minimises validation negative log-likelihood (Guo et al., 2017),
   so probabilities mean what they say.
2. crisis_lo = highest threshold that still catches >= --recall-lo of validation crises.
   Messages between crisis_lo and crisis_hi go to the LLM.
3. crisis_hi = lowest threshold (>= crisis_lo) where precision >= --precision-hi.
   Above it the classifier acts alone. If no threshold reaches that precision,
   crisis_hi = 1.01 and every possible crisis goes to the LLM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from safety.classifier import softmax


def nll(logits, y, T):
    p = softmax(logits / T)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--recall-lo", type=float, default=0.98)
    ap.add_argument("--precision-hi", type=float, default=0.95)
    ap.add_argument("--max-lo", type=float, default=0.5)
    ap.add_argument("--min-conf", type=float, default=0.55)
    args = ap.parse_args()

    d = Path(args.model_dir)
    logits, y = np.load(d / "val_logits.npy"), np.load(d / "val_labels.npy").astype(int)

    grid = np.arange(0.1, 5.01, 0.05)
    T = float(grid[np.argmin([nll(logits, y, t) for t in grid])])
    p_before, p = softmax(logits), softmax(logits / T)
    p3, is3 = p[:, 3], y == 3

    if is3.sum() == 0:
        raise SystemExit("validation set has no crisis items; cannot set thresholds")
    s = np.sort(p3[is3])
    k = int(np.floor((1 - args.recall_lo) * len(s)))
    lo = float(min(s[k], args.max_lo))

    hi = 1.01
    for t in np.unique(np.round(p3, 4)):
        if t < lo:
            continue
        flagged = p3 >= t
        if flagged.sum() >= 5 and (is3 & flagged).sum() / flagged.sum() >= args.precision_hi:
            hi = float(t)
            break

    def stats(t):
        f = p3 >= t
        return {"threshold": round(float(t), 3), "recall": round(float((f & is3).sum() / is3.sum()), 3),
                "precision": round(float((f & is3).sum() / max(1, f.sum())), 3), "flagged_rate": round(float(f.mean()), 3)}

    table = [stats(t) for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    grey = (p3 >= lo) & (p3 < hi)
    out = {
        "temperature": T, "crisis_lo": lo, "crisis_hi": hi, "min_conf": args.min_conf,
        "val_n": int(len(y)), "val_crisis_n": int(is3.sum()),
        "val_recall_at_lo": stats(lo)["recall"], "val_precision_at_hi": stats(hi)["precision"] if hi <= 1 else None,
        "val_llm_call_rate": round(float(grey.mean()), 3),
        "nll_before": round(nll(logits, y, 1.0), 4), "nll_after": round(nll(logits, y, T), 4),
        "ece_before": round(ece(p_before, y), 4), "ece_after": round(ece(p, y), 4),
        "pr_table": table,
    }
    (d / "thresholds.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "pr_table"}, indent=2))
    print("threshold  recall  precision  flagged")
    for r in table:
        print(f"  {r['threshold']:<8} {r['recall']:<7} {r['precision']:<10} {r['flagged_rate']}")


if __name__ == "__main__":
    main()
