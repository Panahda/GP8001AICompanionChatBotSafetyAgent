"""Turn the test set into a spreadsheet and back, so teammates can edit it in Excel or Google Sheets.

    python scripts/sheet.py to-csv   data/test_set_draft.jsonl test_set_edit.csv            # to rewrite messages
    python scripts/sheet.py to-csv   data/test_set.jsonl labels_jeff.csv --blank-labels      # to label blind
    python scripts/sheet.py to-jsonl test_set_edit.csv data/test_set.jsonl                   # back, with checks

`to-jsonl` checks every row first and writes nothing if a row is wrong:
level 0-3, a subtype that belongs to that level (data/taxonomy.yaml), a known scenario,
unique ids, and no self-harm means in crisis messages. Empty ids get new ones and the
C-SSRS item is filled in from the subtype. It also tells you how much of the set is new
compared with the draft (the team rule is at least half).

CSV files stay out of git (.gitignore); send them around by chat or Drive.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import _common  # noqa: F401  (sets sys.path)
import yaml
from _common import ROOT, read_jsonl, write_jsonl

COLUMNS = ["id", "text", "context", "scenario", "level", "subtype", "note", "author"]
BLANK_COLUMNS = ["id", "text", "context", "scenario", "level"]
SCENARIOS = ["hawker_centre", "new_colleague", "job_interview", "free_chat"]
# Same idea as the generator's filter: no means of self-harm in crisis messages.
MEANS = re.compile(
    r"\b(pills?|tablets?|overdos\w*|rope|noose|hang(ing)?|bridge|ledge|roof|jump(ing)? (off|from)|blade|razor|knife|knives|"
    r"gun|poison\w*|bleach|tracks?|charcoal|lethal|mg|milligrams?|wrists?)\b", re.IGNORECASE)


def taxonomy() -> dict[str, tuple[int, int | None]]:
    """subtype -> (level, cssrs item or None)"""
    tax = yaml.safe_load((ROOT / "data/taxonomy.yaml").read_text(encoding="utf-8"))
    out = {}
    for lvl, info in tax["levels"].items():
        for st, d in info["subtypes"].items():
            out[st] = (int(lvl), d.get("cssrs") if isinstance(d, dict) else None)
    return out


def norm(t: str) -> str:
    return re.sub(r"\W+", " ", (t or "").lower()).strip()


def to_csv(src: Path, dst: Path, blank: bool) -> None:
    rows = read_jsonl(src)
    cols = BLANK_COLUMNS if blank else COLUMNS
    with open(dst, "w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel reads UTF-8
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "level": "" if blank else r.get("level", "")})
    what = "level column left empty for blind labelling" if blank else "edit text, level, subtype; add rows at the bottom"
    print(f"wrote {len(rows)} rows to {dst} ({what})")


def read_csv(path: Path) -> list[dict]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
        print("warning: the file is not UTF-8. In Excel use Save As -> 'CSV UTF-8'. Check quotes and emoji.")
    return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(text.splitlines())]


def to_jsonl(src: Path, dst: Path, draft: Path) -> None:
    tax = taxonomy()
    rows, errors, used = [], [], set()
    sheet = read_csv(src)
    for i, r in enumerate(sheet, start=2):  # row 1 is the header in Excel
        if not any(r.values()):
            continue
        where = f"row {i}"
        text = r.get("text", "")
        if not text:
            errors.append(f"{where}: text is empty")
            continue
        try:
            level = int(float(r.get("level", "")))
            assert 0 <= level <= 3
        except (ValueError, AssertionError):
            errors.append(f"{where}: level must be 0, 1, 2 or 3 (got {r.get('level', '')!r})")
            continue
        subtype = r.get("subtype", "")
        if subtype not in tax:
            errors.append(f"{where}: unknown subtype {subtype!r} (see data/taxonomy.yaml)")
            continue
        if tax[subtype][0] != level:
            errors.append(f"{where}: subtype {subtype} belongs to level {tax[subtype][0]}, not {level}")
            continue
        scenario = r.get("scenario") or "free_chat"
        if scenario not in SCENARIOS:
            errors.append(f"{where}: scenario must be one of {SCENARIOS}")
            continue
        if level >= 2 and MEANS.search(text):
            errors.append(f"{where}: names a means of self-harm ({MEANS.search(text).group(0)!r}); rephrase in general words")
            continue
        rid = r.get("id", "")
        if rid and rid in used:
            errors.append(f"{where}: id {rid} is used twice")
            continue
        rows.append({"id": rid, "text": text, "context": r.get("context", ""), "scenario": scenario, "level": level,
                     "subtype": subtype, "cssrs": tax[subtype][1], "note": r.get("note", ""),
                     "author": r.get("author") or "team"})
        if rid:
            used.add(rid)

    if errors:
        print(f"{len(errors)} problem(s), nothing written:")
        print("\n".join("  " + e for e in errors))
        sys.exit(1)

    n = 1
    for r in rows:  # new rows get new ids
        if not r["id"]:
            while f"test_new_{n:03d}" in used:
                n += 1
            r["id"] = f"test_new_{n:03d}"
            used.add(r["id"])

    write_jsonl(dst, rows)
    old = {norm(r["text"]) for r in read_jsonl(draft)} if draft.exists() else set()
    new = sum(norm(r["text"]) not in old for r in rows)
    print(f"wrote {len(rows)} rows to {dst}")
    print("per level:", dict(sorted(Counter(r["level"] for r in rows).items())))
    print(f"new or rewritten vs the draft: {new}/{len(rows)} ({new / max(1, len(rows)):.0%}); the team rule is at least 50%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("to-csv", help="jsonl -> csv")
    a.add_argument("src")
    a.add_argument("dst")
    a.add_argument("--blank-labels", action="store_true", help="only id/text/context/scenario + an empty level column")
    b = sub.add_parser("to-jsonl", help="csv -> jsonl, with checks")
    b.add_argument("src")
    b.add_argument("dst")
    b.add_argument("--draft", default=str(ROOT / "data/test_set_draft.jsonl"), help="for the 'how much is new' count")
    args = ap.parse_args()
    if args.cmd == "to-csv":
        to_csv(Path(args.src), Path(args.dst), args.blank_labels)
    else:
        to_jsonl(Path(args.src), Path(args.dst), Path(args.draft))


if __name__ == "__main__":
    main()
