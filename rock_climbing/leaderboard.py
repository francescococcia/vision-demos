"""Team leaderboard over finished runs.

Run `python main.py` first, then `python leaderboard.py`. It reads every
`sequence.json` + `climb.json` under data/output/ (or a folder you pass) and
ranks the attempts:

    1. topped out beats not topped
    2. faster time to top
    3. fewer holds used (more efficient beta)

Each climber also gets a one-line coach tip derived from their own numbers, and
the table is written to leaderboard.md next to the runs so it can be shared.
No API calls: everything comes from files the pipeline already wrote.

    python leaderboard.py                      # latest run folder
    python leaderboard.py --all                # every run ever made
    python leaderboard.py data/output/<stamp>  # one specific run folder
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "output"
HANDS = ("left_hand", "right_hand")
FEET = ("left_foot", "right_foot")


def load_attempts(root: Path) -> list[dict]:
    attempts = []
    for seq_path in sorted(root.rglob("sequence.json")):
        seq = json.loads(seq_path.read_text())
        climb_path = seq_path.with_name("climb.json")
        per_limb = {}
        if climb_path.exists():
            per_limb = json.loads(climb_path.read_text()).get(
                "utilization", {}).get("per_limb", {})
        share = {k: v.get("share_of_contact", 0.0) for k, v in per_limb.items()}
        attempts.append({
            "name": seq.get("label") or Path(seq.get("video", seq_path.parent.name)).stem,
            "topped": bool(seq.get("topped_out")),
            "time": seq.get("elapsed_seconds"),
            "holds": seq.get("holds_used") or 0,
            "moves": len(seq.get("moves") or []),
            "share": share,
            "dir": seq_path.parent,
        })
    return attempts


def rank_key(a: dict):
    return (not a["topped"], a["time"] if a["time"] is not None else 1e9, a["holds"])


def coach_tip(a: dict, field: list[dict]) -> str:
    s = a["share"]
    hands = sum(s.get(k, 0.0) for k in HANDS)
    feet = sum(s.get(k, 0.0) for k in FEET)
    lh, rh = s.get("left_hand", 0.0), s.get("right_hand", 0.0)
    fewest = min((x["holds"] for x in field if x["holds"]), default=0)

    if not a["topped"]:
        return "Didn't reach the top hold. Try matching both hands on the finish."
    if hands and feet and hands > 2 * feet:
        return f"Arms doing the work ({hands:.0%} hand contact). Trust your feet more."
    if lh + rh and max(lh, rh) / (lh + rh) > 0.65:
        side = "right" if rh > lh else "left"
        return f"Favouring your {side} hand ({max(lh, rh) / (lh + rh):.0%}). Try leading with the other."
    if fewest and a["holds"] > fewest + 1:
        return f"Used {a['holds']} holds; the best line used {fewest}. Look for skips."
    return "Clean and balanced climb. Try it faster."


def fmt_time(t) -> str:
    return f"{t:.1f}s" if t is not None else "—"


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")   # medals on a Windows console
    if argv and argv[0] != "--all":
        root = Path(argv[0])
    elif argv and argv[0] == "--all":
        root = OUTPUT_DIR
    else:
        runs = sorted((p for p in OUTPUT_DIR.glob("*") if p.is_dir()), reverse=True)
        root = runs[0] if runs else OUTPUT_DIR

    attempts = load_attempts(root) if root.exists() else []
    if not attempts:
        print(f"No finished runs under {root}. Run `python main.py` first.")
        return 1

    attempts.sort(key=rank_key)
    lines = ["| # | Climber | Topped | Time | Holds | Moves | Coach tip |",
             "|---|---|---|---|---|---|---|"]
    for i, a in enumerate(attempts, start=1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, str(i))
        lines.append(f"| {medal} | {a['name']} | {'yes' if a['topped'] else 'no'} | "
                     f"{fmt_time(a['time'])} | {a['holds']} | {a['moves']} | "
                     f"{coach_tip(a, attempts)} |")
    table = "\n".join(lines)

    out = root / "leaderboard.md"
    out.write_text("# Build & Boulder leaderboard\n\n" + table + "\n", encoding="utf-8")
    print(table)
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
