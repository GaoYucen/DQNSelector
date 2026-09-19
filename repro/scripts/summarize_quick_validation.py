#!/usr/bin/env python3
"""Summarize one-budget exploratory runs and select the best scenario."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-piano", action="store_true")
    args = parser.parse_args()

    root = Path(args.result_root)
    summaries = []
    for baseline_path in sorted(root.glob("*/baseline/results.json")):
        scenario_dir = baseline_path.parent.parent
        trained_path = scenario_dir / "train-2024" / "results.json"
        if not trained_path.exists():
            continue
        combined = [row for row in rows(baseline_path) + rows(trained_path) if int(row["k"]) == args.budget]
        by_method = {row["method"]: float(row["ec"]) for row in combined}
        if "DQNSelector-J" not in by_method:
            continue
        excluded = {"CELF", "DQNSelector-J"}
        competitors = {name: value for name, value in by_method.items() if name not in excluded}
        if args.require_piano and "PIANO" not in competitors:
            continue
        best_method, best_ec = max(competitors.items(), key=lambda item: item[1])
        journal_ec = by_method["DQNSelector-J"]
        summaries.append({
            "scenario": scenario_dir.name,
            "k": args.budget,
            "methods": by_method,
            "best_non_celf_method": best_method,
            "best_non_celf_ec": best_ec,
            "journal_ec": journal_ec,
            "relative_gain": (journal_ec - best_ec) / best_ec if best_ec else None,
        })
    summaries.sort(key=lambda row: row["relative_gain"] if row["relative_gain"] is not None else float("-inf"), reverse=True)
    payload = {"budget": args.budget, "scenarios": summaries, "best": summaries[0] if summaries else None}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()
