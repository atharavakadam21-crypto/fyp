"""Compare two mixed-domain evaluation output folders by class.

The comparison is diagnostic when both folders use the same labeled manifest.
It does not replace evaluation on an untouched test set.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_metrics(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {r["class"]: r for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True, help="Evaluation folder for previous candidate")
    parser.add_argument("--candidate", type=Path, required=True, help="Evaluation folder for new candidate")
    parser.add_argument("--output", type=Path, default=Path("candidate_class_comparison.csv"))
    args = parser.parse_args()

    a_path = args.baseline / "per_class_metrics.csv"
    b_path = args.candidate / "per_class_metrics.csv"
    for path in (a_path, b_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    a, b = read_metrics(a_path), read_metrics(b_path)
    if set(a) != set(b):
        raise ValueError("Class sets differ between evaluation reports.")

    rows = []
    for name in sorted(a):
        support_a, support_b = int(a[name]["support"]), int(b[name]["support"])
        if support_a != support_b:
            raise ValueError(f"Support differs for {name}: {support_a} vs {support_b}; ensure the same manifest was used.")
        rows.append({
            "class": name,
            "support": support_a,
            "baseline_correct": int(a[name]["correct"]),
            "candidate_correct": int(b[name]["correct"]),
            "baseline_f1": float(a[name]["f1"]),
            "candidate_f1": float(b[name]["f1"]),
            "f1_delta": float(b[name]["f1"]) - float(a[name]["f1"]),
            "baseline_recall": float(a[name]["recall"]),
            "candidate_recall": float(b[name]["recall"]),
            "recall_delta": float(b[name]["recall"]) - float(a[name]["recall"]),
        })
    rows.sort(key=lambda r: r["f1_delta"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== PER-CLASS CANDIDATE COMPARISON ==========")
    print("Negative delta means the balanced candidate did worse.")
    print(f"Saved: {args.output.resolve()}")
    print("\nLargest F1 regressions:")
    for r in rows[:8]:
        if r["support"]:
            print(f'{r["f1_delta"]:+.3f}  {r["class"]} (baseline {r["baseline_f1"]:.3f}, candidate {r["candidate_f1"]:.3f}, n={r["support"]})')
    print("\nLargest F1 improvements:")
    for r in reversed(rows[-8:]):
        if r["support"]:
            print(f'{r["f1_delta"]:+.3f}  {r["class"]} (baseline {r["baseline_f1"]:.3f}, candidate {r["candidate_f1"]:.3f}, n={r["support"]})')


if __name__ == "__main__":
    main()
