#!/usr/bin/env python3
"""Re-score archived response files with the strict country parser."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoloc.countries import canonicalize_label, load_alias_data  # noqa: E402
from geoloc.manifest import load_records  # noqa: E402
from geoloc.metrics import compute_metrics, format_summary, write_metrics  # noqa: E402
from geoloc.parsing import default_matcher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Archived result file; may be repeated",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "tables" / "rescored"
    )
    return parser.parse_args()


def inputs_from_args(values: list[str]) -> list[tuple[str, Path]]:
    if values:
        pairs = []
        for value in values:
            if "=" not in value:
                raise SystemExit(f"--input must be LABEL=PATH, got {value!r}")
            label, path = value.split("=", 1)
            pairs.append((label, Path(path)))
        return pairs
    return [(path.stem, path) for path in sorted((ROOT / "results" / "raw").glob("*.json"))]


def main() -> None:
    args = parse_args()
    pairs = inputs_from_args(args.input)
    if not pairs:
        raise SystemExit("no input files found")

    _, labels = load_alias_data()
    matcher = default_matcher()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, path in pairs:
        records = load_records(path)
        rescored = []
        old_correct = []
        for row in records:
            true_label = row.get("true_country") or row.get("country")
            if not true_label:
                raise SystemExit(f"{path}: missing true country")
            country_true = canonicalize_label(true_label, labels)
            parsed = matcher.parse(row.get("response", ""))
            if isinstance(row.get("country_correct"), bool):
                old_correct.append(row["country_correct"])
            rescored.append(
                {
                    "filename": row.get("filename") or row.get("image") or "",
                    "country_true": country_true,
                    "country_pred": parsed.country,
                    "parse_method": parsed.method,
                    "old_country_correct": row.get("country_correct"),
                }
            )

        summary = compute_metrics(rescored)
        write_metrics(args.out_dir, summary, prefix=label)
        (args.out_dir / f"{label}.json").write_text(
            json.dumps(rescored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        old_accuracy = sum(old_correct) / len(old_correct) if old_correct else None
        rows.append(
            {
                "run": label,
                "images": summary["total"],
                "old_accuracy": old_accuracy,
                "strict_accuracy": summary["accuracy"],
                "coverage": summary["coverage"],
                "unparsed": summary["unparsed"],
                "accuracy_parsed": summary["accuracy_parsed"],
                "delta": None if old_accuracy is None else summary["accuracy"] - old_accuracy,
            }
        )
        print(format_summary(summary, label=label))
        print()

    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Rescored archived results",
        "",
        "The old `country_correct` field used naive substring matching. Unparsed responses",
        "were counted as wrong and were never mapped to a ground-truth centroid.",
        "",
        "| Run | Images | Old accuracy | Strict accuracy | Coverage | Unparsed | Change |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        old = "n/a" if row["old_accuracy"] is None else f"{row['old_accuracy'] * 100:.1f}%"
        change = "n/a" if row["delta"] is None else f"{row['delta'] * 100:+.1f} pp"
        lines.append(
            f"| {row['run']} | {row['images']} | {old} | "
            f"{row['strict_accuracy'] * 100:.1f}% | {row['coverage'] * 100:.1f}% | "
            f"{row['unparsed']} | {change} |"
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {args.out_dir / 'summary.csv'}")
    print(f"wrote {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
