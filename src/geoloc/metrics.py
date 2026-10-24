"""Country-level evaluation metrics without distance or centroid fallbacks."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


UNPARSED = "UNPARSED"


def compute_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    total = len(records)
    parsed = sum(1 for row in records if row.get("country_pred"))
    correct = sum(
        1
        for row in records
        if row.get("country_pred") and row.get("country_pred") == row.get("country_true")
    )

    per_country: list[dict[str, Any]] = []
    by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_country[row.get("country_true", "UNKNOWN")].append(row)

    for country, rows in by_country.items():
        support = len(rows)
        country_correct = sum(
            1
            for row in rows
            if row.get("country_pred") and row.get("country_pred") == country
        )
        per_country.append(
            {
                "country": country,
                "support": support,
                "correct": country_correct,
                "unparsed": sum(1 for row in rows if not row.get("country_pred")),
                "accuracy": country_correct / support if support else 0.0,
            }
        )
    per_country.sort(key=lambda row: (-row["support"], row["country"]))

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        predicted = row.get("country_pred") or UNPARSED
        confusion[row.get("country_true", "UNKNOWN")][predicted] += 1

    macro = (
        sum(row["accuracy"] for row in per_country) / len(per_country)
        if per_country
        else 0.0
    )
    return {
        "total": total,
        "parsed": parsed,
        "unparsed": total - parsed,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_parsed": correct / parsed if parsed else 0.0,
        "coverage": parsed / total if total else 0.0,
        "macro_accuracy": macro,
        "countries": len(per_country),
        "per_country": per_country,
        "confusion": {key: dict(value) for key, value in confusion.items()},
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    limit = min(b, c)
    log_terms = [
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        - n * math.log(2)
        for k in range(limit + 1)
    ]
    maximum = max(log_terms)
    log_tail = maximum + math.log(math.fsum(math.exp(term - maximum) for term in log_terms))
    return min(1.0, 2.0 * math.exp(log_tail))


def write_metrics(
    out_dir: str | Path, summary: dict[str, Any], prefix: str | None = None
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{prefix}_" if prefix else ""

    serializable = {key: value for key, value in summary.items() if key not in {"per_country", "confusion"}}
    (out_dir / f"{prefix}summary.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with (out_dir / f"{prefix}per_country.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["country", "support", "correct", "unparsed", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(summary["per_country"])

    predicted_labels = sorted(
        {
            predicted
            for counts in summary["confusion"].values()
            for predicted in counts
        }
    )
    true_labels = sorted(summary["confusion"])
    with (out_dir / f"{prefix}confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_country", *predicted_labels])
        for true_label in true_labels:
            counts = summary["confusion"][true_label]
            writer.writerow([true_label, *[counts.get(label, 0) for label in predicted_labels]])


def format_summary(summary: dict[str, Any], label: str = "results") -> str:
    return "\n".join(
        [
            f"{label}",
            f"  images: {summary['total']}",
            f"  country accuracy: {summary['accuracy'] * 100:.2f}%",
            f"  parsed coverage:  {summary['coverage'] * 100:.2f}% "
            f"({summary['parsed']}/{summary['total']}, {summary['unparsed']} unparsed)",
            f"  accuracy | parsed: {summary['accuracy_parsed'] * 100:.2f}%",
            f"  macro accuracy:   {summary['macro_accuracy'] * 100:.2f}% "
            f"over {summary['countries']} countries",
        ]
    )
