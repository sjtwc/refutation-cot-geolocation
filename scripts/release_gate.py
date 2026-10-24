#!/usr/bin/env python3
"""Release gate for the refutation-CoT adapter.

Compares paired predictions on the same frozen manifest. The gate passes only if
the refutation model beats zero-shot by at least 5 percentage points, beats
normal-CoT with a two-sided exact McNemar p < 0.05, and parses at least 95% of
responses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoloc.metrics import mcnemar_exact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refutation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="Zero-shot predictions")
    parser.add_argument("--normal", type=Path, help="Normal-CoT predictions")
    parser.add_argument("--min-gain", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def load_predictions(path: Path) -> tuple[dict[str, dict], str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        predictions = payload.get("predictions")
        manifest_sha = payload.get("manifest_sha256")
    else:
        predictions = payload
        manifest_sha = None
    if not isinstance(predictions, list):
        raise SystemExit(f"{path}: expected a prediction list")
    indexed = {}
    for row in predictions:
        image_path = row.get("image_path")
        if not image_path:
            raise SystemExit(f"{path}: prediction without image_path")
        indexed[image_path] = row
    return indexed, manifest_sha


def score(rows: list[dict]) -> tuple[float, float]:
    total = len(rows)
    correct = sum(
        1
        for row in rows
        if row.get("country_pred") and row.get("country_pred") == row.get("country_true")
    )
    coverage = sum(1 for row in rows if row.get("country_pred")) / total if total else 0.0
    return correct / total if total else 0.0, coverage


def main() -> None:
    args = parse_args()
    ref, ref_sha = load_predictions(args.refutation)
    base, base_sha = load_predictions(args.baseline)
    normal, normal_sha = load_predictions(args.normal) if args.normal else ({}, None)

    for label, other in (("baseline", base), ("normal", normal)):
        if other and set(other) != set(ref):
            raise SystemExit(f"{label} predictions do not cover the same images as refutation")
    for label, other_sha in (("baseline", base_sha), ("normal", normal_sha)):
        if ref_sha and other_sha and ref_sha != other_sha:
            raise SystemExit(f"{label} was evaluated on a different manifest hash")

    paths = sorted(ref)
    ref_rows = [ref[path] for path in paths]
    base_rows = [base[path] for path in paths]
    normal_rows = [normal[path] for path in paths] if normal else []

    ref_accuracy, ref_coverage = score(ref_rows)
    base_accuracy, _ = score(base_rows)
    gain = ref_accuracy - base_accuracy

    def pair_counts(other_rows: list[dict]) -> tuple[int, int]:
        b = c = 0
        for ref_row, other_row in zip(ref_rows, other_rows):
            ref_correct = bool(
                ref_row.get("country_pred")
                and ref_row.get("country_pred") == ref_row.get("country_true")
            )
            other_correct = bool(
                other_row.get("country_pred")
                and other_row.get("country_pred") == other_row.get("country_true")
            )
            b += int(ref_correct and not other_correct)
            c += int(other_correct and not ref_correct)
        return b, c

    normal_accuracy = None
    normal_p = None
    if normal:
        normal_accuracy, _ = score(normal_rows)
        b, c = pair_counts(normal_rows)
        normal_p = mcnemar_exact(b, c)

    checks = {
        "gain_at_least_min": gain >= args.min_gain,
        "coverage_at_least_min": ref_coverage >= args.min_coverage,
        "normal_mcnemar_significant": (
            normal_p is not None and normal_p < args.alpha
        ),
        "normal_comparison_present": bool(normal),
    }
    reasons = []
    if not checks["gain_at_least_min"]:
        reasons.append(
            f"gain {gain * 100:.1f} pp is below {args.min_gain * 100:.1f} pp"
        )
    if not checks["coverage_at_least_min"]:
        reasons.append(
            f"coverage {ref_coverage * 100:.1f}% is below {args.min_coverage * 100:.1f}%"
        )
    if not checks["normal_comparison_present"]:
        reasons.append("normal-CoT baseline was not provided")
    elif not checks["normal_mcnemar_significant"]:
        reasons.append(
            f"McNemar p={normal_p:.4f} is not below {args.alpha}"
        )

    passed = all(checks.values())
    result = {
        "passed": passed,
        "images": len(paths),
        "refutation_accuracy": ref_accuracy,
        "baseline_accuracy": base_accuracy,
        "normal_accuracy": normal_accuracy,
        "gain_over_baseline": gain,
        "refutation_coverage": ref_coverage,
        "normal_mcnemar_p": normal_p,
        "thresholds": {
            "min_gain": args.min_gain,
            "min_coverage": args.min_coverage,
            "alpha": args.alpha,
        },
        "checks": checks,
        "reasons": reasons,
        "manifest_sha256": ref_sha,
    }

    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
