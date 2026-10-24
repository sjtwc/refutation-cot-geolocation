#!/usr/bin/env python3
"""Build deterministic, training-disjoint evaluation manifests."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoloc.countries import canonicalize_label, load_alias_data  # noqa: E402
from geoloc.manifest import compute_manifest_sha256, file_sha256, load_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        default=ROOT / "data" / "manifests" / "synthetic_pool_6434.json",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT / "data" / "synthetic" / "refutation_1k_final.json",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data" / "manifests"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-4800", type=int, default=4800)
    parser.add_argument("--eval-1000", type=int, default=1000)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    pool = load_records(args.pool)
    train = load_records(args.train)
    _, labels = load_alias_data()

    pool_paths = [row["image_path"] for row in pool]
    if len(pool_paths) != len(set(pool_paths)):
        raise SystemExit("pool contains duplicate image paths")

    train_paths = [row["image"] for row in train]
    if len(train_paths) != len(set(train_paths)):
        raise SystemExit("training file contains duplicate image paths")

    missing = sorted(set(train_paths) - set(pool_paths))
    if missing:
        raise SystemExit(
            f"{len(missing)} training images are absent from the pool, e.g. {missing[:3]}"
        )

    train_records = [
        {
            "image_path": row["image"],
            "filename": Path(row["image"]).name,
            "country": row["country"],
            "country_canonical": canonicalize_label(row["country"], labels),
            "true_lat": row["true_lat"],
            "true_lon": row["true_lon"],
            "coordinate_source": "country_centroid_plus_noise_not_gps",
            "trace_file": args.train.name,
        }
        for row in train
    ]

    excluded = set(train_paths)
    candidates = [row for row in pool if row["image_path"] not in excluded]
    if len(candidates) < args.eval_4800:
        raise SystemExit(
            f"only {len(candidates)} non-training images, need {args.eval_4800}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    eval_4800 = candidates[: args.eval_4800]

    subset_rng = random.Random(args.seed + 1)
    eval_1000 = subset_rng.sample(eval_4800, args.eval_1000)

    def to_eval_record(row: dict, split: str) -> dict:
        return {
            "image_path": row["image_path"],
            "filename": Path(row["image_path"]).name,
            "country": row["country"],
            "country_canonical": canonicalize_label(row["country"], labels),
            "source": "synthetic_pool_6434",
            "split": split,
        }

    eval_4800_records = [to_eval_record(row, "eval_4800") for row in eval_4800]
    eval_1000_records = [to_eval_record(row, "eval_1000") for row in eval_1000]

    overlap_4800 = {row["image_path"] for row in eval_4800_records} & excluded
    overlap_1000 = {row["image_path"] for row in eval_1000_records} & excluded
    if overlap_4800 or overlap_1000:
        raise SystemExit("evaluation manifests overlap the training set")

    eval_4800_paths = {row["image_path"] for row in eval_4800_records}
    if not {row["image_path"] for row in eval_1000_records} <= eval_4800_paths:
        raise SystemExit("eval_1000 is not a subset of eval_4800")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train_967.json": train_records,
        "eval_4800.json": eval_4800_records,
        "eval_1000.json": eval_1000_records,
    }
    for name, records in outputs.items():
        write_json(args.out_dir / name, records)

    index = {
        "version": 1,
        "seed": args.seed,
        "notes": [
            "eval_4800 is sampled from the pool after excluding all 967 training image paths",
            "eval_1000 is a nested random subset of eval_4800",
            "coordinates in train_967 are country centroids plus noise, not GPS",
        ],
        "source": {
            "pool": str(args.pool.relative_to(ROOT)),
            "pool_count": len(pool),
            "pool_file_sha256": file_sha256(args.pool),
            "train": str(args.train.relative_to(ROOT)),
            "train_file_sha256": file_sha256(args.train),
        },
        "manifests": {
            name: {
                "path": str((args.out_dir / name).relative_to(ROOT)),
                "count": len(records),
                "sha256": compute_manifest_sha256(records),
            }
            for name, records in outputs.items()
        },
        "disjointness": {
            "excluded_train_images": len(excluded),
            "remaining_pool": len(candidates),
            "eval_4800_train_overlap": len(overlap_4800),
            "eval_1000_train_overlap": len(overlap_1000),
            "eval_1000_subset_of_eval_4800": True,
        },
    }
    write_json(args.out_dir / "manifest_index.json", index)

    for name, records in outputs.items():
        print(f"{name}: {len(records)} images, sha256={index['manifests'][name]['sha256']}")
    print(
        f"excluded {len(excluded)} training images; "
        f"{len(candidates)} candidates remained for evaluation"
    )


if __name__ == "__main__":
    main()
