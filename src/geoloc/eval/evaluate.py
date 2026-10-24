#!/usr/bin/env python3
"""Evaluate a base model or LoRA adapter on a frozen country-level manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from geoloc.countries import canonicalize_label, load_alias_data  # noqa: E402
from geoloc.eval.model_loader import load_model  # noqa: E402
from geoloc.manifest import (  # noqa: E402
    compute_manifest_sha256,
    load_records,
    resolve_image_path,
)
from geoloc.metrics import compute_metrics, format_summary, write_metrics  # noqa: E402
from geoloc.parsing import default_matcher  # noqa: E402


PROMPTS = {
    "country": (
        "Where in the world is this GeoGuessr location? Think step by step, then "
        "end with a single line: Country: <country name>"
    ),
    "train": "Think step by step...",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--adapter", default="none", help="Adapter path or HF repo; 'none' for base")
    parser.add_argument(
        "--base-model", default="llava-hf/llava-v1.6-mistral-7b-hf"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path)
    parser.add_argument("--prompt-style", choices=sorted(PROMPTS), default="country")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model in NF4 4-bit (CUDA only)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Compute dtype; use float16 for a T4",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="auto, cuda, cuda:0, mps, or cpu",
    )
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--allow-train-overlap",
        action="store_true",
        help="Skip the frozen training-manifest overlap check (not recommended)",
    )
    return parser.parse_args()


def check_training_overlap(manifest: list[dict[str, Any]]) -> None:
    train_path = ROOT / "data" / "manifests" / "train_967.json"
    if not train_path.exists():
        return
    train_paths = {row["image_path"] for row in load_records(train_path)}
    overlap = {row["image_path"] for row in manifest} & train_paths
    if overlap:
        raise SystemExit(
            f"manifest overlaps training data on {len(overlap)} images, "
            f"e.g. {sorted(overlap)[:3]}"
        )


def make_prompt(style: str) -> str:
    instruction = PROMPTS[style]
    return (
        f"<|im_start|>user\n<image>\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main() -> None:
    args = parse_args()
    manifest = load_records(args.manifest)
    if not manifest:
        raise SystemExit("manifest is empty")
    manifest_sha = compute_manifest_sha256(manifest)
    if args.limit is not None:
        manifest = manifest[: args.limit]
    if not manifest:
        raise SystemExit("manifest is empty after applying --limit")
    if not args.allow_train_overlap:
        check_training_overlap(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    _, labels = load_alias_data()
    matcher = default_matcher()
    prompt = make_prompt(args.prompt_style)

    partial_path = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    completed: list[dict[str, Any]] = []
    done_paths: set[str] = set()
    if args.resume and partial_path.exists():
        completed = load_records(partial_path)
        done_paths = {row["image_path"] for row in completed}
        print(f"resuming with {len(completed)} completed images")
    elif partial_path.exists():
        partial_path.unlink()

    model = processor = torch = None
    for index, item in enumerate(manifest, start=1):
        if item["image_path"] in done_paths:
            continue
        if model is None:
            model, processor, torch = load_model(
                args.base_model,
                args.adapter,
                local_files_only=args.local_files_only,
                load_in_4bit=args.load_in_4bit,
                dtype=args.dtype,
                device_map=args.device_map,
                attn_implementation=args.attn_implementation,
            )
            print(
                f"loaded {args.base_model} adapter={args.adapter} "
                f"dtype={args.dtype} device_map={args.device_map} "
                f"4bit={args.load_in_4bit}"
            )

        image_path = resolve_image_path(item, args.image_root)
        if not image_path.exists():
            raise SystemExit(f"missing image: {image_path}")

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        response = processor.decode(generated, skip_special_tokens=True).strip()

        parsed = matcher.parse(response)
        country_true = item.get("country_canonical") or canonicalize_label(
            item.get("country", ""), labels
        )
        completed.append(
            {
                "image_path": item["image_path"],
                "filename": item.get("filename", Path(item["image_path"]).name),
                "country_true": country_true,
                "country_pred": parsed.country,
                "parse_method": parsed.method,
                "response": response,
            }
        )
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(completed[-1], ensure_ascii=False) + "\n")

        if index % 10 == 0 or index == len(manifest):
            print(f"[{index}/{len(manifest)}] last={parsed.country!r}")

    by_path = {row["image_path"]: row for row in completed}
    predictions = [
        by_path[row["image_path"]] for row in manifest if row["image_path"] in by_path
    ]
    summary = compute_metrics(predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "manifest_sha256": manifest_sha,
                "base_model": args.base_model,
                "adapter": args.adapter,
                "prompt_style": args.prompt_style,
                "prompt": prompt,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "summary": {k: v for k, v in summary.items() if k not in {"per_country", "confusion"}},
                "predictions": predictions,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    tables_dir = args.tables_dir or args.output.parent / f"{args.output.stem}_tables"
    write_metrics(tables_dir, summary)
    if partial_path.exists():
        partial_path.unlink()

    print()
    print(format_summary(summary, label=args.output.name))
    print(f"predictions -> {args.output}")
    print(f"tables      -> {tables_dir}")


if __name__ == "__main__":
    main()
