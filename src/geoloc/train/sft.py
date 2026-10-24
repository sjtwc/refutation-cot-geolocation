#!/usr/bin/env python3
"""Config-driven QLoRA training for refutation-CoT and normal-CoT adapters."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from datasets import Dataset, Features, Value  # noqa: E402
from PIL import Image  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer  # noqa: E402


TRACE_FILES = {
    "refutation": ROOT / "data" / "synthetic" / "refutation_1k_final.json",
    "normal": ROOT / "data" / "synthetic" / "normal_1k_final.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(TRACE_FILES), default="refutation")
    parser.add_argument("--trace-file", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--base-model", default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=11)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--wandb-project", default="geoguessr-llava-next")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=50)
    return parser.parse_args()


def resolve_image(image_root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    candidates = [image_root / relative_path]
    if relative_path.parts and relative_path.parts[0] == "compressed_dataset":
        candidates.append(image_root / Path(*relative_path.parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_examples(
    trace_file: Path, image_root: Path, skip_missing: bool, limit: int | None
) -> list[dict[str, Any]]:
    raw = json.loads(trace_file.read_text(encoding="utf-8"))
    if limit is not None:
        raw = raw[:limit]

    examples = []
    missing = []
    for item in raw:
        message = item["messages"]
        if len(message) < 2 or message[0]["role"] != "user" or message[1]["role"] != "assistant":
            raise SystemExit(f"unexpected message structure in {trace_file}")
        image_path = resolve_image(image_root, item["image"])
        if not image_path.exists():
            missing.append(str(image_path))
            if skip_missing:
                continue
            continue
        examples.append(
            {
                "image_path": str(image_path),
                "user_prompt": message[0]["content"].replace("<image>\n", "").strip(),
                "assistant_response": message[1]["content"],
                "country": item["country"],
                "lat": item["true_lat"],
                "lon": item["true_lon"],
            }
        )

    if missing and not skip_missing:
        raise SystemExit(
            f"{len(missing)} training images are missing, e.g. {missing[:3]}. "
            "Pass --skip-missing to train on the remainder."
        )
    if missing:
        print(f"skipped {len(missing)} missing images")
    if not examples:
        raise SystemExit("no training examples found")
    return examples


def build_dataset(examples: list[dict[str, Any]]) -> Dataset:
    features = Features(
        {
            "image_path": Value("string"),
            "user_prompt": Value("string"),
            "assistant_response": Value("string"),
            "country": Value("string"),
            "lat": Value("float64"),
            "lon": Value("float64"),
        }
    )
    return Dataset.from_dict(
        {key: [example[key] for example in examples] for key in examples[0]},
        features=features,
    )


def find_last_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence:
        return -1
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return -1


class LLaVADataCollator:
    def __init__(self, processor, tokenizer, assistant_header: str):
        self.processor = processor
        self.tokenizer = tokenizer
        self.header_ids = tokenizer(
            assistant_header, add_special_tokens=False
        )["input_ids"]

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [example["text"] for example in examples]
        images = [Image.open(example["image_path"]).convert("RGB") for example in examples]
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        labels = batch["input_ids"].clone()
        attention_mask = batch.get("attention_mask")
        for index in range(labels.shape[0]):
            token_ids = batch["input_ids"][index].tolist()
            header_start = find_last_subsequence(token_ids, self.header_ids)
            if header_start < 0:
                raise RuntimeError("could not locate the assistant header in a training example")
            labels[index, : header_start + len(self.header_ids)] = -100
            if attention_mask is not None:
                labels[index, attention_mask[index] == 0] = -100
        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()
    trace_file = args.trace_file or TRACE_FILES[args.mode]
    if not trace_file.is_absolute():
        trace_file = ROOT / trace_file
    if not 0 < args.val_fraction < 0.5:
        raise SystemExit("--val-fraction must be between 0 and 0.5")
    if args.bf16:
        args.fp16 = False

    output_dir = args.output_dir or ROOT / "artifacts" / "checkpoints" / f"llava-geoguessr-{args.mode}"
    run_name = args.run_name or f"geoguessr-{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(trace_file, args.image_root, args.skip_missing, args.limit)
    dataset = build_dataset(examples)
    dataset = dataset.train_test_split(test_size=args.val_fraction, seed=args.seed)

    processor = AutoProcessor.from_pretrained(
        args.base_model, local_files_only=args.local_files_only
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        offload_folder=str(output_dir / "offload"),
        local_files_only=args.local_files_only,
    )
    model = prepare_model_for_kbit_training(model)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    def preprocess(example: dict[str, Any]) -> dict[str, str]:
        text = (
            f"<|im_start|>user\n<image>\n{example['user_prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['assistant_response']}<|im_end|>"
        )
        return {"text": text}

    train_dataset = dataset["train"].map(preprocess)
    eval_dataset = dataset["test"].map(preprocess)
    collator = LLaVADataCollator(processor, processor.tokenizer, "<|im_start|>assistant\n")

    report_to = [] if args.report_to == "none" else args.report_to
    if report_to:
        import wandb

        wandb.init(project=args.wandb_project, name=run_name)

    config_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": max(args.batch_size, 1),
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "logging_steps": 5,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "torch_compile": args.torch_compile,
        "report_to": report_to,
        "run_name": run_name,
        "dataloader_num_workers": 6,
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "max_grad_norm": 1.0,
        "optim": args.optim,
        "dataloader_persistent_workers": True,
        "dataset_text_field": "text",
        "packing": False,
    }
    if args.max_length is not None:
        config_kwargs["max_length"] = args.max_length
    sft_config = SFTConfig(**config_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = processor
    trainer = SFTTrainer(**trainer_kwargs)

    split = {
        "train": [row["image_path"] for row in dataset["train"]],
        "validation": [row["image_path"] for row in dataset["test"]],
    }
    (output_dir / "split.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": {key: str(value) for key, value in vars(args).items()},
                "trace_file": str(trace_file),
                "examples": len(examples),
                "train_examples": len(train_dataset),
                "validation_examples": len(eval_dataset),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    logging.getLogger("transformers").setLevel(logging.INFO)
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    (output_dir / "training_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"adapter saved to {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
