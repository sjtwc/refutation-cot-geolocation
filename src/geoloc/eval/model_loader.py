"""Load the base LLaVA model or a PEFT adapter for evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


DTYPES = {
    "auto": None,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def detect_device_kind(device_map: str) -> str:
    if device_map == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_map.startswith("cuda"):
        return "cuda"
    if device_map == "mps":
        return "mps"
    return "cpu"


def choose_dtype(name: str, device_kind: str, bf16_supported: bool = True) -> torch.dtype:
    if name not in DTYPES:
        raise ValueError(f"unknown dtype {name!r}")
    if name != "auto":
        return DTYPES[name]
    if device_kind == "cuda":
        return torch.bfloat16 if bf16_supported else torch.float16
    if device_kind == "mps":
        return torch.bfloat16
    return torch.float32


def normalize_device_map(device_map: str) -> str | dict[str, str]:
    if device_map == "auto":
        return "auto"
    return {"": device_map}


def validate_4bit(load_in_4bit: bool, cuda_available: bool) -> None:
    if load_in_4bit and not cuda_available:
        raise ValueError(
            "--load-in-4bit requires CUDA. On macOS or CPU, use --dtype bfloat16 "
            "or --dtype float16 without 4-bit."
        )


def load_processor(
    base_model: str, adapter_path: str | None, local_files_only: bool
) -> Any:
    from transformers import AutoProcessor

    sources = []
    if adapter_path:
        sources.append(adapter_path)
    sources.append(base_model)
    last_error = None
    for source in sources:
        try:
            processor = AutoProcessor.from_pretrained(
                source, local_files_only=local_files_only
            )
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            return processor
        except Exception as error:  # adapter dirs may not contain processor files
            last_error = error
    raise RuntimeError(f"could not load a processor for {base_model}") from last_error


def load_model(
    base_model: str,
    adapter: str = "none",
    *,
    local_files_only: bool = False,
    load_in_4bit: bool = False,
    dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: str | None = None,
) -> tuple[Any, Any, Any]:
    from transformers import LlavaNextForConditionalGeneration

    adapter_path = None if adapter.lower() == "none" else adapter
    validate_4bit(load_in_4bit, torch.cuda.is_available())
    device_kind = detect_device_kind(device_map)
    torch_dtype = choose_dtype(
        dtype, device_kind, torch.cuda.is_bf16_supported() if device_kind == "cuda" else True
    )

    model_kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "device_map": normalize_device_map(device_map),
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.float16 if torch_dtype == torch.float16 else torch.bfloat16
            ),
            bnb_4bit_use_double_quant=True,
        )

    processor = load_processor(base_model, adapter_path, local_files_only)
    base = LlavaNextForConditionalGeneration.from_pretrained(base_model, **model_kwargs)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base
    model.eval()
    return model, processor, torch
