"""Manifest loading, hashing, and path resolution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON list or a JSONL file into a list of dictionaries."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list of records")
    return records


def compute_manifest_sha256(records: Iterable[dict[str, Any]]) -> str:
    """Hash manifest content independently of file formatting."""
    payload = json.dumps(
        list(records), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image_path(record: dict[str, Any], image_root: str | Path | None) -> Path:
    """Resolve an image path against an optional dataset root.

    ``image_root`` is the directory that contains the ``compressed_dataset``
    folder, matching the relative paths stored in the manifests.
    """
    path = Path(record["image_path"])
    if path.is_absolute() or image_root is None:
        return path
    return Path(image_root) / path
