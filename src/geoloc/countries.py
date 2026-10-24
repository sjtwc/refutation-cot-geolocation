"""Country labels, aliases, and canonicalization."""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ALIAS_PATH = REPO_ROOT / "data" / "reference" / "country_aliases.json"


def normalize(
    text: str, *, keep_sentence_punctuation: bool = False, strip: bool = True
) -> str:
    """Normalize text for country matching while keeping newlines and colons.

    With ``keep_sentence_punctuation``, periods and similar marks are preserved
    instead of replaced by spaces. Both modes preserve character positions, so
    spans from one string can be used to inspect the other when ``strip=False``.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("’", "'").replace("'", " ")
    if keep_sentence_punctuation:
        text = re.sub(r"[^a-z0-9\s:.;!?]", " ", text)
    else:
        text = re.sub(r"[^a-z0-9\s:]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip() if strip else text


@lru_cache(maxsize=4)
def load_alias_data(path: str | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(aliases, label_to_canonical)`` with normalized alias keys."""
    if path is None:
        path = str(ALIAS_PATH)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    aliases = {normalize(key): value for key, value in payload["aliases"].items()}
    labels = {
        normalize(key): value for key, value in payload["label_to_canonical"].items()
    }
    return aliases, labels


def canonicalize_label(label: str, labels: dict[str, str] | None = None) -> str:
    """Map a dataset folder label (e.g. ``Turkey``) to its canonical name."""
    if labels is None:
        _, labels = load_alias_data()
    return labels.get(normalize(label), label)
