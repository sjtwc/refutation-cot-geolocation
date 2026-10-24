#!/usr/bin/env python3
"""Build the canonical country alias map used by data and evaluation scripts."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "data" / "manifests" / "synthetic_pool_6434.json"
REST_PATH = ROOT / "data" / "reference" / "countries_rest.json"
OUT_PATH = ROOT / "data" / "reference" / "country_aliases.json"

# Dataset folder names that differ from the reference common name.
LABEL_OVERRIDES = {
    "Aland": "Åland Islands",
    "Curacao": "Curaçao",
    "Macao": "Macau",
    "Reunion": "Réunion",
    "South Georgia and South Sandwich Islands": "South Georgia",
    "Turkey": "Türkiye",
    "US Virgin Islands": "United States Virgin Islands",
}

# Variants that language models emit but that are not in the reference data.
MANUAL_ALIASES = {
    "usa": "United States",
    "u s a": "United States",
    "u s": "United States",
    "united states of america": "United States",
    "america": "United States",
    "new jersey": "United States",
    "u s virgin islands": "United States Virgin Islands",
    "virgin islands us": "United States Virgin Islands",
    "uk": "United Kingdom",
    "u k": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    "uae": "United Arab Emirates",
    "u a e": "United Arab Emirates",
    "emirates": "United Arab Emirates",
    "holland": "Netherlands",
    "czech republic": "Czechia",
    "burma": "Myanmar",
    "swaziland": "Eswatini",
    "macedonia": "North Macedonia",
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "macau": "Macau",
    "macao": "Macau",
    "curacao": "Curaçao",
    "aland": "Åland Islands",
    "aaland": "Åland Islands",
    "reunion": "Réunion",
    "new south wales": "Australia",
    "vatican": "Vatican City",
}

# Aliases that are too ambiguous to match on their own.
BLOCKED_ALIASES = {"us", "and", "the", "in", "of", "a", "s"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("’", "'").replace("'", " ")
    text = re.sub(r"[^a-z0-9\s:]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    rest = json.loads(REST_PATH.read_text(encoding="utf-8"))

    rest_by_common = {row["name"]["common"]: row for row in rest}
    rest_by_normalized: dict[str, dict] = {}
    for row in rest:
        rest_by_normalized[normalize(row["name"]["common"])] = row
        for spelling in row.get("altSpellings", []):
            rest_by_normalized.setdefault(normalize(spelling), row)

    pool_labels = sorted({row["country"] for row in pool})
    label_to_canonical: dict[str, str] = {}
    for label in pool_labels:
        target = LABEL_OVERRIDES.get(label)
        if target is None:
            match = rest_by_normalized.get(normalize(label))
            target = match["name"]["common"] if match else label
        label_to_canonical[label] = target

    aliases: dict[str, str] = {}
    collisions: dict[str, set[str]] = {}

    def add(alias: str, canonical: str) -> None:
        key = normalize(alias)
        if len(key) < 3 and key != "uk":
            return
        if key in BLOCKED_ALIASES:
            return
        previous = aliases.get(key)
        if previous and previous != canonical:
            collisions.setdefault(key, {previous, canonical})
            return
        aliases[key] = canonical

    for row in rest:
        canonical = row["name"]["common"]
        add(canonical, canonical)
        add(row["name"]["official"], canonical)
        for spelling in row.get("altSpellings", []):
            add(spelling, canonical)

    for alias, canonical in MANUAL_ALIASES.items():
        aliases[normalize(alias)] = canonical

    for label, canonical in label_to_canonical.items():
        add(label, canonical)

    payload = {
        "version": 1,
        "source": "countries_rest.json + synthetic_pool_6434.json",
        "label_to_canonical": dict(sorted(label_to_canonical.items())),
        "aliases": dict(sorted(aliases.items())),
    }
    OUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"wrote {OUT_PATH}")
    print(f"labels: {len(label_to_canonical)}  aliases: {len(aliases)}")
    if collisions:
        print(f"skipped {len(collisions)} ambiguous alias(es):")
        for key, values in sorted(collisions.items()):
            print(f"  {key!r}: {sorted(values)}")
    missing = [label for label in pool_labels if label not in rest_by_common and label not in LABEL_OVERRIDES]
    if missing:
        print(f"labels without a reference match: {missing}")


if __name__ == "__main__":
    main()
