"""Strict country-name extraction from model responses.

The parser deliberately never falls back to a ground-truth country. If a response
does not contain a usable country name, the prediction is reported as unparsed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from geoloc.countries import load_alias_data, normalize


@dataclass(frozen=True)
class Mention:
    canonical: str
    start: int
    end: int
    negated: bool


@dataclass(frozen=True)
class ParseResult:
    country: str | None
    method: str
    mentions: tuple[Mention, ...] = ()


_LABEL_RE = re.compile(
    r"(?im)^[ \t]*(?:country|image location|location)[ \t]*(?::|is)[ \t]*"
    r"(?P<value>[^\n]{1,120})$"
)

_NEGATION_RE = re.compile(
    r"\b(?:not|never|neither|nor|cannot|isn t|wasn t|weren t|doesn t|didn t|"
    r"can t|couldn t|wouldn t|won t|ain t)\b"
)
_CONTRAST_RE = re.compile(r"\b(?:but|however|instead|rather|though|although|yet)\b")

_US_ABBREVIATION_RE = re.compile(r"\bU\.S\.A?\.?(?!\s+Virgin)", re.IGNORECASE)
_US_UPPER_RE = re.compile(r"\bUS\b(?!\s+Virgin)")

_CONCLUSION_TAIL_RE = re.compile(
    r"(?:"
    r"(?:this|the)\s+(?:\w+\s+){0,2}(?:image|photo|location|place|scene)\s+"
    r"(?:is|was|appears to be|looks like)\s+(?:located\s+)?"
    r"(?:likely\s+|probably\s+|possibly\s+|most likely\s+)?(?:in\s+)?"
    r"|(?:it|this)\s+(?:is|was)\s+(?:located\s+)?"
    r"(?:likely\s+|probably\s+|possibly\s+)?(?:in\s+)?"
    r"|located\s+in\s+"
    r"|situated\s+in\s+"
    r"|(?:country|answer)\s+(?:is|was)\s+"
    r"(?:likely\s+|probably\s+|possibly\s+)?"
    r"|my\s+(?:guess|answer)\s+is\s+"
    r"|i\s+(?:think|believe|guess)\s+(?:it\s+is|this\s+is|the\s+country\s+is)\s+"
    r")(?:the\s+)?$"
)


def _is_negated(normalized: str, start: int) -> bool:
    window = normalized[max(0, start - 90) : start]
    window = re.split(r"[.;!?\n]", window)[-1]
    matches = list(_NEGATION_RE.finditer(window))
    if not matches:
        return False
    after = window[matches[-1].end() :]
    return not _CONTRAST_RE.search(after)


class CountryMatcher:
    def __init__(self, aliases: dict[str, str]):
        self.aliases = {normalize(key): value for key, value in aliases.items()}
        alternatives = sorted(self.aliases, key=len, reverse=True)
        if not alternatives:
            raise ValueError("alias map is empty")
        self._pattern = re.compile(
            r"(?<![a-z0-9])(?:" + "|".join(map(re.escape, alternatives)) + r")(?![a-z0-9])"
        )

    def find_mentions(
        self,
        text: str,
        *,
        normalized: bool = False,
        negation_text: str | None = None,
    ) -> list[Mention]:
        normalized_text = text if normalized else normalize(text)
        negative_context = negation_text if negation_text is not None else normalized_text
        mentions = []
        for match in self._pattern.finditer(normalized_text):
            alias = match.group(0)
            mentions.append(
                Mention(
                    canonical=self.aliases[alias],
                    start=match.start(),
                    end=match.end(),
                    negated=_is_negated(negative_context, match.start()),
                )
            )
        return mentions

    def parse(self, text: str) -> ParseResult:
        text = _US_ABBREVIATION_RE.sub("USA", text)
        text = _US_UPPER_RE.sub("USA", text)
        normalized = normalize(text, strip=False)
        negation_text = normalize(
            text, keep_sentence_punctuation=True, strip=False
        )
        mentions = self.find_mentions(
            normalized, normalized=True, negation_text=negation_text
        )

        for label in reversed(list(_LABEL_RE.finditer(normalized))):
            value_mentions = self.find_mentions(label.group("value"), normalized=True)
            positive = [mention for mention in value_mentions if not mention.negated]
            if positive:
                return ParseResult(positive[-1].canonical, "label", tuple(mentions))

        positive = [mention for mention in mentions if not mention.negated]
        if not positive:
            return ParseResult(None, "none", tuple(mentions))

        conclusions = [
            mention
            for mention in positive
            if _CONCLUSION_TAIL_RE.search(normalized[max(0, mention.start - 90) : mention.start])
        ]
        if conclusions:
            return ParseResult(conclusions[-1].canonical, "conclusion", tuple(mentions))
        return ParseResult(positive[-1].canonical, "mention", tuple(mentions))


@lru_cache(maxsize=1)
def default_matcher() -> CountryMatcher:
    aliases, _ = load_alias_data()
    return CountryMatcher(aliases)
