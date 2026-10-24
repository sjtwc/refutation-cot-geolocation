import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoloc.countries import canonicalize_label, load_alias_data, normalize  # noqa: E402
from geoloc.parsing import default_matcher  # noqa: E402


class CountryAliasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aliases, cls.labels = load_alias_data()
        cls.matcher = default_matcher()
        pool = json.loads(
            (ROOT / "data" / "manifests" / "synthetic_pool_6434.json").read_text(
                encoding="utf-8"
            )
        )
        cls.pool_labels = sorted({row["country"] for row in pool})

    def test_every_pool_label_canonicalizes(self):
        for label in self.pool_labels:
            self.assertIn(normalize(label), self.labels)
            self.assertTrue(canonicalize_label(label, self.labels))

    def test_every_pool_label_parses_to_its_canonical_form(self):
        for label in self.pool_labels:
            expected = canonicalize_label(label, self.labels)
            parsed = self.matcher.parse(f"Country: {label}")
            self.assertEqual(parsed.country, expected, label)

    def test_common_aliases(self):
        cases = {
            "USA": "United States",
            "U.S.A.": "United States",
            "UK": "United Kingdom",
            "England": "United Kingdom",
            "Holland": "Netherlands",
            "Czech Republic": "Czechia",
            "Burma": "Myanmar",
            "Swaziland": "Eswatini",
            "Macedonia": "North Macedonia",
            "Korea": "South Korea",
            "Türkiye": "Türkiye",
            "Macao": "Macau",
        }
        for alias, expected in cases.items():
            self.assertEqual(self.matcher.parse(f"Country: {alias}").country, expected, alias)

    def test_ambiguous_short_alias_is_blocked(self):
        self.assertNotIn("us", self.aliases)
        parsed = self.matcher.parse("Tell us about this road.")
        self.assertIsNone(parsed.country)


if __name__ == "__main__":
    unittest.main()
