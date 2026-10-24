import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoloc.parsing import default_matcher  # noqa: E402


class ParsingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = default_matcher()

    def assert_country(self, text, expected):
        self.assertEqual(self.matcher.parse(text).country, expected, text)

    def test_explicit_label(self):
        self.assert_country("Country: Finland", "Finland")
        self.assert_country("Image location: Switzerland\nReasoning follows.", "Switzerland")
        self.assert_country("Country: USA", "United States")
        self.assert_country("The location is in the US.", "United States")
        self.assert_country("Country: U.S.A.", "United States")
        self.assert_country("Country: Korea", "South Korea")
        self.assert_country("Country: Åland", "Åland Islands")
        self.assert_country("Country: Türkiye", "Türkiye")

    def test_negation_is_not_a_match(self):
        self.assert_country("This is NOT Finland.", None)
        self.assert_country("I cannot determine the country from this image.", None)
        self.assert_country("This is not Finland, but Sweden.", "Sweden")
        self.assert_country("Not Finland. Sweden.", "Sweden")

    def test_negation_does_not_cross_sentences(self):
        self.assert_country("It is not Finland. The plate is Swiss.", None)

    def test_longest_alias_wins(self):
        self.assert_country("This is Northern Ireland.", "United Kingdom")
        self.assert_country("The location is in New Jersey.", "United States")
        self.assert_country("This is South Sudan.", "South Sudan")
        self.assert_country("This is American Samoa.", "American Samoa")
        self.assert_country("This is Nigeria.", "Nigeria")

    def test_conclusion_beats_later_rejected_candidates(self):
        text = (
            "This GeoGuessr location is in Canada.\n"
            "It is not the United States, and the vegetation is not Australian."
        )
        self.assert_country(text, "Canada")

    def test_last_label_wins(self):
        self.assert_country("Country: United States\nCountry: Canada", "Canada")

    def test_unparsed_is_none(self):
        self.assert_country("The image shows a road and some trees.", None)
        self.assert_country("Country: None", None)

    def test_negated_only_mentions_are_unparsed(self):
        result = self.matcher.parse("This is not Finland or Sweden.")
        self.assertIsNone(result.country)
        self.assertTrue(all(mention.negated for mention in result.mentions))


if __name__ == "__main__":
    unittest.main()
