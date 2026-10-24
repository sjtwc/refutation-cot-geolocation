import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoloc.metrics import compute_metrics, mcnemar_exact  # noqa: E402


class MetricsTest(unittest.TestCase):
    def test_unparsed_counts_as_wrong(self):
        records = [
            {"country_true": "France", "country_pred": "France"},
            {"country_true": "France", "country_pred": "Spain"},
            {"country_true": "Spain", "country_pred": None},
        ]
        summary = compute_metrics(records)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["parsed"], 2)
        self.assertEqual(summary["unparsed"], 1)
        self.assertEqual(summary["correct"], 1)
        self.assertAlmostEqual(summary["accuracy"], 1 / 3)
        self.assertAlmostEqual(summary["coverage"], 2 / 3)
        self.assertAlmostEqual(summary["accuracy_parsed"], 1 / 2)

    def test_per_country_and_confusion(self):
        records = [
            {"country_true": "France", "country_pred": "France"},
            {"country_true": "France", "country_pred": "Spain"},
            {"country_true": "Spain", "country_pred": None},
        ]
        summary = compute_metrics(records)
        rows = {row["country"]: row for row in summary["per_country"]}
        self.assertEqual(rows["France"]["support"], 2)
        self.assertEqual(rows["France"]["correct"], 1)
        self.assertEqual(rows["Spain"]["unparsed"], 1)
        self.assertEqual(summary["confusion"]["Spain"]["UNPARSED"], 1)
        self.assertEqual(summary["confusion"]["France"]["Spain"], 1)

    def test_empty(self):
        summary = compute_metrics([])
        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["accuracy"], 0.0)
        self.assertEqual(summary["coverage"], 0.0)

    def test_mcnemar(self):
        self.assertAlmostEqual(mcnemar_exact(10, 0), 2 * 0.5**10)
        self.assertEqual(mcnemar_exact(5, 5), 1.0)
        self.assertEqual(mcnemar_exact(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
