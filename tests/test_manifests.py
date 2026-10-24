import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoloc.manifest import compute_manifest_sha256, load_records  # noqa: E402


class ManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_dir = ROOT / "data" / "manifests"
        cls.index = json.loads(
            (cls.manifest_dir / "manifest_index.json").read_text(encoding="utf-8")
        )
        cls.train = load_records(cls.manifest_dir / "train_967.json")
        cls.eval_4800 = load_records(cls.manifest_dir / "eval_4800.json")
        cls.eval_1000 = load_records(cls.manifest_dir / "eval_1000.json")

    def test_counts(self):
        self.assertEqual(len(self.train), 967)
        self.assertEqual(len(self.eval_4800), 4800)
        self.assertEqual(len(self.eval_1000), 1000)

    def test_no_training_overlap(self):
        train_paths = {row["image_path"] for row in self.train}
        eval_paths = {row["image_path"] for row in self.eval_4800}
        self.assertFalse(train_paths & eval_paths)
        self.assertFalse(train_paths & {row["image_path"] for row in self.eval_1000})

    def test_eval_1000_is_nested(self):
        subset = {row["image_path"] for row in self.eval_1000}
        superset = {row["image_path"] for row in self.eval_4800}
        self.assertTrue(subset <= superset)

    def test_no_duplicates(self):
        for records in (self.train, self.eval_4800, self.eval_1000):
            paths = [row["image_path"] for row in records]
            self.assertEqual(len(paths), len(set(paths)))

    def test_hashes_match_index(self):
        for name, records in {
            "train_967.json": self.train,
            "eval_4800.json": self.eval_4800,
            "eval_1000.json": self.eval_1000,
        }.items():
            expected = self.index["manifests"][name]["sha256"]
            self.assertEqual(compute_manifest_sha256(records), expected)

    def test_index_reports_disjointness(self):
        disjointness = self.index["disjointness"]
        self.assertEqual(disjointness["eval_4800_train_overlap"], 0)
        self.assertEqual(disjointness["eval_1000_train_overlap"], 0)
        self.assertTrue(disjointness["eval_1000_subset_of_eval_4800"])


if __name__ == "__main__":
    unittest.main()
