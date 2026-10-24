import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from geoloc.eval.model_loader import (  # noqa: E402
    choose_dtype,
    normalize_device_map,
    validate_4bit,
)


class ModelLoaderTest(unittest.TestCase):
    def test_auto_dtype_by_device(self):
        self.assertEqual(choose_dtype("auto", "cuda", True), torch.bfloat16)
        self.assertEqual(choose_dtype("auto", "cuda", False), torch.float16)
        self.assertEqual(choose_dtype("auto", "mps"), torch.bfloat16)
        self.assertEqual(choose_dtype("auto", "cpu"), torch.float32)
        self.assertEqual(choose_dtype("float16", "mps"), torch.float16)

    def test_device_map(self):
        self.assertEqual(normalize_device_map("auto"), "auto")
        self.assertEqual(normalize_device_map("mps"), {"": "mps"})
        self.assertEqual(normalize_device_map("cuda:0"), {"": "cuda:0"})

    def test_4bit_requires_cuda(self):
        with self.assertRaises(ValueError):
            validate_4bit(True, False)
        validate_4bit(True, True)
        validate_4bit(False, False)


if __name__ == "__main__":
    unittest.main()
