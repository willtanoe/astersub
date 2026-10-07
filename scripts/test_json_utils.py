import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from json_utils import to_json_safe, save_json


class JsonSafeTests(unittest.TestCase):
    def test_nested_numpy_conversion(self):
        raw = {
            "count": np.int64(44),
            "metrics": {"max_cps": np.float64(31.4), "valid": np.bool_(True)},
            "ids": np.array([1, 2, 3], dtype=np.int64),
            "small": np.int32(7),
        }
        safe = to_json_safe(raw)
        self.assertEqual(safe["count"], 44)
        self.assertIs(type(safe["count"]), int)
        self.assertAlmostEqual(safe["metrics"]["max_cps"], 31.4)
        self.assertIs(type(safe["metrics"]["valid"]), bool)
        self.assertEqual(safe["ids"], [1, 2, 3])
        self.assertIs(type(safe["small"]), int)

    def test_serializes_without_string_coercion(self):
        safe = to_json_safe({"n": np.int64(5), "f": np.float64(1.5)})
        text = json.dumps(safe)
        self.assertIn('"n": 5', text)
        self.assertIn('"f": 1.5', text)
        self.assertNotIn('"5"', text)

    def test_unsupported_type_raises(self):
        class Weird:
            pass
        with self.assertRaises(TypeError):
            to_json_safe({"x": Weird()})

    def test_atomic_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out.json"
            save_json(target, {"ids": np.array([1, 2, 3]), "b": np.bool_(False)})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")),
                             {"ids": [1, 2, 3], "b": False})
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
