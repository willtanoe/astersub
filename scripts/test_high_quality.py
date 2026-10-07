import json
from pathlib import Path
import unittest

from high_quality import cleanup, format_english, validate_response
from transcribe import natural_sort_key


class PipelineTests(unittest.TestCase):
    def test_response_rejects_changed_duplicate_missing_and_empty_ids(self):
        targets = [dict(id=1, text_zh="你好"), dict(id=2, text_zh="再见")]
        for response in (
            [dict(id=2, en="Bye"), dict(id=1, en="Hi")],
            [dict(id=1, en="Hi"), dict(id=1, en="Bye")],
            [dict(id=1, en="Hi")],
            [dict(id=1, en=""), dict(id=2, en="Bye")],
        ):
            with self.assertRaises(ValueError):
                validate_response(json.dumps({"translations": response}), targets)
        with self.assertRaises(ValueError):
            validate_response("not JSON", targets)

    def test_object_and_exact_fence(self):
        targets = [dict(id=1, text_zh="你好")]
        content = json.dumps({"translations": [dict(id=1, en='He said "Hello".')]})
        self.assertEqual(validate_response(content, targets)[0]["id"], 1)
        self.assertEqual(validate_response("```json\n" + content + "\n```", targets)[0]["id"], 1)
        with self.assertRaises(ValueError):
            validate_response('[{"id":1,"en":"Hi"}]', targets)

    def test_outer_suffix_is_exact(self):
        from high_quality import parse_completion
        self.assertEqual(parse_completion('{}\n\ndata: [DONE]\n'), {})
        with self.assertRaises(ValueError):
            parse_completion('{}\ndata: [DONE]\nother text')

    def test_cleanup_preserves_repetition_after_silence_and_no_overlap(self):
        result = cleanup([
            dict(start=0, end=1, text_zh="你好。"),
            dict(start=.9, end=1.2, text_zh="你好。"),
            dict(start=3, end=4, text_zh="你好。"),
            dict(start=3.9, end=5, text_zh="再见。"),
        ])
        self.assertEqual(len(result), 3)
        # Source cleanup keeps speech anchors; final presentation fixes overlaps.
        self.assertEqual(result[-1]["start"], 3.9)

    def test_format_keeps_words_and_proper_name(self):
        source = "Wu Tian must return to the ancestral temple before the elders arrive."
        formatted = format_english(source, {"characters": {"吴天": "Wu Tian"}})
        self.assertEqual(" ".join(formatted.split()), source)
        self.assertIn("Wu Tian", formatted)
        self.assertLessEqual(len(formatted.splitlines()), 2)

    def test_generic_numeric_order(self):
        paths = [Path("Episode 10.mp4"), Path("Episode 2.mp4"), Path("Episode 1.mp4")]
        self.assertEqual([p.name for p in sorted(paths, key=natural_sort_key)],
                         ["Episode 1.mp4", "Episode 2.mp4", "Episode 10.mp4"])


if __name__ == "__main__":
    unittest.main()
