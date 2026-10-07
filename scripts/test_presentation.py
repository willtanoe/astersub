import unittest
from subtitle_presentation import refine
from validate_srt import validate


class PresentationTests(unittest.TestCase):
    def test_overlap_shortens_previous_without_cascading_drift(self):
        segments = [dict(id=1, start=10, end=12.5, en="Hello."),
                    dict(id=2, start=12.3, end=14, en="Goodbye.")]
        cues, qc = refine(segments, {}, 0, 0)
        self.assertAlmostEqual(cues[0]["end"], 12.3)
        self.assertEqual(cues[1]["start"], 12.3)
        self.assertEqual(validate(cues)[0], [])

    def test_split_preserves_words_and_source_interval(self):
        text = "The Demon Ancestor has arrived, and the elders are waiting at the gate to welcome him into the ancestral temple."
        cues, qc = refine([dict(id=1, start=10, end=15, en=text)],
                          {"terms": {"x": "Demon Ancestor"}}, 0, 0)
        self.assertEqual(" ".join(" ".join(c["text"].split()) for c in cues), text)
        self.assertEqual(cues[0]["start"], 10)
        self.assertEqual(cues[-1]["end"], 15)
        self.assertEqual(validate(cues)[0], [])

    def test_word_anchors_authoritative(self):
        cues, _ = refine([dict(id=1, start=0, end=5, en="Hello.",
                              words=[dict(start=1, end=2, word="你好")])], {}, 0, 0)
        self.assertEqual((cues[0]["start"], cues[0]["end"]), (1, 2))


if __name__ == "__main__":
    unittest.main()
