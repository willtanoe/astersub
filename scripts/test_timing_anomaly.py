import unittest
from subtitle_presentation import refine


class TimingAnomalyTests(unittest.TestCase):
    def test_real_cue35_internal_gap_trimmed(self):
        # Actual Episode 2 pattern: segment 19s but words split by a ~17.8s gap.
        segment = dict(id=35, start=81.5, end=100.53, text_zh="我问你刚才",
                       en="I asked what you just said.", words=[
                           dict(start=81.5, end=81.94, word="我"),
                           dict(start=81.94, end=82.16, word="问"),
                           dict(start=82.16, end=82.32, word="你"),
                           dict(start=82.32, end=82.56, word="刚"),
                           dict(start=100.35, end=100.53, word="才")])
        following = dict(id=36, start=101.0, end=102.0, en="Next line.",
                         words=[dict(start=101.0, end=102.0, word="next")])
        cues, qc = refine([segment, following], {"characters": {}, "terms": {}}, 0, 0)
        cue35 = next(c for c in cues if c["source_id"] == 35)
        cue36 = next(c for c in cues if c["source_id"] == 36)
        self.assertLessEqual(cue35["end"] - cue35["start"], 6.0)
        self.assertEqual(qc["internal_speech_gaps_detected"], 1)
        self.assertEqual(qc["timing_anomalies_fixed"], 1)
        self.assertEqual(qc["long_duration_cues_after"], 0)
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(cues, cues[1:])))
        self.assertEqual(cue36["start"], 101.0)

    def test_internal_gap_not_shown_across_silence(self):
        segment = dict(id=1, start=0.0, end=11.0, text_zh="测试", en="Hello world again",
                       words=[dict(start=0.0, end=1.0, word="a"),
                              dict(start=1.0, end=1.5, word="b"),
                              dict(start=10.0, end=11.0, word="c")])
        cues, qc = refine([segment], {"characters": {}, "terms": {}}, 0, 0)
        self.assertEqual(len(cues), 1)
        self.assertLessEqual(cues[0]["end"] - cues[0]["start"], 6.0)
        self.assertEqual(qc["internal_speech_gaps_detected"], 1)

    def test_legacy_fallback_without_words(self):
        segment = dict(id=1, start=0.0, end=19.0, text_zh="测试", en="Short line.")
        following = dict(id=2, start=20.0, end=21.0, text_zh="测试", en="Later.")
        cues, qc = refine([segment, following], {"characters": {}, "terms": {}}, 0, 0)
        self.assertLessEqual(cues[0]["end"] - cues[0]["start"], 6.0)
        self.assertGreaterEqual(cues[0]["end"] - cues[0]["start"], 1.0)
        self.assertEqual(qc["segment_fallback_capped"], 1)
        self.assertEqual(qc["timing_alignment"], "segment")
        self.assertEqual(cues[1]["start"], 20.0)

    def test_normal_long_speech_not_trimmed(self):
        words = [dict(start=i, end=i + 1, word="w") for i in range(0, 5)]
        segment = dict(id=1, start=0.0, end=5.0, en="One two three four five", words=words)
        cues, qc = refine([segment], {"characters": {}, "terms": {}}, 0, 0)
        self.assertEqual(qc["timing_anomalies_detected"], 0)
        self.assertAlmostEqual(cues[0]["end"] - cues[0]["start"], 5.0)


if __name__ == "__main__":
    unittest.main()
