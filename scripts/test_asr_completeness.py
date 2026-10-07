import tempfile
import unittest
from pathlib import Path

from asr_completeness import (audit, merge_recovered, suspect_reasons,
                              uncovered_windows, validate_recovery)
from subtitle_presentation import refine
from validate_srt import validate
from translation_memory import Memory


def seg(id, start, end, text, words=None):
    return dict(id=id, start=start, end=end, text_zh=text, words=words or [])


class CompletenessTests(unittest.TestCase):
    def test_speech_gap_without_asr_is_recovery_candidate(self):
        segments = [seg(1, 10.0, 12.1, "第一"), seg(2, 15.4, 18.0, "第二")]
        windows = uncovered_windows([(10.0, 12.1), (13.0, 13.8), (15.4, 18.0)], segments)
        self.assertEqual(len(windows), 1)
        self.assertLess(windows[0][0], 13.0)
        self.assertGreater(windows[0][1], 13.8)

    def test_pure_silence_creates_no_candidate(self):
        self.assertEqual(uncovered_windows([], [seg(1, 0, 5, "x")]), [])

    def test_loud_non_speech_recovery_is_rejected(self):
        ok, reason = validate_recovery("哈哈哈哈哈哈", [])
        self.assertFalse(ok)
        self.assertIn(reason, {"repetition", "empty_or_no_chinese"})
        ok, reason = validate_recovery("[Music]", [])
        self.assertFalse(ok)

    def test_valid_mandarin_recovered_inserted(self):
        segments = [seg(1, 0.0, 1.0, "你好")]
        report_holder = {}

        def stub(start, end):
            return dict(start=3.0, end=3.8, text_zh="再见",
                        words=[{"start": 3.0, "end": 3.8, "word": "再见"}], recovered=True)

        merged, report = audit(segments, None, speech=[(0.0, 1.0), (3.0, 4.0)],
                               transcribe=stub, recovery=True)
        self.assertEqual(report["asr_segments_recovered"], 1)
        self.assertEqual([s["text_zh"] for s in merged], ["你好", "再见"])
        self.assertEqual([s["id"] for s in merged], [1, 2])

    def test_duplicate_recovery_rejected(self):
        segments = [seg(1, 0.0, 1.0, "你好世界")]
        merged, report = audit(segments, None, speech=[(0.0, 1.0), (3.0, 4.0)],
                               transcribe=lambda a, b: dict(start=3.0, end=3.8, text_zh="你好世界",
                                                            words=[]), recovery=True)
        self.assertEqual(report["asr_segments_recovered"], 0)
        self.assertEqual(report["asr_recovery_rejected"], 1)
        self.assertEqual(len(merged), 1)

    def test_repetitive_recovery_rejected(self):
        segments = [seg(1, 0.0, 1.0, "你好")]
        merged, report = audit(segments, None, speech=[(0.0, 1.0), (3.0, 4.0)],
                               transcribe=lambda a, b: dict(start=3.0, end=3.8, text_zh="好好好好好",
                                                            words=[]), recovery=True)
        self.assertEqual(report["asr_segments_recovered"], 0)

    def test_cached_asr_reused_when_recovery_disabled(self):
        segments = [seg(1, 0.0, 1.0, "你好")]
        merged, report = audit(segments, None, speech=[(0.0, 1.0), (3.0, 4.0)],
                               transcribe=None, recovery=False)
        self.assertEqual(merged, segments)
        self.assertEqual(report["asr_recovery_windows"], 0)
        self.assertFalse(report["recovery_enabled"])

    def test_old_cache_without_metadata_loads(self):
        segment = seg(1, 0.0, 1.0, "你好")
        reasons, largest, islands = suspect_reasons(segment)
        self.assertEqual(reasons, [])
        _, report = audit([segment], None, speech=[], transcribe=None, recovery=False)
        self.assertIn("asr_suspect", segment)
        self.assertIn("suspect_reasons", segment)

    def test_large_internal_word_gap_flags_suspect(self):
        words = [dict(start=81.5, end=82.56, word="刚"),
                 dict(start=100.35, end=100.53, word="才")]
        reasons, largest, islands = suspect_reasons(seg(35, 81.5, 100.53, "我问你刚才", words))
        self.assertIn("large_internal_word_gap", reasons)
        self.assertIn("multiple_speech_islands", reasons)
        self.assertEqual(islands, 2)
        self.assertGreater(largest, 17)

    def test_recovered_ordering(self):
        base = [seg(1, 10.0, 11.0, "后")]
        merged = merge_recovered(base, [dict(start=3.0, end=3.8, text_zh="前", words=[], recovered=True)])
        self.assertEqual([s["text_zh"] for s in merged], ["前", "后"])
        self.assertEqual([s["id"] for s in merged], [2, 1])

    def test_recovered_goes_through_translation_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = Memory(Path(tmp))
            recovered = dict(id=2, start=3.0, end=3.8, text_zh="再见", words=[], recovered=True)
            calls = []

            def client(targets):
                calls.append([t["id"] for t in targets])
                return [dict(id=t["id"], en="Goodbye") for t in targets]

            result = memory.batch([recovered], "subtitle", "v", client)
            self.assertEqual(calls, [[2]])
            self.assertEqual(result[0]["en"], "Goodbye")
            memory.close()

    def test_recovered_presentation_has_no_overlap(self):
        segments = [seg(1, 0.0, 1.0, "你好", [dict(start=0.0, end=1.0, word="你好")]),
                    dict(id=2, start=3.0, end=3.8, text_zh="再见",
                         words=[dict(start=3.0, end=3.8, word="再见")], recovered=True)]
        final = [dict(s, en="Hello" if s["id"] == 1 else "Goodbye") for s in segments]
        cues, qc = refine(final, {"characters": {}, "terms": {}}, 0, 0)
        self.assertEqual(validate(cues)[0], [])

    def test_recovered_overlap_is_clipped_without_shifting_neighbors(self):
        # Regression for the 0.71s -> 0.16s cue regression.
        original = seg(1, 50.76, 51.37, "还有一件事",
                       [dict(start=50.76, end=51.37, word="还有一件事")])
        later = seg(2, 53.0, 54.0, "后来",
                    [dict(start=53.0, end=54.0, word="后来")])
        recovered = dict(start=50.97, end=52.43, text_zh="还有谁",
                         words=[dict(start=50.92, end=51.5, word="还有谁")], recovered=True)
        merged = merge_recovered([original, later], [recovered])
        first = next(s for s in merged if s["id"] == 1)
        rec = next(s for s in merged if s.get("recovered"))
        self.assertEqual(first["start"], 50.76)
        self.assertEqual(first["end"], 51.37)
        self.assertGreaterEqual(rec["start"], first["end"])
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(merged, merged[1:])))


if __name__ == "__main__":
    unittest.main()
