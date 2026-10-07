import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from subtitle_quality import condense, risk
from subtitle_presentation import refine, visible
from translation_memory import Memory


class FakeTranslator:
    def __init__(self, memory, reply, verdict="ACCEPT"):
        self.memory = memory
        self.glossary = {"characters": {}, "terms": {}}
        self.args = SimpleNamespace(translation_style="subtitle", subtitle_start_pad=-.05, subtitle_end_pad=.10)
        self.prompt = "base-prompt-v1"
        self.reply = reply
        self.verdict = verdict
        self.generation = 0
        self.verification = 0

    def request_raw(self, target, previous, following, label, budget):
        if "Verify the replacement" in self.prompt:
            self.verification += 1
            return self.verdict
        self.generation += 1
        return json.dumps(self.reply)


def make_cue(text, duration=1.0):
    segment = dict(id=1, start=0.0, end=duration, text_zh="测试字幕来源", en=text)
    cues, qc = refine([segment], {"characters": {}, "terms": {}}, 0, 0)
    return segment, cues, qc


class CondensationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.memory = Memory(self.temp.name)

    def tearDown(self):
        self.memory.close()
        self.temp.cleanup()

    def test_cache_miss_then_hit_avoids_generation(self):
        old = "I'm determined to get the Karmic Fire Red Lotus too."
        segment, cues, qc = make_cue(old, 1.2)
        t = FakeTranslator(self.memory, {"translation": "I'll claim the Karmic Fire Red Lotus too.",
                                         "preservation": {"meaning": True}})
        _, qc = condense(t, [segment], cues, qc)
        self.assertEqual(qc["condensation_accepted"], 1)
        self.assertEqual(qc["condensation_verification_calls"], 0)
        self.assertEqual(qc["condensation_verification_calls_avoided"], 1)
        self.assertEqual(qc["condensation_cache_hits"], 0)
        t2 = FakeTranslator(self.memory, {"translation": "SHOULD NOT BE USED"})
        _, qc2 = condense(t2, [segment], cues, qc)
        self.assertEqual(qc2["condensation_cache_hits"], 1)
        self.assertEqual(t2.generation, 0)

    def test_lore_risk_triggers_verifier(self):
        old = "When Houtu becomes the cycle of reincarnation."
        segment, cues, qc = make_cue(old, 1.2)
        t = FakeTranslator(self.memory, {"translation": "When Houtu becomes reincarnation."}, verdict="ACCEPT")
        _, qc = condense(t, [segment], cues, qc)
        self.assertEqual(qc["condensation_verification_calls"], 1)
        self.assertEqual(qc["condensation_accepted"], 1)

    def test_lore_rejection_keeps_original(self):
        old = "When Houtu becomes the cycle of reincarnation."
        segment, cues, qc = make_cue(old, 1.2)
        t = FakeTranslator(self.memory, {"translation": "When Houtu becomes reincarnation."}, verdict="REJECT")
        cues_after, qc = condense(t, [segment], cues, qc)
        self.assertEqual(qc["condensation_rejected_semantic_risk"], 1)
        self.assertEqual(qc["condensation_accepted"], 0)
        self.assertIn("cycle of reincarnation", cues_after[0]["text"])

    def test_rejected_not_cached(self):
        old = "When Houtu becomes the cycle of reincarnation."
        segment, cues, qc = make_cue(old, 1.2)
        t = FakeTranslator(self.memory, {"translation": "When Houtu becomes reincarnation."}, verdict="REJECT")
        condense(t, [segment], cues, qc)
        self.assertEqual(self.memory.db.execute("SELECT COUNT(*) FROM condensation_cache").fetchone()[0], 0)

    def test_terminology_change_rejected(self):
        self.memory.set("甲", "Alpha")
        old = "Alpha is guarding the gate, and this is a long subtitle line."
        segment = dict(id=1, start=0.0, end=1.2, text_zh="甲在看守大门", en=old)
        cues, qc = refine([segment], {"characters": {}, "terms": {}}, 0, 0)
        t = FakeTranslator(self.memory, {"translation": "He guards the gate and this is a long subtitle."})
        _, qc = condense(t, [segment], cues, qc)
        self.assertEqual(qc["condensation_rejected_terminology"], 1)
        self.assertEqual(qc["condensation_accepted"], 0)

    def test_risk_helper(self):
        self.assertIn("lore", risk("轮回", "the cycle of reincarnation", "reincarnation"))
        self.assertEqual(risk("好", "I'll go now.", "I'll go."), [])


if __name__ == "__main__":
    unittest.main()
