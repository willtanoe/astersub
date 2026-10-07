import tempfile
import unittest
from pathlib import Path
from translation_memory import Memory


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.memory = Memory(Path(self.temp.name)/'a')

    def tearDown(self):
        self.memory.close()
        self.temp.cleanup()

    def test_longest_match(self):
        for source in ('甲','甲乙','甲乙丙'): self.memory.set(source,source)
        self.assertEqual(list(self.memory.resolve('甲乙丙')),['甲乙丙'])

    def test_manual_priority(self):
        self.memory.set('玄冥','Other')
        self.memory.glossary={'characters':{'玄冥':'Xuanming'}}
        self.assertEqual(self.memory.resolve('玄冥')['玄冥'][0],'Xuanming')

    def test_restart(self):
        self.memory.set('玄冥','Xuanming')
        self.memory.accept('玄冥来了','Xuanming is here','subtitle','v1')
        self.memory.close()
        self.memory=Memory(Path(self.temp.name)/'a')
        self.assertEqual(self.memory.lookup('玄冥来了','subtitle','v1'),'Xuanming is here')

    def test_invalidation(self):
        self.memory.accept('你好','Hello','subtitle','v1')
        self.assertIsNone(self.memory.lookup('你好','natural','v1'))
        self.memory.glossary={'terms':{'无关':'Unrelated'}}
        self.assertEqual(self.memory.lookup('你好','subtitle','v1'),'Hello')

    def test_candidate_idempotency_promotion_conflict(self):
        for i in range(3):
            self.memory.observe('玄冥','Xuanming','episode',i)
            self.memory.observe('玄冥','Xuanming','episode',i)
        self.assertEqual(self.memory.db.execute('SELECT COUNT(*) FROM term_occurrences').fetchone()[0],3)
        self.assertEqual(self.memory.resolve('玄冥')['玄冥'][1],'confirmed')
        self.memory.observe('玄冥','Wrong','episode',4)
        self.assertEqual(self.memory.resolve('玄冥')['玄冥'][0],'Xuanming')

    def test_series_isolation(self):
        self.memory.set('玄冥','Xuanming')
        other=Memory(Path(self.temp.name)/'b')
        self.assertEqual(other.resolve('玄冥'),{})
        other.close()

    def test_rejected_not_saved(self):
        self.memory.set('玄冥','Xuanming')
        with self.assertRaises(ValueError): self.memory.accept('玄冥来了','Wrong','subtitle','v1')
        self.assertIsNone(self.memory.lookup('玄冥来了','subtitle','v1'))
