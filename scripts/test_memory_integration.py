import tempfile
import unittest
from translation_memory import Memory


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.m=Memory(self.temp.name)

    def tearDown(self):
        self.m.close()
        self.temp.cleanup()

    def test_mixed_batch(self):
        targets=[dict(id=i,text_zh=str(i)) for i in range(1,7)]
        for i in (1,3,6): self.m.accept(str(i),'E'+str(i),'subtitle','v')
        calls=[]
        def client(targets):
            calls.append([t['id'] for t in targets])
            return [dict(id=t['id'],en='E'+str(t['id'])) for t in targets]
        result=self.m.batch(targets,'subtitle','v',client)
        self.assertEqual(calls,[[2,4,5]])
        self.assertEqual([r['id'] for r in result],list(range(1,7)))
        self.m.batch(targets,'subtitle','v',lambda _:self.fail('LLM called on HIT'))

    def test_relevant_change(self):
        self.m.glossary={'terms':{'甲':'Alpha'}}
        self.m.accept('甲来了','Alpha is here','subtitle','v')
        self.m.glossary={'terms':{'甲':'Beta'}}
        self.assertIsNone(self.m.lookup('甲来了','subtitle','v'))

    def test_locked_conflict(self):
        self.m.set('玄冥','Xuanming')
        for i in range(4): self.m.observe('玄冥','Wrong','ep',i)
        self.assertEqual(self.m.resolve('玄冥')['玄冥'][0],'Xuanming')
        self.assertEqual(self.m.db.execute('SELECT COUNT(*) FROM term_conflicts').fetchone()[0],4)

    def test_discovery_local(self):
        self.m.discover('黑莲在手','Black Lotus is mine','ep',1)
        self.assertEqual(self.m.local.get('黑莲'),'Black Lotus')

    def test_generic_excluded(self):
        self.m.discover('你好现在走','Hello now go','ep',1)
        self.assertEqual(self.m.local,{})

    def test_candidate_not_constraint(self):
        self.m.observe('黑莲','Black Lotus','ep',1)
        self.m.local.clear()
        self.m.validate('黑莲','A different translation')

    def test_suspect_no_promotion(self):
        for i in range(4): self.m.observe('黑莲','Black Lotus','ep',i,suspect=True)
        self.assertEqual(self.m.db.execute('SELECT COUNT(*) FROM terms').fetchone()[0],0)
