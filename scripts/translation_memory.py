"""Transactional per-series exact memory and terminology authority."""
import hashlib
import json
from pathlib import Path
import sqlite3
import unicodedata
import re
from json_utils import to_json_safe


def normalize(text):
    return unicodedata.normalize('NFKC', text).strip()


class Memory:
    def __init__(self, work, glossary=None):
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.work / 'translation_memory.sqlite3')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.glossary = glossary or {'characters': {}, 'terms': {}}
        self.local = {}
        self.stats = dict.fromkeys(('manual','locked','confirmed','local','candidate','added','promoted','conflicts','hits','misses'),0)
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS terms(source TEXT PRIMARY KEY, en TEXT NOT NULL,
          status TEXT NOT NULL, locked INTEGER NOT NULL DEFAULT 0, source_type TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS term_occurrences(episode TEXT, cue TEXT, source TEXT, en TEXT,
          context TEXT, suspect INTEGER, PRIMARY KEY(episode,cue,source));
        CREATE TABLE IF NOT EXISTS term_candidates(source TEXT, en TEXT, count INTEGER,
          PRIMARY KEY(source,en));
        CREATE TABLE IF NOT EXISTS term_conflicts(source TEXT, established TEXT, alternative TEXT,
          episode TEXT, cue TEXT, UNIQUE(source,established,alternative,episode,cue));
        CREATE TABLE IF NOT EXISTS translation_memory(source TEXT, signature TEXT, en TEXT,
          PRIMARY KEY(source,signature));
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS condensation_cache(signature TEXT PRIMARY KEY, payload TEXT);
        ''')

    def close(self):
        self.db.close()

    def set(self, source, en, locked=True):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO terms VALUES(?,?,?,?,?)',
                (normalize(source), en, 'manual', int(locked), 'manual'))

    def resolve(self, text):
        text = normalize(text)
        choices = {s: (e, 'local') for s,e in self.local.items()}
        for s,e in self.db.execute('SELECT source,en FROM term_candidates WHERE count>=1 ORDER BY count'):
            choices.setdefault(s,(e,'candidate'))
        for s,e,status,locked,_ in self.db.execute('SELECT * FROM terms'):
            if status in ('confirmed','manual') or locked:
                choices[s] = (e, 'locked' if locked else 'confirmed')
        for section in ('characters','terms'):
            for s,e in self.glossary.get(section,{}).items():
                choices[normalize(s)] = (e, 'manual')
        matches = {}
        position = 0
        while position < len(text):
            found = next((s for s in sorted(choices,key=len,reverse=True) if text.startswith(s,position)), None)
            if found:
                matches[found] = choices[found]
                position += len(found)
            else:
                position += 1
        return matches

    def signature(self, source, style, version):
        mappings = sorted((s,e) for s,(e,authority) in self.resolve(source).items() if authority not in ('local','candidate'))
        return hashlib.sha256(json.dumps([style,version,mappings],ensure_ascii=False).encode()).hexdigest()

    def lookup(self, source, style, version):
        row = self.db.execute('SELECT en FROM translation_memory WHERE source=? AND signature=?',
            (normalize(source),self.signature(source,style,version))).fetchone()
        return row[0] if row else None

    def validate(self, source, en):
        for term,(expected,authority) in self.resolve(source).items():
            if authority not in ('local','candidate') and expected not in en:
                raise ValueError(f'Terminology constraint missing: {term} -> {expected}')

    def accept(self, source, en, style, version):
        self.validate(source,en)
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO translation_memory VALUES(?,?,?)',
                (normalize(source),self.signature(source,style,version),en))

    def observe(self, source, en, episode, cue, context='', suspect=False):
        """Explicitly aligned candidate observations only; never guess bilingual alignment."""
        source = normalize(source)
        with self.db:
            inserted = self.db.execute('INSERT OR IGNORE INTO term_occurrences VALUES(?,?,?,?,?,?)',
                (str(episode),str(cue),source,en,context,int(suspect))).rowcount
            if not inserted:
                return
            self.stats['added'] += 1
            if suspect:
                self.stats['suspect'] = self.stats.get('suspect', 0) + 1
            established = self.db.execute('SELECT en FROM terms WHERE source=?',(source,)).fetchone()
            if established and established[0] != en:
                self.stats['conflicts'] += 1
                self.db.execute('INSERT OR IGNORE INTO term_conflicts VALUES(?,?,?,?,?)',
                    (source,established[0],en,str(episode),str(cue)))
            self.db.execute('INSERT INTO term_candidates VALUES(?,?,1) ON CONFLICT(source,en) DO UPDATE SET count=count+1',(source,en))
            observations = self.db.execute('SELECT en,count FROM term_candidates WHERE source=?',(source,)).fetchall()
            trustworthy = self.db.execute('SELECT COUNT(*) FROM term_occurrences WHERE source=? AND suspect=0',(source,)).fetchone()[0]
            if len(observations)==1 and trustworthy >= 3 and not established:
                self.db.execute('INSERT INTO terms VALUES(?,?,?,0,?)',(source,en,'confirmed','learned'))
                self.stats['promoted'] += 1
            if not established and not suspect:
                self.local[source] = en

    def discover(self, source, en, episode, cue, suspect=False):
        self.validate(source,en)
        # Only unambiguous one-to-one entity-bearing sentence pairs are aligned.
        zh = set(re.findall(r'[\u4e00-\u9fff]{1,8}(?:族|宗|宫|殿|莲|剑|阵|城|山|境)',source))
        english = set(re.findall(r'\b[A-Z][a-z]+(?:[- ][A-Z][a-z]+){1,7}\b',en))
        if len(zh)==1 and len(english)==1:
            self.observe(next(iter(zh)),next(iter(english)),episode,cue,source,suspect)

    def batch(self, targets, style, version, client):
        result, missing = {}, []
        for target in targets:
            value = self.lookup(target['text_zh'],style,version)
            if value is None:
                missing.append(target)
                self.stats['misses'] += 1
            else:
                result[target['id']] = value
                self.stats['hits'] += 1
        if missing:
            values = client(missing)
            if [v['id'] for v in values] != [v['id'] for v in missing]:
                raise ValueError('Client returned invalid IDs')
            for source,value in zip(missing,values): self.validate(source['text_zh'],value['en'])
            for source,value in zip(missing,values):
                self.accept(source['text_zh'],value['en'],style,version)
                result[source['id']] = value['en']
        return [dict(id=t['id'],en=result[t['id']]) for t in targets]

    def condensation_signature(self, source, original_en, style, version):
        mappings = sorted((s,e) for s,(e,authority) in self.resolve(source).items()
                          if authority not in ('local','candidate'))
        return hashlib.sha256(json.dumps(['condense-v1',style,version,normalize(source),
            original_en,mappings],ensure_ascii=False).encode()).hexdigest()

    def condensation_lookup(self, source, original_en, style, version):
        row = self.db.execute('SELECT payload FROM condensation_cache WHERE signature=?',
            (self.condensation_signature(source,original_en,style,version),)).fetchone()
        return json.loads(row[0]) if row else None

    def condensation_store(self, source, original_en, style, version, translation):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO condensation_cache VALUES(?,?)',
                (self.condensation_signature(source,original_en,style,version),
                 json.dumps(to_json_safe(translation),ensure_ascii=False)))

    def summarize(self, segments):
        counts = dict(manual=0, locked=0, confirmed=0, local=0, candidate=0)
        for source in segments:
            for term, (en, authority) in self.resolve(source["text_zh"]).items():
                counts[authority] = counts.get(authority, 0) + 1
        return counts

    def export(self):
        data = {table:self.db.execute('SELECT * FROM '+table).fetchall()
                for table in ('terms','term_candidates','term_conflicts')}
        path = self.work / 'terminology.json'
        path.write_text(json.dumps(to_json_safe(data),ensure_ascii=False,indent=2),encoding='utf-8')
        return path
