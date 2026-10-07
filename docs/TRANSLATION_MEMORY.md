# Translation Memory and Terminology

AsterSub keeps a persistent, per-series SQLite database at:

```
<work-dir>/translation_memory.sqlite3
```

There is one database per series (work directory), so different shows never
contaminate each other's terminology or translations.

## Why it exists

Repeated phrases and character names are common across episodes. Exact reuse
avoids repeated LLM calls and keeps terminology consistent across a series.

## Exact translation memory

For every target segment, AsterSub computes a signature from:

- normalized Chinese source text
- translation style
- prompt version
- applicable authoritative terminology (only the terms present in that source)

If a stored entry matches, it is reused and the LLM is not called. Matching is
**exact** on the normalized source; there is no fuzzy sentence substitution, so
`杀了他` and `别杀他` are never treated as the same line.

Changing a translation style, or changing a glossary term that appears in the
source, invalidates the affected entries only. Unrelated glossary changes do not
invalidate unrelated translations.

## Terminology authority

```
manual glossary (glossary.json)
 > locked database term
 > confirmed learned term
 > episode-local hint
 > candidate hint
 > LLM choice
```

- **Manual glossary** entries in `glossary.json` always win.
- **Locked** terms can never be overwritten by automatic learning.
- **Confirmed** terms are reused as hard constraints.
- **Episode-local hints** are populated during an episode so later batches keep a
  consistent rendering, but are not globally authoritative.
- **Candidates** are hints only and never hard constraints.

Term matching uses longest-match-first so that overlapping terms do not apply
inside a longer term (for example, `ABC` wins over `AB` and `A`).

## Learning and conflicts

Observations are recorded with a stable occurrence key
(`episode`, `cue`, `source`), so reprocessing an episode is idempotent and never
double-counts. A term is promoted from candidate to confirmed only after several
trustworthy, consistent observations. If a term has multiple renderings, the
conflict is recorded and the established value is kept.

## Managing terminology

```powershell
python scripts\terminology.py stats     --work "<work-dir>"
python scripts\terminology.py list      --work "<work-dir>"
python scripts\terminology.py search "红莲" --work "<work-dir>"
python scripts\terminology.py candidates --work "<work-dir>"
python scripts\terminology.py conflicts  --work "<work-dir>"
python scripts\terminology.py set "玄冥" "Xuanming" --work "<work-dir>"
python scripts\terminology.py lock "玄冥" --work "<work-dir>"
python scripts\terminology.py unlock "玄冥" --work "<work-dir>"
python scripts\terminology.py export    --work "<work-dir>"
```

`export` writes a human-readable `<work-dir>/terminology.json`. The SQLite
database remains the source of truth.
