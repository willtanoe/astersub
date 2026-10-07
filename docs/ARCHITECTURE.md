# AsterSub Architecture

AsterSub turns Mandarin video into English SRT subtitles. It is organised as a
small set of focused modules under `scripts/`.

## Pipeline

```
Video
 -> ffmpeg audio decode (16 kHz mono float32)
 -> faster-whisper Mandarin ASR (task=transcribe, word_timestamps=True)
 -> ASR cache (work/<series>/<episode>.en.json)
 -> translation-memory lookup (per-series SQLite)
 -> contextual LLM translation for cache misses only
 -> terminology validation
 -> subtitle presentation (speech islands, timing, wrapping)
 -> optional severe-CPS condensation
 -> QC (<episode>.qc.json)
 -> SRT (<episode>.en.srt)
```

## Modules

| Module | Responsibility |
| --- | --- |
| `transcribe.py` | CLI entry point. Fast mode (direct Whisper translate) and high mode orchestration. |
| `high_quality.py` | ASR, contextual translation via an OpenAI-compatible endpoint, validation, resume. |
| `subtitle_presentation.py` | Deterministic timing refinement, speech islands, line wrapping, QC metrics. |
| `subtitle_quality.py` | Optional severe-CPS condensation with cache and conditional verification. |
| `translation_memory.py` | Per-series SQLite translation memory and terminology authority. |
| `terminology.py` | CLI for inspecting and managing the terminology store. |
| `json_utils.py` | Recursive numpy-safe JSON conversion and atomic JSON writes. |
| `validate_srt.py` | Standalone SRT structural validator. |
| `refine_subtitles.py` | Re-run presentation on cached translations with zero ASR/LLM calls. |

## ASR cache

High mode stores Chinese ASR segments, word timestamps and translations in
`work/<series>/<episode>.en.json`. The state is keyed by source file identity
(path, size, mtime, model, beam size). Reruns reuse the cached ASR and only
request translations that are not already available.

## Translation path

1. For each target segment, look up the exact translation memory entry.
2. Cache hits bypass the LLM entirely.
3. Remaining IDs are sent in a single contextual request with read-only previous
   and following context.
4. Responses are validated for exact IDs, ordering and terminology.
5. Only accepted results are persisted to translation memory and terminology.

## Translation memory

See [TRANSLATION_MEMORY.md](TRANSLATION_MEMORY.md). Each series has its own
`translation_memory.sqlite3`; unrelated series never share terminology.

## Terminology resolution

Authority order:

```
manual glossary
 > locked database term
 > confirmed learned term
 > episode-local hint
 > candidate hint
 > LLM choice
```

Longest-match-first lookup prevents overlapping shorter terms from being applied
inside a longer term. Only manual, locked and confirmed mappings are hard
constraints.

## Presentation and timing

`subtitle_presentation.refine()` is fully deterministic and local:

- Word timestamps are authoritative when present.
- Speech islands are detected using a configurable silence gap
  (`--speech-island-gap`, default `0.8s`).
- Large intermittent silences no longer stretch a cue across the gap.
- Leading/trailing silence is trimmed and legacy segment-only caches receive a
  readability-based duration cap.
- Adjacent cues never overlap; overlaps are resolved by shortening the previous
  cue, never by shifting the whole timeline.
- Text is wrapped to at most two lines, targeting 38 characters per line.

## Quality control

Each episode emits `<episode>.qc.json` with timing, CPS, terminology,
translation-memory and condensation counters. `validate_srt.py` provides an
independent structural check.
