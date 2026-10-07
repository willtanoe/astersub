# AsterSub

AsterSub is a high-quality AI subtitle pipeline for Mandarin video. It combines
faster-whisper ASR, contextual LLM translation, persistent translation memory,
terminology consistency, subtitle timing refinement, readability QC and SRT
generation.

## Features

- faster-whisper Mandarin ASR with CUDA / CTranslate2 acceleration
- ASR caching and resumable, idempotent batch processing
- Contextual LLM translation through an OpenAI-compatible endpoint
- Translation styles: `literal`, `natural`, `subtitle` (default)
- Subtitle-oriented natural localization (not word-for-word)
- Persistent per-series SQLite translation memory
- Exact translation-memory reuse; mixed HIT/MISS batching avoids redundant LLM calls
- Terminology candidates, confirmation, locking and manual glossary authority
- Subtitle presentation refinement with word-timestamp alignment
- Speech-island detection and abnormal-silence handling
- Zero-overlap timing normalization and balanced two-line wrapping
- CPS / readability quality control
- Optional severe-CPS condensation
- JSON-safe QC persistence
- Standalone SRT validation
- Fast mode (direct Whisper translation) retained alongside high mode

## Pipeline

```
Video
 -> ffmpeg audio decode
 -> faster-whisper Mandarin transcription
 -> ASR cache
 -> translation-memory lookup
 -> contextual LLM translation for misses
 -> terminology validation
 -> subtitle presentation
 -> optional severe-CPS condensation
 -> QC
 -> SRT
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Requirements

- Python 3.11 or 3.12 (3.12 tested)
- ffmpeg on `PATH`
- numpy, faster-whisper, CTranslate2, tqdm (see `requirements.txt`)
- Optional NVIDIA GPU with a working driver for CUDA acceleration
- An OpenAI-compatible `chat/completions` endpoint for high-quality translation

No NVIDIA DLLs, CUDA libraries or models are bundled.

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Verify GPU support (expect a value of 1 or more):

```powershell
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
```

If CTranslate2 cannot find the CUDA runtime on Windows, add your CUDA/cuDNN bin
directory to `PATH` for the session:

```powershell
. .\cuda-env.ps1 -CudaBin "C:\Program Files\NVIDIA\CUDNN\bin"
```

## Configuration

High-quality translation uses environment variables. Never commit real keys.

```powershell
$env:TRANSLATOR_BASE_URL = "https://example.com/v1"
$env:TRANSLATOR_API_KEY  = "YOUR_API_KEY"
$env:TRANSLATOR_MODEL    = "provider/model-name"
```

See `.env.example` for the placeholder template.

## Usage

### High-quality mode

```powershell
python scripts\transcribe.py `
  --input "D:\Videos\MySeries" `
  --output "D:\Subtitles\MySeries" `
  --quality high `
  --translation-style subtitle `
  --model medium `
  --device cuda `
  --compute-type float16 `
  --batch-size 10 `
  --beam-size 5 `
  --srt-only
```

Process a single episode for a quick test:

```powershell
python scripts\transcribe.py --input "D:\Videos\MySeries" --output "D:\Subtitles\MySeries" `
  --quality high --start-episode 1 --limit 1 --srt-only
```

Enable selective condensation for severe CPS cues:

```powershell
python scripts\transcribe.py --input "D:\Videos\MySeries" --output "D:\Subtitles\MySeries" `
  --quality high --condense-severe-cps --srt-only
```

### Fast mode

Fast mode uses Whisper's direct translation task and requires no LLM endpoint:

```powershell
python scripts\transcribe.py --input "D:\Videos\MySeries" --output "D:\Subtitles\MySeries" `
  --quality fast --model large-v3 --device cuda --compute-type float16
```

Existing `.en.srt` files are skipped unless `--overwrite` is passed.

## Translation Memory

Each series stores its own SQLite database at
`<work-dir>/translation_memory.sqlite3`. Exact normalized source matches are
reused across runs and across episodes, so repeated lines avoid the LLM. Matching
is exact — there is no fuzzy sentence substitution. See
[docs/TRANSLATION_MEMORY.md](docs/TRANSLATION_MEMORY.md).

## Terminology

Terminology authority order:

```
manual glossary > locked term > confirmed learned term > episode-local hint > candidate > LLM choice
```

Manage the store with the terminology CLI:

```powershell
python scripts\terminology.py stats     --work "<work-dir>"
python scripts\terminology.py list      --work "<work-dir>"
python scripts\terminology.py search "红莲" --work "<work-dir>"
python scripts\terminology.py candidates --work "<work-dir>"
python scripts\terminology.py conflicts  --work "<work-dir>"
python scripts\terminology.py set "玄冥" "Xuanming" --work "<work-dir>"
python scripts\terminology.py lock "玄冥" --work "<work-dir>"
python scripts\terminology.py export    --work "<work-dir>"
```

Automatic candidate discovery is conservative and intended as a review aid, not
a substitute for a curated glossary. Manual glossary entries in `glossary.json`
always take precedence.

## Subtitle Timing

Timing is deterministic and local. Word timestamps are authoritative; speech
islands are detected with a configurable silence gap, and large internal silences
no longer stretch a cue across the gap. Adjacent cues never overlap, and the
timeline is never globally shifted. Text wraps to at most two lines.

Relevant CLI options: `--subtitle-start-pad`, `--subtitle-end-pad`,
`--speech-island-gap`.

## Validation

Validate any generated SRT:

```powershell
python scripts\validate_srt.py "D:\Subtitles\MySeries\Episode 001.en.srt"
```

Hard structural errors (overlap, invalid ordering, more than two lines, lines over
42 characters, cue duration over 6s) cause a non-zero exit. Readability warnings
(high CPS, duplicate adjacent text) do not.

## Testing

```powershell
python -m unittest discover -s scripts -p "test_*.py"
```

## Project Status

ASR, translation memory, terminology, presentation/timing and validation are
implemented and tested. Severe-CPS condensation works but may have higher latency
than necessary because its batching/optimisation is still in progress.
