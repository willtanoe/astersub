"""Batch translation of Chinese video audio directly to English SRT."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
import time

from faster_whisper import WhisperModel
from tqdm import tqdm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def timestamp(seconds):
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def resource_error(error):
    message = str(error).lower()
    return any(term in message for term in (
        "out of memory", "cuda_error_out_of_memory", "cudaerror_memoryallocation",
        "cublas_status_alloc_failed", "failed to allocate", "not enough memory",
    ))


class Translator:
    def __init__(self, model, device, compute_type):
        self.configs = [(model, compute_type)]
        if device == "cuda" and model == "large-v3" and compute_type == "float16":
            self.configs.extend([("large-v3", "int8_float16"), ("medium", "float16")])
        elif device == "cuda" and model == "large-v3" and compute_type == "int8_float16":
            self.configs.append(("medium", "float16"))
        self.device = device
        self.index = 0
        self.model = None

    def write(self, source, destination):
        temporary = destination.with_name(destination.name + ".tmp")
        decoded = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
             "-vn", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
            capture_output=True, check=True,
        )
        audio = np.frombuffer(decoded.stdout, dtype=np.float32).copy()
        while True:
            try:
                model_name, compute_type = self.configs[self.index]
                if self.model is None:
                    tqdm.write(f"Loading {model_name}, device={self.device}, compute_type={compute_type}")
                    self.model = WhisperModel(model_name, device=self.device, compute_type=compute_type)
                segments, info = self.model.transcribe(
                    audio, language="zh", task="translate", vad_filter=True,
                )
                count = 0
                destination.parent.mkdir(parents=True, exist_ok=True)
                with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                    for segment in segments:
                        text = segment.text.strip()
                        if not text:
                            continue
                        count += 1
                        handle.write(f"{count}\n{timestamp(segment.start)} --> {timestamp(segment.end)}\n{text}\n\n")
                os.replace(temporary, destination)
                return count, info
            except Exception as error:
                temporary.unlink(missing_ok=True)
                if not resource_error(error) or self.index + 1 >= len(self.configs):
                    raise
                tqdm.write(f"VRAM/resource failure: {error}. Trying next configuration.")
                self.model = None
                self.index += 1


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "input")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--quality", choices=("fast", "high"), default="fast")
    parser.add_argument("--srt-only", action="store_true", help="Export only English SRT; keep resume data in work")
    parser.add_argument("--subtitle-start-pad", type=float, default=-.05)
    parser.add_argument("--subtitle-end-pad", type=float, default=.10)
    parser.add_argument("--speech-island-gap", type=float, default=.8)
    parser.add_argument("--glossary", type=Path, default=ROOT / "glossary.json")
    parser.add_argument("--translator-model")
    parser.add_argument("--translation-style", choices=("literal", "natural", "subtitle"), default="subtitle")
    parser.add_argument("--condense-severe-cps", action="store_true")
    parser.add_argument("--asr-recovery", choices=("off", "auto"), default="auto",
                        help="Targeted retranscription of speech gaps missed by ASR (high mode)")
    parser.add_argument("--asr-recovery-max-windows", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Process only the first N videos (development test)")
    parser.add_argument("--start-episode", type=int, help="Start at this episode number, using the last number in the filename")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.model = args.model or ("medium" if args.quality == "high" else "large-v3")
    args.output = args.output or ROOT / ("subtitles_high" if args.quality == "high" else "subtitles")
    if (args.batch_size < 1 or args.beam_size < 1 or args.asr_recovery_max_windows < 1
            or (args.limit is not None and args.limit < 1)):
        parser.error("batch-size, beam-size, asr-recovery-max-windows and limit must be positive")
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        parser.error(f"Input directory does not exist: {input_root}")
    videos = sorted(path for path in input_root.rglob("*")
                    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    videos = sorted(videos, key=natural_sort_key)
    if args.start_episode is not None:
        videos = [path for path in videos
                  if (numbers := re.findall(r"\d+", path.stem))
                  and int(numbers[-1]) >= args.start_episode]
    if args.limit:
        videos = videos[:args.limit]
    if args.quality == "high":
        from high_quality import HighQuality
        translator = HighQuality(args, timestamp)
    else:
        translator = Translator(args.model, args.device, args.compute_type)
    succeeded = skipped = failed = 0
    batch_start = time.perf_counter()
    for source in tqdm(videos, desc="Videos", unit="video"):
        relative = source.relative_to(input_root)
        destination = output_root / relative.with_suffix(".en.srt")
        if destination.exists() and not args.overwrite:
            skipped += 1
            tqdm.write(f"SKIP {relative}: {destination}")
            continue
        tqdm.write(f"Processing: {relative}")
        start = time.perf_counter()
        try:
            count, info = translator.write(source, destination)
            elapsed = time.perf_counter() - start
            speed = info.duration / elapsed if elapsed > 0 else 0
            tqdm.write(f"DONE {relative} | elapsed={elapsed:.1f}s | language={info.language} "
                       f"(source forced zh) | segments={count} | speed={speed:.2f}x realtime "
                       f"| output={destination}")
            succeeded += 1
        except Exception as error:
            failed += 1
            tqdm.write(f"ERROR {relative} | elapsed={time.perf_counter() - start:.1f}s | {error}")
    tqdm.write(f"Finished: success={succeeded}, skipped={skipped}, failed={failed}, "
               f"elapsed={time.perf_counter() - batch_start:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
