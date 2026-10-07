"""Validate an SRT: python scripts/validate_srt.py file.srt."""
import argparse
from pathlib import Path
import re
from subtitle_presentation import metrics


def seconds(value):
    h, m, s, ms = map(int, re.split(r"[:,]", value))
    return h * 3600 + m * 60 + s + ms / 1000


def read_srt(path):
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8-sig").strip()):
        lines = block.splitlines()
        if len(lines) < 3:
            raise ValueError("Incomplete SRT block")
        match = re.fullmatch(r"(\d{2,}:\d{2}:\d{2},\d{3}) --> (\d{2,}:\d{2}:\d{2},\d{3})", lines[1])
        if not match:
            raise ValueError("Invalid timestamp syntax")
        cues.append(dict(id=int(lines[0]), start=seconds(match[1]), end=seconds(match[2]), text="\n".join(lines[2:])))
    return cues


def validate(cues):
    errors, warnings = [], []
    for index, cue in enumerate(cues):
        duration = cue["end"] - cue["start"]
        if cue["start"] < 0 or duration <= 0:
            errors.append(f"Cue {cue['id']}: invalid ordering")
        if len(cue["text"].splitlines()) > 2 or any(len(l) > 42 for l in cue["text"].splitlines()):
            errors.append(f"Cue {cue['id']}: line limits")
        if duration > 6.001:
            errors.append(f"Cue {cue['id']}: duration >6s")
        if duration < .5:
            warnings.append(f"Cue {cue['id']}: duration <0.5s")
        if index and cues[index - 1]["end"] > cue["start"] + .0001:
            errors.append(f"Cue {cue['id']}: overlap")
        if index and cues[index - 1]["text"] == cue["text"]:
            warnings.append(f"Cue {cue['id']}: duplicate adjacent text")
    report = metrics(cues)
    if report["high_cps_over_20"]:
        warnings.append(f"{report['high_cps_over_20']} cues exceed 20 CPS; {report['severe_cps_over_25']} exceed 25 CPS")
    return errors, warnings, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        errors, warnings, report = validate(read_srt(args.path))
        print(report)
        for message in errors + warnings:
            print(message)
        raise SystemExit(1 if errors else 0)
    except ValueError as error:
        print(error)
        raise SystemExit(1)
