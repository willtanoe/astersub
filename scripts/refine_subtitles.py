"""Reformat cached translations without ASR, network requests or model loading."""
import argparse
import json
from pathlib import Path
from subtitle_presentation import refine, metrics
from validate_srt import validate
from high_quality import format_english
from json_utils import to_json_safe

ROOT = Path(__file__).resolve().parents[1]


def timestamp(value):
    ms = round(value * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "subtitles_refined")
    parser.add_argument("--glossary", type=Path, default=ROOT / "glossary.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--subtitle-start-pad", type=float, default=-.05)
    parser.add_argument("--subtitle-end-pad", type=float, default=.10)
    args = parser.parse_args()
    glossary = json.loads(args.glossary.read_text(encoding="utf-8"))
    completed = 0
    for path in sorted(args.work.rglob("*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        segments = state.get("segments", [])
        translations = state.get("translations", {})
        if not segments or any(str(s["id"]) not in translations for s in segments):
            continue
        final = [dict(s, en=translations[str(s["id"])]) for s in segments]
        before = [dict(start=s["start"], end=s["end"], text=format_english(s["en"], glossary)) for s in final]
        cues, qc = refine(final, glossary, args.subtitle_start_pad, args.subtitle_end_pad)
        errors, warnings, report = validate(cues)
        relative = path.relative_to(args.work)
        stem = path.stem.removesuffix(".en")
        destination = args.output / relative.parent / (stem + ".en.srt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(f"{i+1}\n{timestamp(c['start'])} --> {timestamp(c['end'])}\n{c['text']}\n" for i,c in enumerate(cues)), encoding="utf-8")
        qc.update(before=metrics(before), validator_errors=errors, validator_warnings=warnings)
        qc["timing_drift_check"] = all(
            c["start"] >= next(s for s in final if s["id"] == c["source_id"])["start"] - .051
            and c["end"] <= next(s for s in final if s["id"] == c["source_id"])["end"] + .101
            for c in cues)
        destination.with_name(destination.name.replace(".en.srt", ".qc.json")).write_text(json.dumps(to_json_safe(qc), indent=2), encoding="utf-8")
        print(f"{path.stem}: BEFORE {metrics(before)} AFTER {report} | split={qc['long_cues_split']} | validator errors={errors}")
        examples = [s for s in final if len(s["en"]) > 84 or s["end"] - s["start"] > 6][:1] or final[:1]
        for source in examples:
            print('BEFORE:', source)
            print('AFTER:', [c for c in cues if c['source_id'] == source['id']])
        completed += 1
        if args.limit and completed >= args.limit:
            break
    print(f"Refined {completed} cached episodes; zero ASR/LLM calls")


if __name__ == "__main__":
    main()
