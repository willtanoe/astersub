"""Explicit one-episode cached localization A/B harness.

Compares the cached translations against a fresh run in the configured
translation style. Requires TRANSLATOR_* environment variables and never
modifies the cache. This is a manual harness, not a unit test.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from high_quality import HighQuality, ROOT, save_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="Cached <episode>.en.json state file")
    parser.add_argument("--out", type=Path, default=Path("localization_ab.json"))
    parser.add_argument("--translation-style", default="subtitle",
                        choices=("literal", "natural", "subtitle"))
    args = parser.parse_args()
    state = json.loads(args.cache.read_text(encoding="utf-8"))
    segments = state["segments"]
    translator = HighQuality(SimpleNamespace(glossary=ROOT / "glossary.json",
        translator_model=os.getenv("TRANSLATOR_MODEL", ""),
        translation_style=args.translation_style), None)
    translator.episode = args.cache.stem
    result = {}
    for offset in range(0, len(segments), 10):
        targets = segments[offset:offset + 10]
        previous = [dict(id=s["id"], zh=s["text_zh"], en=result[str(s["id"])])
                    for s in segments[max(0, offset - 5):offset]]
        for attempt in range(4):
            try:
                values = translator.request(targets, previous, segments[offset+10:offset+13],
                    f"Translation batch {offset//10+1}/{(len(segments)+9)//10}", 4096, attempt, attempt > 0)
                result.update({str(v["id"]): v["en"] for v in values})
                break
            except Exception:
                if attempt == 3:
                    raise
    comparisons = [dict(id=s["id"], zh=s["text_zh"], old=state["translations"][str(s["id"])],
                        new=result[str(s["id"])]) for s in segments]
    save_json(args.out, comparisons)
    for item in comparisons[:15]:
        print(json.dumps(item, ensure_ascii=False))
