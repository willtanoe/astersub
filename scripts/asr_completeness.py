"""ASR completeness audit and targeted speech-gap recovery.

The normal ASR pass can silently drop short, quiet or fast dialogue. This module
detects audio regions that contain speech activity but no recognized segment and
optionally retranscribes only those small local windows.

Nothing here fabricates dialogue: recovery is driven by the actual audio and every
candidate must pass validation before it is merged.
"""
import re

import numpy as np

from subtitle_presentation import speech_islands

SAMPLE_RATE = 16000
AUDIT_VERSION = 2

_CJK = re.compile(r"[\u3400-\u9fff]")


def detect_speech(waveform, sample_rate=SAMPLE_RATE):
    """Return likely speech intervals using the Silero VAD bundled with faster-whisper."""
    from faster_whisper.vad import get_speech_timestamps
    stamps = get_speech_timestamps(waveform, sampling_rate=sample_rate)
    return [(t["start"] / sample_rate, t["end"] / sample_rate) for t in stamps]


def merge_windows(windows, join=.2):
    if not windows:
        return []
    ordered = sorted(windows)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + join:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def uncovered_windows(speech, segments, min_speech=.35, min_uncovered=.35, pad=.4):
    """Speech intervals (padded) that existing ASR segments do not cover."""
    covered = sorted((s["start"], s["end"]) for s in segments)
    windows = []
    for start, end in speech:
        if end - start < min_speech:
            continue
        pieces = [(start, end)]
        for cs, ce in covered:
            remaining = []
            for ps, pe in pieces:
                if ce <= ps or cs >= pe:
                    remaining.append((ps, pe))
                    continue
                if cs > ps:
                    remaining.append((ps, min(cs, pe)))
                if ce < pe:
                    remaining.append((max(ce, ps), pe))
            pieces = remaining
        for ps, pe in pieces:
            if pe - ps >= min_uncovered:
                windows.append((max(0.0, ps - pad), pe + pad))
    return merge_windows(windows)


def segment_islands(segment, gap=2.0):
    words = segment.get("words", [])
    if not words:
        return []
    return speech_islands(words, gap)


def suspect_reasons(segment, gap=2.0, long_duration=6.0):
    """Deterministic flags for segments that look like ASR segmentation anomalies."""
    words = segment.get("words", [])
    reasons, largest, islands = [], 0.0, []
    if words:
        islands = segment_islands(segment, gap)
        gaps = [words[i + 1]["start"] - words[i]["end"] for i in range(len(words) - 1)]
        largest = max(gaps, default=0.0)
        if len(islands) > 1:
            reasons.append("multiple_speech_islands")
        if largest >= gap:
            reasons.append("large_internal_word_gap")
    start = words[0]["start"] if words else segment.get("start", 0.0)
    end = words[-1]["end"] if words else segment.get("end", 0.0)
    if end - start > long_duration:
        reasons.append("long_duration")
    return reasons, largest, len(islands)


def validate_recovery(text, neighbors):
    """Reject empty, noisy, hallucinated or duplicated recovery text."""
    text = (text or "").strip()
    if not text or not _CJK.search(text):
        return False, "empty_or_no_chinese"
    if not re.sub(r"[\s\W_]", "", text, flags=re.UNICODE):
        return False, "punctuation_only"
    if len(_CJK.findall(text)) / max(1, len(text)) < .5:
        return False, "mostly_non_chinese"
    if re.search(r"(.{1,3})\1{2,}", text):
        return False, "repetition"
    norm = re.sub(r"\s", "", text)
    for other in neighbors:
        other = re.sub(r"\s", "", other or "")
        if other and (norm in other or other in norm):
            return False, "duplicate_neighbor"
    return True, "ok"


def transcribe_window(model, waveform, start, end, beam_size=5, sample_rate=SAMPLE_RATE):
    """Transcribe one local audio window; returns a segment dict or None."""
    clip = waveform[int(start * sample_rate):int(end * sample_rate)]
    if clip.size == 0:
        return None
    segments, _ = model.transcribe(
        clip, language="zh", task="transcribe", vad_filter=False,
        beam_size=max(beam_size, 5), word_timestamps=True,
        condition_on_previous_text=False, temperature=0.0)
    words = []
    for segment in segments:
        for word in segment.words or []:
            words.append({"start": start + word.start, "end": start + word.end, "word": word.word})
        if not segment.words and segment.text.strip():
            words.append({"start": start + segment.start, "end": start + segment.end,
                          "word": segment.text})
    if not words:
        return None
    text = "".join(w["word"] for w in words).strip()
    if not text:
        return None
    return {"start": words[0]["start"], "end": words[-1]["end"], "text_zh": text,
            "words": words, "recovered": True, "recovery_pass": True}


def merge_recovered(segments, recovered):
    """Insert recovered segments with fresh ids, keep chronological order, no overlap.

    Recovered windows are padded for transcription context, so their words may
    intrude into neighbouring segments. Recovered words are clipped into the gap
    between the surrounding segments; original segments are never modified.
    """
    next_id = max((s["id"] for s in segments), default=0)
    added = []
    for item in sorted(recovered, key=lambda s: s["start"]):
        next_id += 1
        added.append(dict(item, id=next_id))
    merged = sorted(segments + added, key=lambda s: (s["start"], s["id"]))
    result = []
    for index, segment in enumerate(merged):
        low = result[-1]["end"] if result else 0.0
        high = merged[index + 1]["start"] if index + 1 < len(merged) else float("inf")
        if segment.get("recovered") and segment.get("words"):
            kept = []
            for word in segment["words"]:
                start, end = max(word["start"], low), min(word["end"], high)
                if end > start:
                    kept.append(dict(word, start=start, end=end))
            if not kept:
                continue
            segment = dict(segment, words=kept, text_zh="".join(w["word"] for w in kept),
                           start=kept[0]["start"], end=kept[-1]["end"])
        else:
            segment = dict(segment)
        if segment["end"] > segment["start"]:
            result.append(segment)
    return result


def audit(segments, waveform, speech=None, transcribe=None, beam_size=5, recovery=True,
          max_windows=20, sample_rate=SAMPLE_RATE, gap=2.0):
    """Run completeness audit; optionally recover uncovered speech windows.

    `transcribe(start, end)` must return a segment dict or None. When None and
    recovery is requested, no recovery is performed. Previously recovered
    segments are stripped first so the audit is idempotent across versions.
    """
    segments = [s for s in segments if not s.get("recovered")]
    report = {
        "version": AUDIT_VERSION,
        "recovery_enabled": bool(recovery and transcribe is not None),
        "asr_segments_original": len(segments),
        "asr_suspect_segments": 0,
        "asr_large_internal_word_gaps": 0,
        "asr_suspicious_gaps": 0,
        "asr_recovery_windows": 0,
        "asr_segments_recovered": 0,
        "asr_recovery_rejected": 0,
        "recovered": [],
        "rejected": [],
        "suspect_ids": [],
    }
    if speech is None and waveform is not None:
        speech = detect_speech(waveform, sample_rate)
    if speech is None:
        report["recovery_enabled"] = False
        return list(segments), report
    windows = uncovered_windows(speech, segments)
    report["asr_suspicious_gaps"] = len(windows)
    flagged = []
    for segment in segments:
        reasons, largest, islands = suspect_reasons(segment, gap=gap)
        segment["suspect_reasons"] = reasons
        segment["largest_internal_gap"] = round(largest, 3)
        segment["speech_islands"] = islands
        segment["asr_suspect"] = bool(reasons)
        if reasons:
            flagged.append(segment["id"])
            report["asr_suspect_segments"] += 1
        if largest >= gap:
            report["asr_large_internal_word_gaps"] += 1
    report["suspect_ids"] = flagged
    if not windows or not report["recovery_enabled"]:
        return list(segments), report
    selected = windows[:max_windows]
    report["asr_recovery_windows"] = len(selected)
    recovered = []
    for start, end in selected:
        candidate = transcribe(start, end)
        if candidate is None:
            report["asr_recovery_rejected"] += 1
            continue
        neighbors = [s["text_zh"] for s in segments if s["end"] > start - 2 and s["start"] < end + 2]
        ok, reason = validate_recovery(candidate["text_zh"], neighbors)
        if not ok:
            report["asr_recovery_rejected"] += 1
            report["rejected"].append({"window": [round(start, 2), round(end, 2)],
                                       "text": candidate["text_zh"], "reason": reason})
            continue
        recovered.append(candidate)
        report["recovered"].append({"window": [round(start, 2), round(end, 2)],
                                    "text": candidate["text_zh"]})
    report["asr_segments_recovered"] = len(recovered)
    return merge_recovered(segments, recovered), report
