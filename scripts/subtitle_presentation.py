"""Deterministic, local subtitle presentation. Never translates or drops words."""
import math
import re
import statistics

MIN_GAP = .080
MIN_DURATION = .70
PREFERRED_MIN_DURATION = 1.0
MAX_DURATION = 6.0


def visible(text):
    return len(re.sub(r"\s", "", text))


def tokens(text, glossary):
    text = re.sub(r"\s+", " ", text).strip()
    expressions = list(glossary.get("characters", {}).values()) + list(glossary.get("terms", {}).values())
    expressions += re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b", text)
    expressions += re.findall(r"\b\d+(?:\.\d+)?\s+(?:years?|days?|seconds?|meters?|miles?|kg|km|percent)\b", text)
    for expression in sorted(set(expressions), key=len, reverse=True):
        text = text.replace(expression, expression.replace(" ", "\u00a0"))
    return text.split(" ")


def wrap(words):
    text = " ".join(words)
    if len(text) <= 42:
        return text.replace("\u00a0", " ")
    options = []
    for index in range(1, len(words)):
        left, right = " ".join(words[:index]), " ".join(words[index:])
        if max(len(left), len(right)) <= 42:
            score = abs(len(left) - len(right)) + (100 if min(len(left), len(right)) < 8 else 0)
            score += max(0, len(left) - 38) + max(0, len(right) - 38)
            score += boundary_penalty(words[index - 1], words[index])
            options.append((score, left, right))
    if not options:
        return None
    _, left, right = min(options)
    return (left + "\n" + right).replace("\u00a0", " ")


def boundary_penalty(left, right):
    last = left.lower().strip('".,;:!?')
    if last in {"a", "an", "the", "of", "in", "on", "at", "to", "from", "with", "for",
                "by", "into", "my", "your", "his", "her", "our", "their", "is", "are",
                "was", "were", "must", "can", "will", "would", "has", "have", "had"} or last.endswith("'s"):
        return 150
    if left.endswith(tuple(".!?")):
        return -20
    if left.endswith(tuple(",;:")):
        return -10
    if right.lower() in {"and", "but", "so", "because", "unless", "while", "although"}:
        return -8
    if left.lower() == "so":
        return 100
    return 0


def split_text(text, glossary, duration):
    words = tokens(text, glossary)
    whole = wrap(words)
    if whole is not None and duration <= MAX_DURATION:
        return [whole]
    chunks = []
    # Split at punctuation where possible, while requiring each chunk to wrap.
    desired = max(1, math.ceil(duration / MAX_DURATION), math.ceil(len(text) / 80))
    while words:
        target = max(20, math.ceil(len(" ".join(words)) / max(1, desired - len(chunks))))
        candidates = []
        for count in range(1, len(words) + 1):
            formatted = wrap(words[:count])
            if formatted is None:
                continue
            boundary = words[count - 1][-1:]
            penalty = 0 if count == len(words) or boundary in ".?!" else 8 if boundary in ",;:" else 16
            if count < len(words):
                penalty += boundary_penalty(words[count - 1], words[count])
            candidates.append((abs(len(" ".join(words[:count])) - target) + penalty, count, formatted))
        if not candidates:
            # An indivisible name/term over 42 chars is retained and flagged by QC.
            chunks.append(words.pop(0).replace("\u00a0", " "))
            continue
        _, count, formatted = min(candidates)
        chunks.append(formatted)
        words = words[count:]
    return chunks


def metrics(cues):
    durations = [c["end"] - c["start"] for c in cues]
    speeds = [visible(c["text"]) / max(.001, d) for c, d in zip(cues, durations)]
    return {
        "total_cues": len(cues),
        "overlaps": sum(a["end"] > b["start"] + .0001 for a, b in zip(cues, cues[1:])),
        "gap_violations": sum(a["end"] > b["start"] - MIN_GAP + .001 for a, b in zip(cues, cues[1:])),
        "over_two_lines": sum(len(c["text"].splitlines()) > 2 for c in cues),
        "long_lines": sum(any(len(line) > 42 for line in c["text"].splitlines()) for c in cues),
        "short_cues": sum(d < MIN_DURATION for d in durations),
        "high_cps_cues": sum(s > 22 for s in speeds),
        "high_cps_over_20": sum(s > 20 for s in speeds),
        "severe_cps_over_25": sum(s > 25 for s in speeds),
        "median_cps": round(statistics.median(speeds), 2) if speeds else 0,
        "p95_cps": round(sorted(speeds)[max(0, math.ceil(len(speeds) * .95) - 1)], 2) if speeds else 0,
        "max_cps": round(max(speeds, default=0), 2),
        "max_lines": max((len(c["text"].splitlines()) for c in cues), default=0),
        "max_duration": round(max(durations, default=0), 3),
    }


def speech_islands(words, gap):
    """Group words into islands separated by silence gaps larger than `gap`."""
    islands = [[words[0]]]
    for word in words[1:]:
        if word["start"] - islands[-1][-1]["end"] > gap:
            islands.append([word])
        else:
            islands[-1].append(word)
    return islands


def speech_span(segment):
    words = segment.get("words", [])
    return (words[0]["start"], words[-1]["end"]) if words else (segment["start"], segment["end"])


def refine(segments, glossary, start_pad=-.05, end_pad=.10, speech_island_gap=.8):
    cues = []
    split_count = adjustments = 0
    issues = []
    leading_trimmed = trailing_trimmed = internal_gaps = island_splits = 0
    anomalies_detected = anomalies_fixed = fallback_capped = 0
    corrections = []
    word_bounds_used = sum(bool(s.get("words")) for s in segments)
    long_before = sum(1 for s in segments if speech_span(s)[1] - speech_span(s)[0] > MAX_DURATION + .001)
    for index, segment in enumerate(segments):
        words = segment.get("words", [])
        speech_start, speech_end = speech_span(segment)
        if words:
            if speech_start - segment["start"] > .3:
                leading_trimmed += 1
            if segment["end"] - speech_end > .3:
                trailing_trimmed += 1
        following = segments[index + 1] if index + 1 < len(segments) else None
        next_source_start = ((following.get("words") or [{"start": following["start"]}])[0]["start"]
                             if following else float("inf"))
        capped = False
        cap_reason = None
        if speech_end - speech_start > MAX_DURATION + .001:
            anomalies_detected += 1
            old_start, old_end = speech_start, speech_end
            if words:
                islands = speech_islands(words, speech_island_gap)
                if len(islands) > 1:
                    internal_gaps += 1
                    dominant = max(islands, key=lambda isl: (len(isl), -isl[0]["start"]))
                    island_start, island_end = dominant[0]["start"], dominant[-1]["end"]
                    next_island = min((isl[0]["start"] for isl in islands if isl[0]["start"] > island_start),
                                      default=float("inf"))
                    readable = min(MAX_DURATION, max(MIN_DURATION, visible(segment["en"]) / 15))
                    limit = min(next_island - MIN_GAP, next_source_start - MIN_GAP)
                    speech_start = island_start
                    speech_end = min(max(island_end, island_start + readable), limit)
                    island_splits += 1
                    capped = True
                    cap_reason = "internal_speech_gap"
            else:
                readable = min(MAX_DURATION, max(1.0, visible(segment["en"]) / 15))
                speech_end = min(speech_start + readable, next_source_start - MIN_GAP)
                fallback_capped += 1
                capped = True
                cap_reason = "segment_fallback_capped"
            if capped:
                anomalies_fixed += 1
                corrections.append(dict(cue_id=segment["id"], old_start=round(old_start, 3),
                                        old_end=round(old_end, 3), new_start=round(speech_start, 3),
                                        new_end=round(speech_end, 3), reason=cap_reason))
        start = max(0, speech_start + max(-.05, min(.05, start_pad)))
        next_start = max(0, next_source_start + max(-.05, min(.05, start_pad)))
        gap = MIN_GAP if next_start - speech_end >= MIN_GAP + .05 else 0
        end = min(speech_end + max(-.10, min(.10, end_pad)), next_start - gap)
        if capped:
            end = min(end, start + MAX_DURATION)
        if end <= start:
            issues.append({"source_id": segment["id"], "error": "No positive source interval"})
            raise ValueError(f"Cannot present source ID {segment['id']} without timing drift")
        parts = split_text(segment["en"], glossary, end - start)
        split_count += len(parts) > 1
        split_gap = MIN_GAP if visible(segment["en"]) / max(.001, end - start - MIN_GAP * (len(parts) - 1)) <= 17 else 0
        available = end - start - split_gap * (len(parts) - 1)
        if available <= 0:
            raise ValueError(f"Source ID {segment['id']} too short for nonoverlapping text")
        weights = [max(1, visible(part)) for part in parts]
        position = start
        for part, weight in zip(parts, weights):
            stop = position + available * weight / sum(weights)
            cues.append({"id": len(cues) + 1, "source_id": segment["id"],
                         "start": round(position, 3), "end": round(stop, 3), "text": part})
            position = stop + split_gap
        adjustments += start != segment["start"] or end != segment["end"] or len(parts) > 1
    fixed = 0
    for current, following in zip(cues, cues[1:]):
        limit = following["start"]
        if current["end"] > limit:
            current["end"] = limit
            fixed += 1
    for cue in cues:
        if cue["start"] >= cue["end"]:
            raise ValueError("Timing normalization produced an invalid cue")
        if cue["end"] - cue["start"] > MAX_DURATION + .001:
            issues.append({"cue": cue["id"], "error": "Duration exceeds 6s"})
        if any(len(line) > 42 for line in cue["text"].splitlines()):
            issues.append({"cue": cue["id"], "error": "Indivisible expression exceeds 42 chars"})
    qc = metrics(cues)
    qc.update(overlaps_fixed=fixed, long_cues_split=split_count, timing_adjustments=adjustments,
               missing_word_timestamps=sum(not s.get("words") for s in segments), issues=issues)
    missing = qc["missing_word_timestamps"]
    qc.update(timing_alignment="segment" if missing == len(segments) else "mixed" if missing else "word",
              overlaps_before=metrics([dict(start=s["start"], end=s["end"], text=s["en"]) for s in segments])["overlaps"],
              overlaps_after=qc["overlaps"], preferred_gaps_broken=qc["gap_violations"],
              needs_condensation=[c["id"] for c in cues if visible(c["text"])/(c["end"]-c["start"]) > 20],
              awkward_split_warnings=[c["id"] for c in cues[:-1] if boundary_penalty(c["text"].split()[-1], "") >= 150])
    qc["splits_proposed"] = split_count
    qc["splits_rejected_due_to_cps"] = 0
    qc["translationese"] = {phrase: sum(c["text"].lower().count(phrase.lower()) for c in cues)
        for phrase in ("Could it be that", "This matter", "You actually dare", "It seems that",
                       "In that case", "Naturally", "Indeed")}
    qc["asr_suspect"] = [s["id"] for s in segments if re.search(r"(.)\1{3,}", s.get("text_zh", ""))
        or s.get("avg_logprob", 0) < -1 or s.get("no_speech_prob", 0) > .6]
    qc.update(long_duration_cues_before=long_before,
              long_duration_cues_after=sum(c["end"] - c["start"] > MAX_DURATION + .001 for c in cues),
              word_bounds_used=word_bounds_used, leading_silence_trimmed=leading_trimmed,
              trailing_silence_trimmed=trailing_trimmed, internal_speech_gaps_detected=internal_gaps,
              speech_island_splits=island_splits, timing_anomalies_detected=anomalies_detected,
              timing_anomalies_fixed=anomalies_fixed, segment_fallback_capped=fallback_capped,
              timing_corrections=corrections)
    if qc["overlaps"] or qc["over_two_lines"]:
        raise ValueError("Final subtitle timing/line invariants failed")
    return cues, qc
