"""Targeted severe-CPS condensation with cache and conditional verification."""
import json
import re
from subtitle_presentation import refine, visible

CAUSAL = ("because", "so that", "so ", "therefore", "thus", "hence", "in order to", "if ", "unless", "when ", "before", "after", "while", "until")
LORE = ("reincarnation", "dao", "heaven", "tribulation", "sage", "realm", "cultivation", "cycle", "essence", "soul", "tribulation", "immortal")


def numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def risk(source_zh, old, new):
    reasons = []
    if numbers(old) - numbers(new):
        reasons.append("numbers")
    if len(old) and visible(new) / visible(old) < .55:
        reasons.append("aggressive_compression")
    if any(word in old.lower() for word in LORE) or any(k in source_zh for k in ("轮回", "天道", "洪荒", "因果", "业火", "转世")):
        reasons.append("lore")
    if any(word in old.lower() for word in CAUSAL) != any(word in new.lower() for word in CAUSAL):
        reasons.append("causal_temporal")
    if old.count(",") != new.count(",") and old.count(",") > new.count(","):
        reasons.append("clause_collapse")
    return reasons


def condense(translator, segments, cues, qc):
    records = []
    counters = dict(targeted=0, cache_hits=0, cache_misses=0, generation=0, verification=0,
                     verification_avoided=0, accepted=0, rejected_no_improvement=0,
                     rejected_terminology=0, rejected_semantic_risk=0)
    final = [dict(s) for s in segments]
    ids = {c["source_id"] for c in cues if visible(c["text"]) / (c["end"] - c["start"]) > 25}
    original_prompt = translator.prompt
    memory = getattr(translator, "memory", None)
    for index, source in enumerate(final):
        if source["id"] not in ids:
            continue
        counters["targeted"] += 1
        old = source["en"]
        duration = sum(c["end"] - c["start"] for c in cues if c["source_id"] == source["id"])
        style = translator.args.translation_style
        if memory is not None:
            cached = memory.condensation_lookup(source["text_zh"], old, style, original_prompt)
            if cached is not None:
                counters["cache_hits"] += 1
                final[index] = dict(source, en=cached["translation"])
                records.append(dict(source_id=source["id"], original_translation=old,
                    condensed_translation=cached["translation"], reason="Condensation cache HIT",
                    old_cps=visible(old)/duration, new_cps=visible(cached["translation"])/duration))
                continue
            counters["cache_misses"] += 1
        previous = [dict(id=s["id"], zh=s["text_zh"], en=s["en"]) for s in final[max(0, index-2):index]]
        translator.prompt = original_prompt + (
            '\nReturn strict JSON {"translation": "...", "preservation": {"names": true, "terminology": true, '
            '"numbers": true, "conditions": true, "ownership": true, "causal_relations": true, '
            '"temporal_relations": true, "hierarchy": true, "meaning": true}}. Rewrite ONLY the target concisely '
            'for screen reading. For lore/cultivation mechanics prefer lexical shortening and preserve causal, '
            'temporal, ownership, hierarchy and action relations. If equivalence is uncertain, keep the original '
            'unchanged. ' + json.dumps({"existing_english": old, "duration": duration,
            "current_cps": visible(old)/duration, "target_cps": 20}))
        try:
            counters["generation"] += 1
            reply = translator.request_raw(source, previous, final[index+1:index+3],
                f"Condensation {index+1}/{len(final)}", 4096)
            reply = reply.strip()
            if reply.startswith("```json\n") and reply.endswith("\n```"):
                reply = reply[len("```json\n"):-len("\n```")].strip()
            payload = json.loads(reply)
            new = payload.get("translation", "").strip()
            if not new or visible(new) >= visible(old):
                counters["rejected_no_improvement"] += 1
                continue
            try:
                memory.validate(source["text_zh"], new)
                translator.memory.validate(source["text_zh"], new) if memory else None
            except ValueError:
                counters["rejected_terminology"] += 1
                continue
            reasons = risk(source["text_zh"], old, new)
            if reasons:
                counters["verification"] += 1
                translator.prompt = original_prompt + '\nVerify the replacement against the Chinese and original English. ' + \
                    json.dumps({"old": old, "proposed": new}) + \
                    '\nReturn exactly ACCEPT only if all meaning, names, numbers, conditions, ownership, causal/temporal relations, hierarchy and intent are preserved. Otherwise REJECT.'
                saved = translator.glossary
                translator.glossary = {"characters": {}, "terms": {}}
                translator.semantic_verification = True
                try:
                    verdict = translator.request_raw(source, previous, final[index+1:index+3],
                        f"Verification {index+1}/{len(final)}", 4096).strip()
                finally:
                    translator.glossary = saved
                    translator.semantic_verification = False
                if verdict != "ACCEPT":
                    counters["rejected_semantic_risk"] += 1
                    print(f"Condensation rejected (semantic risk: {reasons}) for ID {source['id']}", flush=True)
                    continue
            else:
                counters["verification_avoided"] += 1
            trial = [dict(s, en=new) if s["id"] == source["id"] else s for s in final]
            new_cues, new_qc = refine(trial, translator.glossary, translator.args.subtitle_start_pad,
                                      translator.args.subtitle_end_pad)
            if new_qc["severe_cps_over_25"] > qc["severe_cps_over_25"] or new_qc["long_lines"]:
                counters["rejected_no_improvement"] += 1
                continue
            if memory is not None:
                memory.condensation_store(source["text_zh"], old, style, original_prompt, dict(translation=new))
            records.append(dict(source_id=source["id"], original_translation=old, condensed_translation=new,
                reason="Accepted" + (" with verifier" if reasons else " deterministically"),
                old_cps=visible(old)/duration, new_cps=visible(new)/duration))
            counters["accepted"] += 1
            final, cues, qc = trial, new_cues, new_qc
        except Exception as error:
            print(f"Condensation skipped for ID {source['id']}: {error}", flush=True)
        finally:
            translator.prompt = original_prompt
    qc.update(condensation=records, condensation_targeted=counters["targeted"],
              condensation_cache_hits=counters["cache_hits"], condensation_cache_misses=counters["cache_misses"],
              condensation_generation_calls=counters["generation"],
              condensation_verification_calls=counters["verification"],
              condensation_verification_calls_avoided=counters["verification_avoided"],
              condensation_accepted=counters["accepted"],
              condensation_rejected_no_improvement=counters["rejected_no_improvement"],
              condensation_rejected_terminology=counters["rejected_terminology"],
              condensation_rejected_semantic_risk=counters["rejected_semantic_risk"])
    return cues, qc
