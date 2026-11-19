"""Control-aware synthesis for P2 (Track B), kept separate from mt_adapters so the
running P0/P1 pipeline is not disturbed.

Native-support matrix verified against source (see CONTROL.md):
  indextts25        rate(duration_factor) emotion(emo_vector) pronunciation(<词|拼音>)
  cosyvoice3        rate(speed) emotion/intonation(inference_instruct2)
  fireredtts3_base  none native -> text fallback only
  fireredtts3_instruct  generate_tts is ICL cloning WITHOUT an instruct param;
                    instruction lives in generate_voice_design (designs a voice,
                    not clone-conditioned) -> treated as text fallback for cloning
  voxcpm2           none native
  qwen3tts_base     none native (instruct is on CustomVoice/VoiceDesign variants)

Every item records control_mode = native | text_fallback, so a text-driven result
is never counted as native control capability.
"""
import os
import sys

import numpy as np

import mt_common as C
import mt_adapters as A

NATIVE = {
    "indextts25": {"rate", "emotion", "pronunciation"},
    "cosyvoice3": {"rate", "emotion", "intonation"},
    "fireredtts3_base": set(),
    "fireredtts3_instruct": set(),
    "voxcpm2": set(),
    "qwen3tts_base": set(),
}


def _pinyin_text(text, hints):
    """Apply hints to successive occurrences without touching earlier inserts.

    IndexTTS expects <字|PINYIN+TONE>, e.g. 银<行|HANG2>.  The old implementation
    used text.replace(), so the second hint for the same character could match
    the character inside the first inserted annotation and corrupt it.  Build
    the result with a cursor over the *original* text instead.
    Returns (annotated_text, applied_list).
    """
    if not hints:
        return text, []
    out = []
    pos = 0
    applied = []
    for pair in hints.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        if "|" not in pair:
            raise ValueError(f"bad pinyin hint {pair!r}")
        w, r = pair.split("|", 1)
        i = text.find(w, pos)
        if i < 0:
            raise ValueError(f"hint char {w!r} not found after pos {pos} in {text!r}")
        out.append(text[pos:i])
        reading = r.strip().upper()
        out.append(f"<{w}|{reading}>")
        pos = i + len(w)
        applied.append(f"{w}|{reading}")
    out.append(text[pos:])
    return "".join(out), applied


def synth_control(cid, handle, item, ref, out_wav, baseline_cpm=None):
    """Return (wav, sr, extra). extra carries control_mode / applied params."""
    sub = item["sub"]
    text = item["text"]
    native = sub in NATIVE.get(cid, set())
    seed = int(item.get("seed", 42))
    extra = {"control_mode": "native" if native else "text_fallback",
             "sub": sub, "seed": seed,
             "seed_supported": cid in {"voxcpm2", "fireredtts3_base",
                                       "fireredtts3_instruct"}}
    opts = {"seed": seed}

    if cid == "indextts25":
        if sub == "rate" and native and baseline_cpm:
            # duration_factor is a duration multiplier: higher = slower.
            factor = baseline_cpm / float(item["target_cpm"])
            opts["duration_factor"] = round(factor, 4)
            extra["duration_factor"] = opts["duration_factor"]
            extra["target_cpm"] = item["target_cpm"]
        elif sub == "emotion" and native:
            opts["emo_vector"] = item["emo_vector"]
            extra["emotion"] = item["emotion"]
        elif sub == "pronunciation" and native:
            if item.get("hint_text"):
                text = item["hint_text"]
                applied = []
            else:
                text, applied = _pinyin_text(text, item.get("pinyin_hints", ""))
            extra["pinyin_applied"] = bool(applied) or bool(item.get("hint_text"))
            extra["applied_hints"] = applied
            extra["applied_text"] = text
        return A.synth_indextts25(handle, text, ref, out_wav, opts)[0:2] + (extra,)

    if cid == "cosyvoice3":
        import torch
        # prompt_wav must be a FILE PATH for this build (frontend calls load_wav
        # internally); passing a preloaded tensor raises "Invalid file: tensor".
        prompt = ref["ref_wav"]
        speed = 1.0
        if sub == "rate" and native and baseline_cpm:
            speed = float(item["target_cpm"]) / baseline_cpm
            extra["target_cpm"] = item["target_cpm"]
        extra["speed"] = round(speed, 4)
        if sub in ("emotion", "intonation") and native:
            instr = item.get("instruct_text") or item.get("intonation_label", "")
            # CosyVoice3 instruct format: "You are a helpful assistant. <instr><|endofprompt|>"
            instruct_text = "You are a helpful assistant. " + instr + "<|endofprompt|>"
            gen = handle["model"].inference_instruct2(text, instruct_text, prompt,
                                                      stream=False, speed=speed)
            extra["instruct_text"] = instruct_text
        else:
            prompt_text = ref["ref_text"] + "<|endofprompt|>"
            gen = handle["model"].inference_zero_shot(text, prompt_text, prompt,
                                                      stream=False, speed=speed)
        chunks = [o["tts_speech"] for o in gen]
        if not chunks:
            raise RuntimeError("CosyVoice3 returned no chunks")
        wav = torch.cat(chunks, dim=1).squeeze(0).numpy().astype(np.float32)
        return wav, handle["sr"], extra

    # configs without native control: run the normal zero-shot path as fallback
    wav, sr, base_extra = A.SYNTH[cid](handle, text, ref, out_wav, opts)
    extra.update(base_extra)
    if sub == "rate":
        extra["target_cpm"] = item.get("target_cpm")
    if sub == "emotion":
        extra["emotion"] = item.get("emotion")
    return wav, sr, extra