"""Per-config inference adapters — signatures verified against repo source.

Verified 2026-11-11 against:
  VoxCPM      src/voxcpm/core.py:137 from_pretrained, :207 _generate
  index-tts   indextts/infer_v2_5.py:77 __init__, :506 infer
  CosyVoice   cosyvoice/cli/cosyvoice.py:189 CosyVoice3.__init__, :91 inference_zero_shot
  Qwen3-TTS   qwen_tts/inference/qwen3_tts_model.py:470 generate_voice_clone
  FireRedTTS3 fireredtts3/core.py:194 FireRedTTS3, :302 FireRedTTS3Instruct

Corrections made after reading source (the first draft was wrong on all four):
  * CosyVoice3.__init__ has NO load_jit parameter -> removed.
  * IndexTTS speed control is `duration_factor`, NOT `speed`; passing `speed`
    would land in **generation_kwargs and be silently ignored.
  * Qwen3-TTS uses its own qwen_tts.Qwen3TTSModel, not transformers AutoModel.
  * FireRedTTS3 lives in fireredtts3.core (no api.py), takes
    `pretrained_model_dir`, and language tags are "Chinese"/"English".
    use_fasttext defaults True but the lid.176 model was deliberately not
    downloaded, so it is disabled and language is passed explicitly.

Status vocabulary: ok / failed / unsupported / not_tested (kept distinct).
"""
import os
import sys

import numpy as np

import mt_common as C


def _add_repo(name):
    p = os.path.join(C.E, "repos", name)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def _to_tensor_16k_or_native(ref, want_sr=None):
    """Return (torch.Tensor [1,T], sr) for prompt-audio APIs."""
    import torch
    import torchaudio
    w, sr = C.read_wav(ref["ref_wav"])
    t = torch.from_numpy(w).float().unsqueeze(0)
    if want_sr is not None and sr != want_sr:
        t = torchaudio.functional.resample(t, sr, want_sr)
        sr = want_sr
    return t, sr


# ---------------- VoxCPM2 ----------------
def load_voxcpm2(cfg, device):
    from voxcpm import VoxCPM
    m = VoxCPM.from_pretrained(cfg["model_dir"], load_denoiser=False,
                               local_files_only=True, device=device)
    return {"model": m, "sr": int(m.tts_model.sample_rate)}


def synth_voxcpm2(h, text, ref, out_wav, opts):
    wav = h["model"].generate(
        text=text,
        reference_wav_path=ref["ref_wav"],
        cfg_value=opts.get("cfg_value", 2.0),
        inference_timesteps=opts.get("inference_timesteps", 10),
        normalize=opts.get("normalize", False),
        seed=opts.get("seed", 42),
    )
    return np.asarray(wav, dtype=np.float32).reshape(-1), h["sr"], {}


# ---------------- IndexTTS-2.5 ----------------
def load_indextts25(cfg, device):
    _add_repo("index-tts")
    from indextts.infer_v2_5 import IndexTTS2
    m = IndexTTS2(cfg_path=os.path.join(cfg["model_dir"], "config.yaml"),
                  model_dir=cfg["model_dir"], use_bf16=True, device=device)
    return {"model": m, "sr": 22050}


def synth_indextts25(h, text, ref, out_wav, opts):
    kw = dict(spk_audio_prompt=ref["ref_wav"], text=text, lang="ZH",
              output_path=out_wav)
    if opts.get("emo_vector") is not None:
        kw["emo_vector"] = opts["emo_vector"]
    # source-verified: speed is controlled by duration_factor
    if opts.get("duration_factor") is not None:
        kw["duration_factor"] = opts["duration_factor"]
    h["model"].infer(**kw)
    wav, sr = C.read_wav(out_wav)
    return wav, sr, {"written_by_model": True}


# ---------------- CosyVoice3 ----------------
def load_cosyvoice3(cfg, device):
    p = _add_repo("CosyVoice")
    mp = os.path.join(p, "third_party", "Matcha-TTS")
    if os.path.isdir(mp) and mp not in sys.path:
        sys.path.insert(0, mp)
    from cosyvoice.cli.cosyvoice import CosyVoice3
    # CosyVoice3.__init__(model_dir, load_trt, load_vllm, fp16, trt_concurrent)
    m = CosyVoice3(cfg["model_dir"], load_trt=False, load_vllm=False, fp16=False)
    return {"model": m, "sr": int(m.sample_rate)}


def synth_cosyvoice3(h, text, ref, out_wav, opts):
    import torch
    # This CosyVoice3 build's inference_zero_shot expects prompt_wav as a FILE
    # PATH: frontend._extract_speech_token/_feat/_spk_embedding each call
    # load_wav(prompt_wav) -> torchaudio.load(path). Passing a preloaded tensor
    # raises "Invalid file: tensor". Verified against cosyvoice/cli/frontend.py.
    # CosyVoice3 also requires the <|endofprompt|> token in prompt_text
    # (llm.py:479 asserts token id 151646 is present).
    prompt_text = ref["ref_text"] + "<|endofprompt|>"
    gen = h["model"].inference_zero_shot(text, prompt_text, ref["ref_wav"],
                                         stream=False,
                                         speed=opts.get("speed", 1.0),
                                         text_frontend=False)
    chunks = [o["tts_speech"] for o in gen]
    if not chunks:
        raise RuntimeError("CosyVoice3 returned no audio chunks")
    wav = torch.cat(chunks, dim=1).squeeze(0).numpy()
    return wav.astype(np.float32), h["sr"], {}


# ---------------- Qwen3-TTS Base ----------------
def load_qwen3tts_base(cfg, device):
    _add_repo("Qwen3-TTS")
    import torch
    from qwen_tts import Qwen3TTSModel
    kw = dict(device_map=device, dtype=torch.bfloat16)
    try:
        import flash_attn  # noqa
        kw["attn_implementation"] = "flash_attention_2"
    except ImportError:
        kw["attn_implementation"] = "sdpa"
    m = Qwen3TTSModel.from_pretrained(cfg["model_dir"], **kw)
    return {"model": m, "sr": 24000, "attn": kw["attn_implementation"]}


QWEN_GEN_KWARGS = dict(max_new_tokens=2048, do_sample=True, top_k=50, top_p=1.0,
                       temperature=0.9, repetition_penalty=1.05,
                       subtalker_dosample=True, subtalker_top_k=50,
                       subtalker_top_p=1.0, subtalker_temperature=0.9)


def synth_qwen3tts_base(h, text, ref, out_wav, opts):
    wavs, sr = h["model"].generate_voice_clone(
        text=text, language="Chinese",
        ref_audio=ref["ref_wav"], ref_text=ref["ref_text"],
        x_vector_only_mode=False, **QWEN_GEN_KWARGS)
    if not wavs:
        raise RuntimeError("Qwen3-TTS returned no wav")
    return np.asarray(wavs[0], dtype=np.float32).reshape(-1), int(sr), \
        {"attn": h["attn"]}


# ---------------- FireRedTTS3 (Base / Instruct) ----------------
def load_fireredtts3(cfg, device):
    _add_repo("FireRedTTS3")
    from fireredtts3.core import FireRedTTS3, FireRedTTS3Instruct
    cls = FireRedTTS3Instruct if cfg["variant"] == "Instruct" else FireRedTTS3
    # use_fasttext=False: lid.176 not downloaded; language passed explicitly.
    m = cls(pretrained_model_dir=cfg["model_dir"], use_fasttext=False)
    return {"model": m, "variant": cfg["variant"], "sr": 24000}


def synth_fireredtts3(h, text, ref, out_wav, opts):
    import torch
    audio, sr = _to_tensor_16k_or_native(ref)
    kw = dict(language="Chinese", seed=opts.get("seed", 42),
              inference_cfg=opts.get("cfg_value", 2.0))
    if h["variant"] == "Instruct":
        out = h["model"].generate_tts(prompt_text=ref["ref_text"],
                                      prompt_audio=audio, prompt_audio_sr=sr,
                                      text=text, **kw)
    else:
        out = h["model"].generate(text=text, prompt_text=ref["ref_text"],
                                  prompt_audio=audio, prompt_audio_sr=sr, **kw)
    if isinstance(out, tuple):
        wav, osr = out
    else:
        wav, osr = out, h["sr"]
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.reshape(-1) if wav.shape[0] == 1 else wav.mean(axis=0)
    return wav, int(osr), {"attn_backend": "sdpa (repo hardcoded flash_attention_2; "
                                          "patched because nvcc/flash-attn unavailable)"}


LOAD = {
    "voxcpm2": load_voxcpm2,
    "indextts25": load_indextts25,
    "cosyvoice3": load_cosyvoice3,
    "qwen3tts_base": load_qwen3tts_base,
    "fireredtts3_base": load_fireredtts3,
    "fireredtts3_instruct": load_fireredtts3,
}
SYNTH = {
    "voxcpm2": synth_voxcpm2,
    "indextts25": synth_indextts25,
    "cosyvoice3": synth_cosyvoice3,
    "qwen3tts_base": synth_qwen3tts_base,
    "fireredtts3_base": synth_fireredtts3,
    "fireredtts3_instruct": synth_fireredtts3,
}