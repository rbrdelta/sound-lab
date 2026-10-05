"""Read an audio-language model's scratch work (hidden states) for a set of clips.

One deterministic forward pass per clip -- no generation, no sampling. Forward hooks record:
  (a) the audio encoder's output ("the ears"), and
  (b) the output of every language-model decoder layer ("the brain layers").
Each is mean-pooled over the audio-token positions only, giving one vector per clip per layer.

Output: <out>.npz (arrays `encoder` [n_clips, d_enc] and `lm` [n_clips, n_layers, d_lm])
        <out>.manifest.json (clip id, label, source, recording per row; model, transformers
        version, dtype, seed, prompt, layer list).

Clip list: a CSV with columns clip_id,path,label,source,recording
  - source    = the collection / recording session a clip came from (cross-source tests hold
                whole sources out)
  - recording = the original recording a clip was cut from (never split across train/test)

Usage (on the GPU pod):
  python extract.py --clips clips.csv --out runs/qwen_hearing
  python extract.py --clips clips.csv --out runs/af_hearing --model nvidia/audio-flamingo-next-hf

Status: the hook/pooling core (`read_clip`) is tested on CPU against tiny randomly-initialised
copies of both model classes (tests/test_extract.py). The processor/loading paths
(`load_model`, `build_inputs`) were written from the model cards + the installed transformers
5.6.2 source and have NOT been run against the real weights -- see UNVERIFIED notes inline.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

# Must be set before CUDA initialises for deterministic cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

QWEN = "Qwen/Qwen2-Audio-7B-Instruct"
AF_NEXT = "nvidia/audio-flamingo-next-hf"
DEFAULT_PROMPT = "Listen to this audio clip."  # placeholder: prompt wording is a design decision for Daniel
SAMPLING_RATE = 16_000  # both model cards: mono 16 kHz


def set_determinism(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # warn_only: some ops have no deterministic kernel; we still get a warning instead of silence.
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------------------------
# Model structure discovery (both Qwen2AudioForConditionalGeneration and
# MusicFlamingoForConditionalGeneration expose .audio_tower and .language_model, and
# config.audio_token_id -- checked in transformers 5.6.2 source).
# ---------------------------------------------------------------------------------------------

def find_parts(model):
    audio_tower = getattr(model, "audio_tower", None)
    lm = getattr(model, "language_model", None)
    if audio_tower is None or lm is None:
        raise RuntimeError("model has no .audio_tower / .language_model; structure not recognised")
    decoder = lm.get_decoder() if hasattr(lm, "get_decoder") else lm
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise RuntimeError("language model decoder has no .layers ModuleList")
    token_id = getattr(model.config, "audio_token_id", None)
    if token_id is None:
        token_id = getattr(model.config, "audio_token_index")
    return audio_tower, list(layers), int(token_id)


def feature_mask(batch):
    """Mel-frame validity mask; the name differs between the two processors."""
    for key in ("feature_attention_mask", "input_features_mask"):
        if key in batch and batch[key] is not None:
            return batch[key]
    return None


def encoder_valid_lengths(audio_tower, fmask, n_chunks, n_frames):
    """How many encoder output frames per chunk are real audio (not padding)."""
    import torch

    if fmask is None:
        return torch.full((n_chunks,), n_frames, dtype=torch.long)
    _, out_len = audio_tower._get_feat_extract_output_lengths(fmask.sum(-1).long())
    return out_len.long().cpu()


def _first_tensor(output):
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def read_clip(model, batch):
    """Single forward pass; returns (encoder_vec [d_enc], lm_vecs [n_layers, d_lm]) in float32.

    batch: dict of tensors already on the model's device, for ONE clip (batch size 1).
    """
    import torch

    audio_tower, layers, audio_token_id = find_parts(model)
    captured: dict = {"layers": [None] * len(layers)}

    def enc_hook(_m, _inp, out):
        captured["encoder"] = _first_tensor(out).detach().float().cpu()

    def make_layer_hook(i):
        def hook(_m, _inp, out):
            captured["layers"][i] = _first_tensor(out).detach().float().cpu()
        return hook

    handles = [audio_tower.register_forward_hook(enc_hook)]
    handles += [layer.register_forward_hook(make_layer_hook(i)) for i, layer in enumerate(layers)]
    try:
        with torch.inference_mode():
            model(**batch, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    input_ids = batch["input_ids"]
    if input_ids.shape[0] != 1:
        raise ValueError("read_clip expects one clip per forward pass (no padding effects)")
    audio_pos = (input_ids[0] == audio_token_id).cpu()
    n_audio_tokens = int(audio_pos.sum())
    if n_audio_tokens == 0:
        raise RuntimeError("no audio tokens in input_ids; processor did not expand the audio placeholder")

    # (a) encoder: [n_chunks, frames, d]; keep only valid frames of each chunk, then mean.
    enc = captured["encoder"]
    lengths = encoder_valid_lengths(audio_tower, feature_mask(batch), enc.shape[0], enc.shape[1])
    valid = torch.cat([enc[c, : int(lengths[c])] for c in range(enc.shape[0])], dim=0)
    if valid.shape[0] != n_audio_tokens:
        # Each valid encoder frame becomes one audio token in both models; a mismatch means the
        # pooling window is wrong, so refuse rather than save a silently-wrong reading.
        raise RuntimeError(f"encoder valid frames {valid.shape[0]} != audio tokens {n_audio_tokens}")
    enc_vec = valid.mean(0)

    # (b) every decoder layer: [1, seq, d] -> mean over audio-token positions.
    lm_vecs = []
    for i, h in enumerate(captured["layers"]):
        if h is None:
            raise RuntimeError(f"layer {i} hook never fired")
        if h.shape[1] != input_ids.shape[1]:
            raise RuntimeError("hidden-state length != input length (legacy un-expanded audio token path?)")
        lm_vecs.append(h[0, audio_pos].mean(0))
    return enc_vec.numpy(), torch.stack(lm_vecs).numpy()


# ---------------------------------------------------------------------------------------------
# Loading + input construction. UNVERIFIED against real weights (no GPU locally).
# ---------------------------------------------------------------------------------------------

def load_model(model_id: str, dtype: str, attn: str):
    import torch
    from transformers import AutoProcessor

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    processor = AutoProcessor.from_pretrained(model_id)
    if model_id == QWEN:
        from transformers import Qwen2AudioForConditionalGeneration as cls
    else:
        # Model card uses AutoModel; for config musicflamingo that maps to
        # MusicFlamingoForConditionalGeneration in 5.6.2. UNVERIFIED on the real checkpoint.
        from transformers import AutoModel as cls
    # `dtype=` is the transformers-5 name (model card still shows torch_dtype=, deprecated alias).
    model = cls.from_pretrained(model_id, dtype=torch_dtype, device_map="auto", attn_implementation=attn)
    model.eval()
    return processor, model


def load_audio(path: str):
    import librosa  # only needed on the pod

    audio, _ = librosa.load(path, sr=SAMPLING_RATE, mono=True)
    return audio


def build_inputs(processor, model, model_id: str, path: str, prompt: str):
    if model_id == QWEN:
        # Following the Qwen2-Audio model card: audio first, then text; chat template rendered
        # to a string, audio passed separately. transformers 5 renamed `audios=` -> `audio=`.
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": path},
                {"type": "text", "text": prompt},
            ]},
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        batch = processor(text=text, audio=[load_audio(path)], sampling_rate=SAMPLING_RATE,
                          return_tensors="pt", padding=True)
    else:
        # Following the Audio Flamingo Next model card: text then audio, processor loads the file.
        # NOTE: text-before-audio means the prompt can influence audio-position states (causal
        # attention); in Qwen's order it cannot. Keep the order fixed within a model.
        conversation = [[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": path},
            ]},
        ]]
        batch = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True,
                                              return_dict=True)
    batch = batch.to(model.device)
    # Float features must match the model dtype; ids/masks stay integer.
    for k in ("input_features",):
        if k in batch:
            batch[k] = batch[k].to(model.dtype)
    return dict(batch)


def read_clip_list(path: str):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"clip_id", "path", "label", "source", "recording"}
    missing = need - set(rows[0].keys()) if rows else need
    if missing:
        raise SystemExit(f"clip CSV missing columns: {sorted(missing)}")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", required=True, help="CSV: clip_id,path,label,source,recording")
    p.add_argument("--out", required=True, help="output prefix (writes .npz and .manifest.json)")
    p.add_argument("--model", default=QWEN, choices=[QWEN, AF_NEXT])
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn", default="eager", help="attention kernel; eager is the safest for determinism")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-repeat-check", action="store_true",
                   help="skip re-reading the first clip to confirm byte-identical output")
    args = p.parse_args(argv)

    set_determinism(args.seed)
    import torch
    import transformers

    rows = read_clip_list(args.clips)
    processor, model = load_model(args.model, args.dtype, args.attn)
    _, layers, _ = find_parts(model)

    enc_all, lm_all = [], []
    for i, r in enumerate(rows):
        batch = build_inputs(processor, model, args.model, r["path"], args.prompt)
        enc, lm = read_clip(model, batch)
        enc_all.append(enc)
        lm_all.append(lm)
        print(f"[{i + 1}/{len(rows)}] {r['clip_id']}  audio tokens read", flush=True)

    repeat_identical = None
    if not args.no_repeat_check and rows:
        batch = build_inputs(processor, model, args.model, rows[0]["path"], args.prompt)
        enc2, lm2 = read_clip(model, batch)
        repeat_identical = bool(np.array_equal(enc2, enc_all[0]) and np.array_equal(lm2, lm_all[0]))
        print(f"repeat check (first clip read twice, identical?): {repeat_identical}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), encoder=np.stack(enc_all), lm=np.stack(lm_all))
    manifest = {
        "model": args.model,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": args.dtype,
        "attn_implementation": args.attn,
        "seed": args.seed,
        "prompt": args.prompt,
        "pooling": "mean over audio-token positions; encoder: mean over valid (non-padding) frames",
        "read_point": "single forward pass, no generation, before any output token",
        "n_lm_layers": len(layers),
        "layers": ["encoder"] + [f"lm_{i}" for i in range(len(layers))],
        "repeat_check_identical": repeat_identical,
        "clips": [{k: r[k] for k in ("clip_id", "label", "source", "recording", "path")} for r in rows],
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out.with_suffix('.npz')} and {out.with_suffix('.manifest.json')}")
    if repeat_identical is False:
        print("WARNING: repeated reading differed -- the instrument is not deterministic on this setup.")
        sys.exit(2)


if __name__ == "__main__":
    main()
