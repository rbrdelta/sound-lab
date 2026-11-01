"""CPU checks of extract.py's hook + pooling core on tiny, randomly initialised copies of the two
model classes (built from config -- nothing downloaded). Skipped if torch/transformers absent.

What this proves: hooks fire on the right modules, pooling picks exactly the audio-token
positions and the valid encoder frames, and two reads of the same input are identical.
What it does NOT prove: that the real checkpoints + processors behave the same (see README).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import extract  # noqa: E402

N_MEL, N_FRAMES, VALID_MEL = 128, 3000, 1000
N_AUDIO_TOK = ((VALID_MEL - 1) // 2 + 1 - 2) // 2 + 1  # 250, the models' own length formula
TEXT = dict(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)


def tiny_qwen():
    from transformers import Qwen2AudioConfig, Qwen2AudioForConditionalGeneration
    cfg = Qwen2AudioConfig(
        audio_config=dict(num_mel_bins=N_MEL, encoder_layers=1, encoder_attention_heads=2,
                          encoder_ffn_dim=32, d_model=16, max_source_positions=1500),
        text_config=dict(model_type="qwen2", **TEXT), audio_token_id=299)
    return Qwen2AudioForConditionalGeneration(cfg).eval(), "feature_attention_mask"


def tiny_flamingo():
    from transformers import MusicFlamingoConfig, MusicFlamingoForConditionalGeneration
    cfg = MusicFlamingoConfig(
        audio_config=dict(model_type="audioflamingo3_encoder", num_mel_bins=N_MEL, num_hidden_layers=1, num_attention_heads=2,
                          intermediate_size=32, hidden_size=16, max_source_positions=1500),
        text_config=dict(model_type="qwen2", **TEXT), audio_token_id=299,
        audio_bos_token_id=297, audio_eos_token_id=298)
    return MusicFlamingoForConditionalGeneration(cfg).eval(), "input_features_mask"


def make_batch(mask_key, token_id, bos=None, eos=None):
    g = torch.Generator().manual_seed(0)
    feats = torch.randn(1, N_MEL, N_FRAMES, generator=g)
    mask = torch.zeros(1, N_FRAMES, dtype=torch.long)
    mask[:, :VALID_MEL] = 1
    pre = [5, 6, 7] + ([bos] if bos is not None else [])
    post = ([eos] if eos is not None else []) + [8, 9]
    ids = torch.tensor([pre + [token_id] * N_AUDIO_TOK + post])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "input_features": feats, mask_key: mask}, len(pre)


@pytest.mark.parametrize("builder", [tiny_qwen, tiny_flamingo], ids=["qwen2-audio", "music-flamingo"])
def test_read_clip_pools_audio_positions_and_is_deterministic(builder):
    extract.set_determinism(0)
    model, mask_key = builder()
    bos = getattr(model.config, "audio_bos_token_id", None)
    eos = getattr(model.config, "audio_eos_token_id", None)
    batch, start = make_batch(mask_key, model.config.audio_token_id, bos, eos)

    enc, lm = extract.read_clip(model, batch)
    n_layers = model.config.text_config.num_hidden_layers
    assert lm.shape == (n_layers, TEXT["hidden_size"])
    assert enc.ndim == 1

    # Reference: the model's own hidden_states output, pooled by hand over the audio span.
    with torch.inference_mode():
        out = model(**batch, use_cache=False, output_hidden_states=True)
    span = slice(start, start + N_AUDIO_TOK)
    for i in range(n_layers - 1):  # last entry may have the final norm applied; skip it
        ref = out.hidden_states[i + 1][0, span].float().mean(0).numpy()
        np.testing.assert_allclose(lm[i], ref, rtol=1e-5, atol=1e-6)

    # Encoder vector = mean over exactly the valid (non-padding) encoder frames.
    tower = model.audio_tower
    with torch.inference_mode():
        kwargs = {}
        if mask_key == "input_features_mask":
            kwargs["input_features_mask"] = batch[mask_key]
        enc_out = extract._first_tensor(tower(batch["input_features"], **kwargs))
    if mask_key == "feature_attention_mask":
        # Qwen2-Audio passes a 4-D attention mask to its encoder; without it the padded frames
        # are attended to, so only check shape/finite here and rely on the token-count check.
        assert np.isfinite(enc).all()
    else:
        np.testing.assert_allclose(enc, enc_out[0, :N_AUDIO_TOK].float().mean(0).numpy(), rtol=1e-5, atol=1e-6)

    enc2, lm2 = extract.read_clip(model, batch)
    assert np.array_equal(enc, enc2) and np.array_equal(lm, lm2)


def test_refuses_when_no_audio_tokens():
    model, mask_key = tiny_qwen()
    batch, _ = make_batch(mask_key, token_id=11)
    with pytest.raises(Exception):
        extract.read_clip(model, batch)


@pytest.mark.parametrize("builder", [tiny_qwen, tiny_flamingo], ids=["qwen2-audio", "music-flamingo"])
def test_moments_match_hidden_states_and_average(builder):
    extract.set_determinism(0)
    model, mask_key = builder()
    bos = getattr(model.config, "audio_bos_token_id", None)
    eos = getattr(model.config, "audio_eos_token_id", None)
    batch, start = make_batch(mask_key, model.config.audio_token_id, bos, eos)
    W = 12

    enc, lm, m = extract.read_clip(model, batch, window=W)
    n_layers = model.config.text_config.num_hidden_layers
    n_win = -(-N_AUDIO_TOK // W)
    assert m["lm_windows"].shape == (n_win, n_layers, TEXT["hidden_size"])
    assert m["enc_windows"].shape[0] == n_win
    n_tail = batch["input_ids"].shape[1] - (start + N_AUDIO_TOK)
    assert m["lm_tail"].shape == (n_tail, n_layers, TEXT["hidden_size"])
    assert m["lm_head"].shape == (start, n_layers, TEXT["hidden_size"])
    assert m["head_ids"].tolist() == batch["input_ids"][0, :start].tolist()
    assert m["n_audio_tokens"] == N_AUDIO_TOK

    with torch.inference_mode():
        out = model(**batch, use_cache=False, output_hidden_states=True)
    for i in range(n_layers - 1):  # last entry may have the final norm applied; skip it
        hs = out.hidden_states[i + 1][0].float()
        first = hs[start:start + W].mean(0).numpy()
        np.testing.assert_allclose(m["lm_windows"][0, i], first, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(m["lm_tail"][:, i], hs[start + N_AUDIO_TOK:].numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(m["lm_head"][:, i], hs[:start].numpy(), rtol=1e-5, atol=1e-6)

    # Windows weighted by their size give back the whole-clip average from the same pass.
    sizes = np.array([min(W, N_AUDIO_TOK - a) for a in range(0, N_AUDIO_TOK, W)], dtype=np.float64)
    back = (m["lm_windows"] * sizes[:, None, None]).sum(0) / sizes.sum()
    np.testing.assert_allclose(back, lm, rtol=1e-4, atol=1e-5)
    # Asking for moments does not change the whole-clip reading.
    enc2, lm2 = extract.read_clip(model, batch)
    assert np.array_equal(enc, enc2) and np.array_equal(lm, lm2)


def test_refuses_when_encoder_frames_and_audio_tokens_disagree(monkeypatch):
    # The models check this too, so simulate our own pooling-window arithmetic going wrong.
    model, mask_key = tiny_qwen()
    batch, _ = make_batch(mask_key, model.config.audio_token_id)
    monkeypatch.setattr(extract, "encoder_valid_lengths", lambda *a: torch.tensor([N_AUDIO_TOK - 1]))
    with pytest.raises(RuntimeError, match="encoder valid frames"):
        extract.read_clip(model, batch)


def test_moments_refuse_audio_tokens_that_are_not_one_run():
    model, mask_key = tiny_qwen()
    tok = model.config.audio_token_id
    batch, start = make_batch(mask_key, tok)
    ids = batch["input_ids"][0].tolist()
    ids = ids[:start + 100] + [5] + ids[start + 100:]  # a text token in the middle of the audio
    batch["input_ids"] = torch.tensor([ids])
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    with pytest.raises(RuntimeError, match="not one contiguous run"):
        extract.read_clip(model, batch, window=12)


def test_find_parts_refuses_an_unrecognised_model():
    with pytest.raises(RuntimeError, match="structure not recognised"):
        extract.find_parts(torch.nn.Linear(2, 2))


def test_clip_list_must_have_the_five_columns(tmp_path):
    p = tmp_path / "clips.csv"
    p.write_text("clip_id,path,label,source\na,a.wav,fear,s\n")
    with pytest.raises(SystemExit, match="recording"):
        extract.read_clip_list(str(p))


def test_main_writes_readings_manifest_and_moments(tmp_path, monkeypatch):
    """The whole extract.py run with the real-weights steps swapped for a tiny model."""
    import json
    import sys
    import types
    model, mask_key = tiny_qwen()
    batch, _ = make_batch(mask_key, model.config.audio_token_id)
    monkeypatch.setattr(extract, "load_model", lambda *a: (None, model))
    monkeypatch.setattr(extract, "build_inputs", lambda *a: dict(batch))
    monkeypatch.setitem(sys.modules, "librosa", types.SimpleNamespace(get_duration=lambda path: 2.5))
    clips = tmp_path / "clips.csv"
    clips.write_text("clip_id,path,label,source,recording\na,a.wav,fear,s1,ra\nb,b.wav,calm,s1,rb\n")
    out = tmp_path / "runs" / "t"
    extract.main(["--clips", str(clips), "--out", str(out), "--moments-window", "100"])

    data = np.load(out.with_suffix(".npz"))
    n_layers, d = TEXT["num_hidden_layers"], TEXT["hidden_size"]
    assert data["lm"].shape == (2, n_layers, d) and data["encoder"].shape[0] == 2
    man = json.loads(out.with_suffix(".manifest.json").read_text())
    assert man["repeat_check_identical"] is True
    assert man["layers"] == ["encoder"] + [f"lm_{i}" for i in range(n_layers)]
    assert [c["recording"] for c in man["clips"]] == ["ra", "rb"] and man["moments_window_tokens"] == 100
    assert man["prompt"] == extract.DEFAULT_PROMPT and man["seed"] == 0 and man["dtype"] == "bf16"
    m = np.load(out.with_name("t.moments.npz"))
    assert m["lm_windows"].dtype == np.float16 and m["window_clip"].tolist() == [0, 0, 0, 1, 1, 1]  # 250 tokens / 100
    assert m["n_audio_tokens"].tolist() == [N_AUDIO_TOK] * 2 and m["seconds"].tolist() == [2.5, 2.5]
    assert len(m["tail_clip"]) == len(m["tail_ids"]) == m["lm_tail"].shape[0]


def test_main_sets_the_window_in_seconds_from_the_measured_token_rate(tmp_path, monkeypatch):
    """--moments-seconds: the window is measured, not assumed, and recorded in the manifest."""
    import json
    import sys
    import types
    model, mask_key = tiny_qwen()
    batch, start = make_batch(mask_key, model.config.audio_token_id)
    monkeypatch.setattr(extract, "load_model", lambda *a: (None, model))
    monkeypatch.setattr(extract, "build_inputs", lambda *a: dict(batch))
    # 250 audio tokens over 2.5 s = 100 tokens per second, so 0.5 s = 50 tokens = 5 windows per clip.
    monkeypatch.setitem(sys.modules, "librosa", types.SimpleNamespace(get_duration=lambda path: 2.5))
    clips = tmp_path / "clips.csv"
    clips.write_text("clip_id,path,label,source,recording\na,a.wav,fear,s1,ra\nb,b.wav,calm,s1,rb\n")
    out = tmp_path / "runs" / "s"
    extract.main(["--clips", str(clips), "--out", str(out), "--moments-seconds", "0.5"])

    man = json.loads(out.with_suffix(".manifest.json").read_text())
    assert man["audio_tokens_per_second"] == pytest.approx(100.0)
    assert man["moments_window_tokens"] == 50 and man["moments_seconds_requested"] == 0.5
    m = np.load(out.with_name("s.moments.npz"))
    assert m["window_clip"].tolist() == [0] * 5 + [1] * 5
    assert m["head_clip"].tolist() == [0] * start + [1] * start
    assert len(m["head_ids"]) == m["lm_head"].shape[0] == 2 * start
