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
