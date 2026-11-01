"""score_decoding.py on synthetic readings only (no real reading is opened by these tests).

Synthetic clips: noise everywhere, a shared "this is music" component on every clip except
silence, and, at layer lm_1 only, either a separate direction per emotion (the planted emotion
signal) or one loudness direction whose sign follows energy (the loudness trap).
"""
import json

import numpy as np
import pytest

import score_decoding as sd

EMO = sd.EMOTIONS
ENERGY = {"anger": 1, "happy": 1, "fear": 1, "sad": -1, "tender": -1}


def make_layers(signal="emotion", n_per=20, d=48, seed=0, opposite_happy_sad=False, neutral_noise=1.0):
    rng = np.random.default_rng(seed)
    labels = np.array([e for e in EMO for _ in range(n_per)] + ["neutral"] * 15 + ["silence", "flat_tone"])
    n = len(labels)
    music = rng.normal(size=d)
    music *= 6 / np.linalg.norm(music)
    tone = rng.normal(size=d)
    tone *= 4 / np.linalg.norm(tone)
    emo_vec = {e: rng.normal(size=d) for e in EMO}
    if opposite_happy_sad:
        emo_vec["sad"] = -emo_vec["happy"]
    emo_vec = {e: 3 * v / np.linalg.norm(v) for e, v in emo_vec.items()}
    loud = rng.normal(size=d)
    loud *= 3 / np.linalg.norm(loud)

    mats = []
    for layer in range(3):                       # encoder, lm_0, lm_1
        X = rng.normal(scale=1.0, size=(n, d))
        for i, lab in enumerate(labels):
            if lab == "silence":
                X[i] *= 0.1
                continue
            if lab == "neutral":
                X[i] *= neutral_noise
            X[i] += tone if lab == "flat_tone" else music
            if layer == 2 and lab in EMO:
                X[i] += emo_vec[lab] if signal == "emotion" else ENERGY[lab] * loud
        mats.append(X)
    return ["encoder", "lm_0", "lm_1"], mats, labels


def test_planted_emotion_signal_is_recovered_at_the_right_layer():
    names, mats, labels = make_layers("emotion")
    res = sd.score_run(names, mats, labels, n_perm=40, seed=0)
    pr = res["primary"]
    assert pr["layer"] == "lm_1"
    five = pr["checks"]["all_five"]
    assert five["fraction"] >= 0.9 and five["at_or_above_mark"]
    assert five["label_shuffle"]["share_at_or_above_observed"] == 0.0
    assert five["guessing_gets_this_or_more"] < 1e-6
    for chk in ("happy_vs_sad", "happy_vs_fear", "tender_vs_sad"):
        assert pr["checks"][chk]["fraction"] >= 0.9
    # The layers with no emotion signal sit near guessing.
    assert res["by_reference"]["silence"]["encoder"]["all_five"]["fraction"] < 0.45


def test_loudness_only_signal_passes_happy_sad_and_fails_the_energy_matched_checks():
    names, mats, labels = make_layers("loudness")
    at = sd.score_run(names, mats, labels, n_perm=0)["by_reference"]["silence"]["lm_1"]
    assert at["happy_vs_sad"]["fraction"] >= 0.9                    # step 1 passes on loudness alone
    assert at["happy_vs_fear"]["fraction"] < sd.PASS_TWO_WAY        # both loud: loudness can't carry it
    assert at["tender_vs_sad"]["fraction"] < sd.PASS_TWO_WAY        # both quiet: same


def test_strip_removes_exactly_the_arrow_component_and_keeps_the_rest():
    rng = np.random.default_rng(1)
    v, arrow = rng.normal(size=(5, 30)), rng.normal(size=30)
    s = sd.strip(v, arrow)
    np.testing.assert_allclose(s @ sd.unit(arrow), 0, atol=1e-12)   # nothing left along the arrow
    removed = v - s
    for row in removed:                                                # what was removed IS the arrow
        assert abs(abs(sd.cosine(row, arrow)) - 1) < 1e-9
    # 1-D form agrees with the matrix form.
    np.testing.assert_allclose(sd.strip(v[0], arrow), s[0])
    # Stripping is not subtraction: subtracting the arrow would leave a component along it.
    assert abs((v - arrow) @ sd.unit(arrow)).max() > 1e-3


@pytest.mark.parametrize("ref", sd.REFERENCES)
def test_leave_one_out_never_uses_the_scored_clip(ref):
    """Fast scores equal a brute-force rebuild with the clip deleted from the data entirely."""
    names, mats, labels = make_layers("emotion", n_per=6, d=12, seed=3)
    X = mats[2]
    idx, S = sd.loo_scores(X, labels, ref)
    for row, i in enumerate(idx):
        keep = np.arange(len(labels)) != i
        X2, labels2 = X[keep], labels[keep]
        U = sd.unit(sd.directions(X2, labels2, ref))
        r = sd.reference_mean(X2, labels2, ref)
        np.testing.assert_allclose(S[row], (X[i] - r) @ U.T, rtol=1e-9, atol=1e-9)


def test_neutral_reference_is_not_stripped_because_stripping_it_would_copy_silence():
    names, mats, labels = make_layers("emotion", seed=4)
    X = mats[2]
    sil = sd.directions(X, labels, "silence")                          # stripped
    neu_raw = sd.directions(X, labels, "neutral")                      # subtracted, not stripped
    np.testing.assert_allclose(sd.strip(neu_raw, sd.music_arrow(X, labels)), sil, atol=1e-9)
    assert not np.allclose(neu_raw, sil)                               # so the comparison can differ


def test_label_shuffle_is_repeatable_with_a_seed():
    names, mats, labels = make_layers("emotion", n_per=8, d=16, seed=5)
    a = sd.label_shuffle(mats, labels, "silence", 15, seed=7)
    b = sd.label_shuffle(mats, labels, "silence", 15, seed=7)
    c = sd.label_shuffle(mats, labels, "silence", 15, seed=8)
    assert a == b and a != c
    assert set(a) == {"happy_vs_sad", "happy_vs_fear", "tender_vs_sad", "all_five"}


def test_sound_check_tells_registering_sound_from_silence_like_clips():
    """The plan's rule passes only when the jump from silence to the clips is larger than the
    clips' spread among themselves (flagged in DECISIONS)."""
    names, mats, labels = make_layers("emotion")
    rng = np.random.default_rng(6)
    sil = labels == "silence"
    shift = np.zeros(mats[2].shape[1]); shift[0] = 10.0
    spread = rng.normal(scale=0.5, size=mats[2].shape)
    registering = np.where(sil[:, None], 0.0, shift + spread)
    assert sd.sound_check(registering, labels)["ratio"] > 1            # sound sits clearly away from silence
    silent_like = np.where(sil[:, None], 0.0, spread)
    assert sd.sound_check(silent_like, labels)["ratio"] < 1            # clips are just silence plus noise
    # The same planted-emotion clips fail the rule when their spread exceeds the jump from silence.
    assert sd.sound_check(mats[2], labels)["ratio"] < 1


def test_opposites_and_reference_comparison():
    names, mats, labels = make_layers("emotion", opposite_happy_sad=True)
    opp = sd.opposites(mats[2], labels)
    assert opp["happy_vs_sad"] < 0 and opp["happy_vs_sad"] == min(opp.values())
    assert abs(opp["anger_vs_tender"]) < 0.6
    # Known pull toward zero (flagged in DECISIONS): the music arrow, estimated from 15 noisy neutral
    # clips, leaves leftover music in every direction after stripping. The same planted opposites
    # read clearly more opposite when the neutral clips carry no noise.
    _, clean, _ = make_layers("emotion", opposite_happy_sad=True, neutral_noise=0.0)
    clean_opp = sd.opposites(clean[2], labels)["happy_vs_sad"]
    assert clean_opp < -0.7 and opp["happy_vs_sad"] > clean_opp + 0.2
    cmp = sd.compare_references(mats[2], labels)
    assert set(cmp) == {"silence_vs_neutral", "silence_vs_flat_tone", "neutral_vs_flat_tone"}
    assert all(-1 <= v <= 1 for pair in cmp.values() for v in pair.values())


def test_clear_clips_are_reported_separately():
    names, mats, labels = make_layers("emotion")
    clear = np.isin(labels, EMO) & (np.arange(len(labels)) % 4 != 0)
    res = sd.score_run(names, mats, labels, clear=clear, n_perm=0)
    c = res["primary"]["checks"]["all_five_clear"]
    assert c["total"] == int(clear.sum()) and c["pass_mark"] == sd.PASS_FIVE_WAY


def test_timeline_follows_a_signal_that_arrives_late_and_margin_has_no_verdict():
    names, mats, labels = make_layers("emotion", seed=9)
    X = mats[2]
    rng = np.random.default_rng(10)
    U = sd.unit(sd.directions(X, labels, "silence"))
    music = X[labels == "neutral"].mean(0)
    rows = music + rng.normal(scale=0.3, size=(8, X.shape[1]))
    rows[4:] += 4 * U[EMO.index("sad")]                                 # the sad turn halfway through
    tl = sd.timeline(rows, X, labels)
    assert tl.shape == (8, 5)
    sad = tl[:, EMO.index("sad")]
    assert sad[4:].min() > sad[:4].max()
    m = sd.margin(tl)
    assert m.shape == (8,) and (m >= 0).all()


def test_binomial_tail():
    assert sd.binom_tail(1, 1, 0.5) == pytest.approx(0.5)
    assert sd.binom_tail(0, 10, 0.2) == pytest.approx(1.0)
    assert sd.binom_tail(10, 10, 0.5) == pytest.approx(0.5 ** 10)


def test_main_writes_the_score_file_with_decisions(write_run, tmp_path, capsys):
    names, mats, labels = make_layers("emotion", n_per=8, d=16, seed=11)
    enc = mats[0]
    lm = np.stack(mats[1:], axis=1)
    clips = [{"label": str(lab), "source": "s", "recording": f"r{i}"} for i, lab in enumerate(labels)]
    prefix = write_run(enc, lm, clips, name="film")
    csv_path = tmp_path / "clips.csv"
    csv_path.write_text("clip_id,label,listeners_top\n" +
                        "".join(f"c{i},{lab},{lab if i % 2 else 'fear'}\n" for i, lab in enumerate(labels)))
    sd.main([prefix, "--clips", str(csv_path), "--perm", "5"])
    res = json.loads((tmp_path / "film.score.json").read_text())
    assert res["primary"]["layer"] in res["layers"] and res["decisions"] == sd.DECISIONS
    assert "all_five_clear" in res["primary"]["checks"]
    assert set(res["by_reference"]) == set(sd.REFERENCES)
    out = capsys.readouterr().out
    assert "Primary layer" in out and "p-value" not in out and "significant" not in out
