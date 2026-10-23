"""CPU tests for probe.py on synthetic scratch work (no model, numpy only)."""
import numpy as np
import pytest

import probe

# --- The worksheet, primer/probe-by-hand.md, exact numbers -----------------------------------

FEAR_A = np.array([[4, 1, 2], [5, 0, 2], [3, 1, 2], [4, 2, 2]])
CALM = np.array([[1, 1, 2], [0, 2, 2], [2, 1, 2], [1, 0, 2]])
FEAR_C = np.array([[4, 1, 5], [5, 0, 4], [3, 1, 5], [4, 2, 6]])


def test_worksheet_part_a_and_b():
    det = probe.fit(FEAR_A, CALM)
    assert det.direction.tolist() == [3, 0, 0]
    assert det.cutoff == 7.5
    H = np.array([[3, 0, 1], [1, 1, 3], [2, 1, 2], [2, 2, 2]])
    y = np.array([True, False, True, False])
    assert det.scores(H).tolist() == [9, 3, 6, 6]
    assert probe.accuracy(det, H, y) == 0.75  # 3 out of 4: H3 (fearful, scores 6) is missed


def test_worksheet_part_c_loudness_trap():
    det = probe.fit(FEAR_C, CALM)
    assert det.direction.tolist() == [3, 0, 3]
    assert det.cutoff == 18
    same_session = np.array([[4, 0, 5], [1, 2, 2]])
    assert probe.accuracy(det, same_session, np.array([True, False])) == 1.0
    elsewhere = np.array([[1, 1, 6], [4, 1, 1]])  # L: loud cheerful calm; Q: quiet tense fear
    assert det.scores(elsewhere).tolist() == [21, 15]
    assert probe.accuracy(det, elsewhere, np.array([False, True])) == 0.0


# --- Synthetic data helpers ------------------------------------------------------------------

def make_clips(rng, n_rec, clips_per_rec, d, mean_fn, label, source):
    X, labels, recs, srcs = [], [], [], []
    for r in range(n_rec):
        for _ in range(clips_per_rec):
            X.append(mean_fn() + rng.normal(0, 1, d))
            labels.append(label)
            recs.append(f"{source}-{label}-{r}")
            srcs.append(source)
    return np.array(X), labels, recs, srcs


def concat(*parts):
    X = np.vstack([p[0] for p in parts])
    return X, sum((p[1] for p in parts), []), sum((p[2] for p in parts), []), sum((p[3] for p in parts), [])


def test_recovers_planted_direction():
    rng = np.random.default_rng(1)
    d = 64
    planted = rng.normal(0, 1, d)
    planted /= np.linalg.norm(planted)
    fear = make_clips(rng, 20, 3, d, lambda: 1.5 * planted, "fear", "s1")
    calm = make_clips(rng, 20, 3, d, lambda: -1.5 * planted, "calm", "s1")
    X, labels, recs, srcs = concat(fear, calm)
    noise_layer = rng.normal(0, 1, X.shape)
    results, info = probe.run({"signal": X, "noise": noise_layer}, labels, recs, srcs, "fear", "calm", seed=0)
    by = {r.layer: r for r in results}
    assert by["signal"].heldout_accuracy > 0.9 and by["signal"].heldout_verdict == "pass"
    assert by["signal"].heldout_ranking > 0.95
    assert 0.25 < by["noise"].heldout_ranking < 0.75
    # the direction found points the planted way
    det = probe.fit(X[np.array(labels) == "fear"], X[np.array(labels) == "calm"])
    cos = det.direction @ planted / np.linalg.norm(det.direction)
    assert cos > 0.8
    # split is by recording: no recording on both sides
    assert not set(info["train_recordings"]) & set(info["test_recordings"])


def test_split_never_breaks_a_recording():
    groups = np.repeat([f"r{i}" for i in range(10)], 4)
    test = probe.split_by_group(groups, 0.3, seed=3)
    for g in set(groups):
        assert len(set(test[groups == g])) == 1


def test_loudness_trap_passes_same_source_fails_cross_source():
    """Worksheet Part C at scale: emotion signal is weak, loudness is confounded with label in
    the training source. Same-source unseen clips pass; another source with the confound
    reversed exposes the detector as a loudness meter."""
    rng = np.random.default_rng(2)
    # dims: 0 = weak emotion signal, 1..6 = nuisance, 7 = loudness
    def m(emotion, loud):
        v = np.zeros(8)
        v[0] = emotion
        v[7] = loud
        return v
    s1_fear = make_clips(rng, 12, 3, 8, lambda: m(1.0, 8.0), "fear", "session1")
    s1_calm = make_clips(rng, 12, 3, 8, lambda: m(0.0, 0.0), "calm", "session1")
    s2_fear = make_clips(rng, 6, 3, 8, lambda: m(1.0, 0.0), "fear", "elsewhere")  # quiet tense
    s2_calm = make_clips(rng, 6, 3, 8, lambda: m(0.0, 8.0), "calm", "elsewhere")  # loud cheerful
    X, labels, recs, srcs = concat(s1_fear, s1_calm, s2_fear, s2_calm)
    (r,), _ = probe.run({"layer": X}, labels, recs, srcs, "fear", "calm", cross_sources=["elsewhere"], seed=0)
    assert r.heldout_verdict == "pass"
    assert r.cross_accuracy < 2 / 3 and r.cross_verdict == "fail"
    assert r.cross_ranking < 0.5  # it ranks them backwards: loudness, not fear


def test_verdict_bands():
    assert probe.verdict(0.66) == "fail"
    assert probe.verdict(2 / 3) == "usable-more-clips"
    assert probe.verdict(0.80) == "usable-more-clips"
    assert probe.verdict(0.81) == "pass"


def test_ranking_ties_count_half():
    det = probe.Detector(direction=np.array([1.0]), cutoff=0.0)
    assert probe.ranking_score(det, np.array([[1.0], [1.0]]), np.array([True, False])) == 0.5


def test_determinism_same_input_same_output():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (60, 16))
    labels = ["fear"] * 30 + ["calm"] * 30
    recs = [f"r{i // 2}" for i in range(60)]
    srcs = ["a"] * 40 + ["b"] * 20
    out1 = probe.run({"l": X.copy()}, labels, recs, srcs, "fear", "calm", cross_sources=["b"], seed=7)
    out2 = probe.run({"l": X.copy()}, labels, recs, srcs, "fear", "calm", cross_sources=["b"], seed=7)
    assert repr(out1) == repr(out2)
    assert probe.report(*out1) == probe.report(*out2)


def test_split_with_labels_puts_both_classes_in_the_test_set():
    groups = np.array([f"f{i}" for i in range(10)] + [f"c{i}" for i in range(10)])
    labels = np.array(["fear"] * 10 + ["calm"] * 10)
    for seed in range(5):
        test = probe.split_by_group(groups, 0.3, seed=seed, labels=labels)
        assert set(labels[test]) == {"fear", "calm"}
        assert test.sum() == 6  # 3 of 10 recordings from each class


def test_run_refuses_when_training_lacks_a_label():
    X = np.zeros((4, 2))
    labels = ["fear", "fear", "calm", "calm"]
    recs = ["a", "b", "c", "d"]
    srcs = ["s1", "s1", "s2", "s2"]  # holding out s2 leaves no calm clip to train on
    with pytest.raises(ValueError, match="lacks one of the two labels"):
        probe.run({"l": X}, labels, recs, srcs, "fear", "calm", cross_sources=["s2"])


def test_run_ignores_clips_with_other_labels():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (30, 4))
    labels = ["fear"] * 12 + ["calm"] * 12 + ["silence"] * 6
    recs = [f"r{i}" for i in range(30)]
    _, info = probe.run({"l": X}, labels, recs, ["s"] * 30, "fear", "calm", seed=0)
    assert info["n_train"] + info["n_test"] == 24
    assert not any(r in info["train_recordings"] + info["test_recordings"] for r in recs[24:])


def test_load_run_names_layers_and_reads_clip_columns(write_run):
    enc = np.zeros((3, 5))
    lm = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    clips = [{"label": "fear", "source": "A", "recording": "r0"}, {"label": "calm", "source": "A", "recording": "r1"},
             {"label": "silence", "source": "silence", "recording": "silence"}]
    feats, labels, recs, srcs, man = probe.load_run(write_run(enc, lm, clips))
    assert list(feats) == ["encoder", "lm_0", "lm_1", "lm_2", "lm_3"]
    assert feats["lm_2"].tolist() == lm[:, 2, :].tolist()
    assert labels == ["fear", "calm", "silence"] and recs == ["r0", "r1", "silence"] and srcs == ["A", "A", "silence"]
    assert man["model"] == "test-model"


def test_cli_writes_per_layer_json(write_run, tmp_path, capsys):
    import json
    rng = np.random.default_rng(6)
    y = np.repeat([True, False], 20)
    enc = rng.normal(0, 1, (40, 6))
    enc[y, 0] += 4.0
    clips = [{"label": "fear" if v else "calm", "source": "s", "recording": f"r{i}"} for i, v in enumerate(y)]
    prefix = write_run(enc, rng.normal(0, 1, (40, 2, 6)), clips)
    out = tmp_path / "out.json"
    probe.main([prefix, "--positive", "fear", "--negative", "calm", "--json", str(out)])
    got = json.loads(out.read_text())
    assert [l["layer"] for l in got["layers"]] == ["encoder", "lm_0", "lm_1"]
    assert got["layers"][0]["heldout_verdict"] == "pass"
    assert "Pass mark" in capsys.readouterr().out
