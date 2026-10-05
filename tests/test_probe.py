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
