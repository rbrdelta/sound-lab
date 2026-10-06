import numpy as np

from make_clips import clip_specs, chord_notes
from hearing_test import leave_one_key_out, registers


def test_clip_set_is_balanced_and_differs_only_in_the_middle_note():
    specs = clip_specs()
    assert len(specs) == 144
    assert sum(s["label"] == "minor" for s in specs) == 72
    by = {(s["key"], s["octave"], s["arrangement"], s["label"]): s["notes"] for s in specs}
    for (key, octv, arr, q), notes in by.items():
        if q != "major":
            continue
        minor = by[(key, octv, arr, "minor")]
        diff = [m - n for m, n in zip(minor, notes)]
        assert sorted(diff) == [-1, 0, 0]  # one note a half-step lower, the rest identical


def test_chord_notes_known_values():
    assert chord_notes(60, "major", "root") == [60, 64, 67]  # C major
    assert chord_notes(60, "minor", "inv1") == [63, 67, 72]  # C minor, first inversion


def test_leave_one_key_out_recovers_planted_signal_and_ignores_noise():
    rng = np.random.default_rng(0)
    keys = np.repeat(np.arange(12), 12)
    y = np.tile([True, False], 72)
    X = rng.standard_normal((144, 50))
    noise_only = leave_one_key_out(X, y, keys)
    assert 0.3 < noise_only["accuracy"] < 0.7
    X[y, 0] += 3.0
    planted = leave_one_key_out(X, y, keys)
    assert planted["accuracy"] > 0.9 and planted["verdict"] == "pass"


def test_registers():
    rng = np.random.default_rng(1)
    X = np.vstack([np.zeros(10), 5 + rng.standard_normal((20, 10)) * 0.1])
    sil = np.array([True] + [False] * 20)
    assert registers(X, sil)["registers"]
    X2 = np.vstack([np.zeros(10), rng.standard_normal((20, 10))])
    assert not registers(X2, sil)["registers"]
