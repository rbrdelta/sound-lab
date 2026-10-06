import numpy as np

from make_clips import clip_specs, chord_notes, tune_specs, tune_notes, roughness_specs, flutter, SR, volume_specs, volume_envelope, VOL_PATTERNS
from hearing_test import leave_one_key_out, registers


def test_clip_set_is_balanced_and_differs_only_in_the_middle_note():
    specs = clip_specs()
    assert len(specs) == 144
    assert sum(s["label"] == "minor" for s in specs) == 72
    by = {(s["key"], s["octave"], s["arrangement"], s["label"]): s["events"][0][2] for s in specs}
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


def test_tune_set_is_balanced_and_differs_only_in_mood_notes():
    specs = tune_specs()
    assert len(specs) == 144 and sum(s["label"] == "minor" for s in specs) == 72
    by = {(s["key"], s["octave"], s["arrangement"], s["label"]): [e[2][0] for e in s["events"]] for s in specs}
    for (key, octv, tune, q), notes in by.items():
        if q != "major":
            continue
        minor = by[(key, octv, tune, "minor")]
        assert all(m - n in (0, -1) for m, n in zip(minor, notes)) and any(m != n for m, n in zip(minor, notes))
    assert tune_notes(60, "minor", "melody") == [60, 62, 63, 67, 68, 67, 63, 60]
    assert all(e[0] + e[1] <= 4.0 for s in specs for e in s["events"])


def test_roughness_set_balanced_and_flutter_rate_is_the_only_class_difference():
    specs = roughness_specs()
    assert len(specs) == 144 and sum(s["label"] == "fear" for s in specs) == 72
    for s in specs:
        assert (s["flutter_hz"] >= 30) == (s["label"] == "fear")
    calm_notes = sorted(s["events"][0][2][0] for s in specs if s["label"] == "calm")
    fear_notes = sorted(s["events"][0][2][0] for s in specs if s["label"] == "fear")
    assert calm_notes == fear_notes  # same notes in both classes


def test_flutter_has_the_requested_rate():
    import numpy as np
    x = np.ones(SR * 2)
    y = flutter(x, 70.0)
    spec = np.abs(np.fft.rfft(y - y.mean()))
    assert abs(np.argmax(spec) * SR / len(y) - 70.0) < 1.0


def test_volume_set_same_levels_only_order_and_transition_differ():
    import numpy as np
    specs = volume_specs()
    assert len(specs) == 144 and sum(s["label"] == "fear" for s in specs) == 72
    for patterns in VOL_PATTERNS.values():
        assert all(sorted(p) == list(range(8)) for p in patterns)  # every pattern visits all 8 levels once
    for p in VOL_PATTERNS["fear"]:
        assert min(abs(a - b) for a, b in zip(p, p[1:])) >= 3
    calm = volume_envelope(VOL_PATTERNS["calm"][0], True, 48000)
    fear = volume_envelope(VOL_PATTERNS["fear"][0], False, 48000)
    assert np.isclose(20 * np.log10(calm.max() / calm.min()), 20 * np.log10(fear.max() / fear.min()))
    assert np.abs(np.diff(20 * np.log10(calm))).max() < 0.01  # glides
    assert np.abs(np.diff(20 * np.log10(fear))).max() > 5  # cuts
