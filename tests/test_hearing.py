import numpy as np

from make_clips import clip_specs, chord_notes, tune_specs, tune_notes, roughness_specs, flutter, SR, volume_specs, volume_envelope, VOL_PATTERNS, tempo_specs, tempo_clip
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


def test_tempo_set_same_note_same_level_more_repeats_when_fast():
    import numpy as np
    specs = tempo_specs()
    assert len(specs) == 144 and sum(s["label"] == "fear" for s in specs) == 72
    slow, fast = tempo_clip(60, 60.0), tempo_clip(60, 200.0)
    assert len(slow) == len(fast)  # clip length fixed
    assert np.isclose(np.abs(slow).max(), np.abs(fast).max())  # same per-note level
    def count(x):  # notes = runs of 10 ms blocks with sound in them
        blocks = np.abs(x[: len(x) // 160 * 160]).reshape(-1, 160).max(axis=1) > 0.01
        return int(np.sum(np.diff(np.r_[0, blocks.astype(int)]) == 1))
    assert count(slow) == 3 and count(fast) == 10


def test_luck_is_the_coin_flip_chance_of_doing_this_well():
    keys = np.repeat(np.arange(12), 12)
    y = np.tile([True, False], 72)
    X = np.zeros((144, 3))
    X[y, 0] = 1.0
    perfect = leave_one_key_out(X, y, keys)
    assert perfect["correct"] == 144 and perfect["ranking"] == 1.0
    assert perfect["chance_of_doing_this_well_by_luck"] == 2.0 ** -144
    rng = np.random.default_rng(3)
    noise = leave_one_key_out(rng.standard_normal((144, 20)), y, keys)
    assert noise["chance_of_doing_this_well_by_luck"] > 0.01


def _hearing_run(write_run):
    """12 keys x 12 clips, fear/calm alternating, plus a silent clip. Encoder: perfect signal, far
    from silence. lm_0: pure noise."""
    rng = np.random.default_rng(8)
    y = np.tile([True, False], 72)
    enc = 10 + rng.standard_normal((144, 8)) * 0.1
    enc[y, 0] += 1.0
    enc = np.vstack([enc, np.zeros(8)])
    lm = rng.standard_normal((145, 1, 8))
    clips = [{"label": "fear" if v else "calm", "source": f"k{i // 12}", "recording": f"r{i}"} for i, v in enumerate(y)]
    clips.append({"label": "silence", "source": "silence", "recording": "silence"})
    return write_run(enc, lm, clips)


def test_hearing_cli_scores_every_layer_and_uses_the_given_pass_mark(write_run, tmp_path, monkeypatch):
    import json
    import sys
    import hearing_test
    prefix = _hearing_run(write_run)
    # main() sets the pass marks as module globals; monkeypatch puts them back after the test
    monkeypatch.setattr(hearing_test, "FAIL_BELOW", hearing_test.FAIL_BELOW)
    monkeypatch.setattr(hearing_test, "PASS_ABOVE", hearing_test.PASS_ABOVE)

    def score(*extra):
        out = tmp_path / "r.json"
        monkeypatch.setattr(sys, "argv", ["hearing_test.py", prefix, "--positive", "fear", "--negative", "calm",
                                          "--json", str(out), *extra])
        hearing_test.main()
        return json.loads(out.read_text())

    r = score("--fail-below", "0.75", "--pass-above", "0.9")
    assert list(r["layers"]) == ["encoder", "lm_0"] and r["primary"] == "encoder"
    assert r["repeat_check_identical"] is True
    enc = r["layers"]["encoder"]
    assert enc["sound"]["registers"] and enc["major_minor"]["n"] == 144  # the silent clip is not scored
    assert enc["major_minor"]["accuracy"] == 1.0 and enc["major_minor"]["verdict"] == "pass"
    assert r["layers"]["lm_0"]["major_minor"]["verdict"] == "fail"
    # the same perfect reading is only "usable" when the bar is set at 100%
    r = score("--pass-above", "1.0")
    assert r["layers"]["encoder"]["major_minor"]["verdict"] == "usable-more-clips"


def test_finish_sets_length_fade_and_loudness():
    from make_clips import finish, TARGET_RMS
    rng = np.random.default_rng(0)
    long = finish(rng.standard_normal(SR * 3) * 0.1, 2.0)
    short = finish(rng.standard_normal(SR) * 0.1, 2.0)
    for x in (long, short):
        assert len(x) == 2 * SR and x.dtype == np.float32
        assert np.isclose(np.sqrt(np.mean(x.astype(np.float64) ** 2)), TARGET_RMS, rtol=1e-4)
        assert x[-1] == 0.0  # the fade ends at silence
    assert np.all(short[SR + 1:] == 0)  # padded, not stretched
    import pytest
    with pytest.raises(ValueError, match="silence"):
        finish(np.zeros(SR), 2.0)


def test_midi_note_offs_come_before_ons_at_the_same_moment(tmp_path):
    import mido
    from make_clips import write_midi
    path = tmp_path / "t.mid"
    write_midi([(0.0, 0.5, [60]), (0.5, 0.5, [62, 65])], path, program=16)
    msgs, now = [], 0
    for m in mido.MidiFile(path).tracks[0]:
        now += m.time
        if m.type in ("note_on", "note_off", "program_change"):
            msgs.append((now, m.type, getattr(m, "note", getattr(m, "program", None))))
    assert msgs[0] == (0, "program_change", 16)
    assert msgs[1:] == [(0, "note_on", 60), (480, "note_off", 60), (480, "note_on", 62), (480, "note_on", 65),
                        (960, "note_off", 62), (960, "note_off", 65)]  # 960 ticks = 1 second


def _build_set(name, out, monkeypatch):
    import sys
    import make_clips
    monkeypatch.setattr(sys, "argv", ["make_clips.py", "--set", name, "--out", str(out)])
    make_clips.main()


def test_volume_set_writes_the_clip_list_and_regenerates_byte_identically(tmp_path, monkeypatch):
    import csv
    import soundfile as sf
    from make_clips import TARGET_RMS
    _build_set("volume", tmp_path / "a", monkeypatch)
    _build_set("volume", tmp_path / "b", monkeypatch)
    rows = list(csv.DictReader(open(tmp_path / "a" / "clips.csv")))
    assert list(rows[0]) == ["clip_id", "path", "label", "source", "recording", "key", "arrangement", "octave"]
    assert len(rows) == 145 and rows[-1]["label"] == "silence"
    assert sum(r["label"] == "fear" for r in rows) == 72
    for r in rows:
        x, sr = sf.read(r["path"])
        assert sr == SR and len(x) == 3 * SR
        rms = np.sqrt(np.mean(x ** 2))
        assert rms == 0 if r["label"] == "silence" else np.isclose(rms, TARGET_RMS, rtol=1e-3)
        assert (tmp_path / "a" / f"{r['clip_id']}.wav").read_bytes() == (tmp_path / "b" / f"{r['clip_id']}.wav").read_bytes()


def test_tempo_set_is_not_loudness_matched(tmp_path, monkeypatch):
    import csv
    import soundfile as sf
    _build_set("tempo", tmp_path, monkeypatch)
    rows = [r for r in csv.DictReader(open(tmp_path / "clips.csv")) if r["label"] != "silence"]
    rms = {lab: np.mean([np.sqrt(np.mean(sf.read(r["path"])[0] ** 2)) for r in rows if r["label"] == lab])
           for lab in ("calm", "fear")}
    assert rms["fear"] > 1.5 * rms["calm"]  # more notes, same per-note level: louder on average
