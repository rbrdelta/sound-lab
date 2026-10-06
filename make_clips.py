"""Build the controlled hearing-test clip set (design agreed with Daniel 2026-10-06).

What moves: chord quality (major vs minor) -- the thing being tested -- and, spread evenly across
both qualities, the key (12 roots), the arrangement (root position + two inversions) and the octave
(two adjacent mid-range octaves). 12 x 2 x 3 x 2 = 144 chord clips. Every key/arrangement/octave
combination exists once as major and once as minor, differing only in the middle note of the
triad (a half-step lower in minor).

Held fixed: instrument (General MIDI acoustic grand piano from the FluidR3_GM SoundFont, reverb
and chorus off), strike strength (velocity 80), length (2.0 s: held 1.5 s, then the natural
release, then a 50 ms fade to exactly 2.0 s), loudness (every clip scaled to the same RMS),
sample rate (16 kHz mono, what the model takes in).

Plus one silent clip of the same length, for the "does the sound register at all" check.

`--set tunes` (added 2026-10-06 after the chord set failed): the same design, but each clip is a
4.0 s, 8-note tune (0.5 s per note) instead of a held chord. Three tunes replace the three
arrangements: a short melody, the chord's notes one after another, and the first six notes of the
scale. Major and minor versions are note-for-note identical except the 3rd and 6th of the scale
(each a half-step lower in minor).

Writes <out>/<clip_id>.wav and <out>/clips.csv (clip_id,path,label,source,recording,
key,arrangement,octave). `source` = the key, so whole keys can be held out; `recording` = the clip.

Needs: fluidsynth (CLI) + a GM SoundFont (apt: fluidsynth fluid-soundfont-gm), mido, soundfile.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SR = 16_000
LENGTH_S = 2.0
HOLD_S = 1.5
FADE_S = 0.05
VELOCITY = 80
TARGET_RMS = 0.05
KEYS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
OCTAVE_BASES = {"oct3": 48, "oct4": 60}  # MIDI note of C3 / C4 (root of the C chord)
ARRANGEMENTS = ["root", "inv1", "inv2"]
THIRD = {"major": 4, "minor": 3}
DEFAULT_SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"


def chord_notes(root: int, quality: str, arrangement: str) -> list[int]:
    """MIDI notes of a triad. Major and minor differ only in the middle note of the triad."""
    r, t, f = root, root + THIRD[quality], root + 7
    return {"root": [r, t, f], "inv1": [t, f, r + 12], "inv2": [f, r + 12, t + 12]}[arrangement]


# scale degree -> semitones above the root; minor lowers the 3rd and 6th by a half-step
DEGREE = {"major": {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 8: 12},
          "minor": {1: 0, 2: 2, 3: 3, 4: 5, 5: 7, 6: 8, 8: 12}}
TUNES = {"melody": [1, 2, 3, 5, 6, 5, 3, 1],
         "arpeggio": [1, 3, 5, 8, 8, 5, 3, 1],
         "scale": [1, 2, 3, 4, 5, 6, 5, 4]}
TUNE_NOTE_S = 0.5
TUNE_LENGTH_S = 4.0


def tune_notes(root: int, quality: str, tune: str) -> list[int]:
    return [root + DEGREE[quality][d] for d in TUNES[tune]]


def tune_specs() -> list[dict]:
    specs = []
    for k, key in enumerate(KEYS):
        for octave, base in OCTAVE_BASES.items():
            for tune in TUNES:
                for quality in ("major", "minor"):
                    cid = f"{key}_{octave}_{tune}_{quality}"
                    notes = tune_notes(base + k, quality, tune)
                    events = [(i * TUNE_NOTE_S, TUNE_NOTE_S, [n]) for i, n in enumerate(notes)]
                    specs.append({"clip_id": cid, "label": quality, "source": key, "recording": cid,
                                  "key": key, "arrangement": tune, "octave": octave, "events": events})
    return specs


# Roughness set (Daniel 2026-10-06): one sustained organ note whose loudness flutters slowly
# (calm) or fast (fear: the scream "roughness" range, 30-150 flutters a second). Only the flutter
# rate differs between the classes; three rates per class are spread evenly so the detector can't
# key on one exact rate. Notes: 12 pitch classes x 2 octaves. Graded on held-out pitch classes.
ORGAN = 16  # General MIDI drawbar organ: a held note stays steady (no natural decay)
ROUGH_LENGTH_S = 3.0
FLUTTER_DEPTH = 0.8  # loudness swings between 20% and 180% of its average
FLUTTER_HZ = {"calm": [3.0, 4.5, 6.0], "fear": [40.0, 70.0, 100.0]}


def roughness_specs() -> list[dict]:
    specs = []
    for k, key in enumerate(KEYS):
        for octave, base in OCTAVE_BASES.items():
            for label, rates in FLUTTER_HZ.items():
                for i, hz in enumerate(rates):
                    cid = f"{key}_{octave}_rate{i}_{label}"
                    specs.append({"clip_id": cid, "label": label, "source": key, "recording": cid,
                                  "key": key, "arrangement": f"{hz:g}Hz", "octave": octave,
                                  "events": [(0.0, ROUGH_LENGTH_S, [base + k])], "program": ORGAN,
                                  "flutter_hz": hz})
    return specs


def flutter(audio: np.ndarray, hz: float) -> np.ndarray:
    t = np.arange(len(audio)) / SR
    return audio * (1.0 + FLUTTER_DEPTH * np.sin(2 * np.pi * hz * t))


def clip_specs() -> list[dict]:
    specs = []
    for k, key in enumerate(KEYS):
        for octave, base in OCTAVE_BASES.items():
            for arr in ARRANGEMENTS:
                for quality in ("major", "minor"):
                    cid = f"{key}_{octave}_{arr}_{quality}"
                    specs.append({"clip_id": cid, "label": quality, "source": key, "recording": cid,
                                  "key": key, "arrangement": arr, "octave": octave,
                                  "events": [(0.0, HOLD_S, chord_notes(base + k, quality, arr))]})
    return specs


def write_midi(events: list[tuple], path: Path, program: int = 0) -> None:
    """events: (start_s, duration_s, [midi notes]). Default tempo -> 960 ticks per second."""
    import mido

    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.Message("program_change", program=program, time=0))  # 0 = acoustic grand piano
    msgs = []  # (tick, order, message): offs sort before ons at the same tick
    for start, dur, notes in events:
        on, off = int(round(start * 960)), int(round((start + dur) * 960))
        for n in notes:
            msgs.append((on, 1, mido.Message("note_on", note=n, velocity=VELOCITY)))
            msgs.append((off, 0, mido.Message("note_off", note=n, velocity=0)))
    now = 0
    for tick, _, msg in sorted(msgs, key=lambda m: (m[0], m[1])):
        tr.append(msg.copy(time=tick - now))
        now = tick
    mid.save(path)


def render(events: list[tuple], sf2: str, program: int = 0) -> np.ndarray:
    import soundfile as sf

    with tempfile.TemporaryDirectory() as td:
        mpath, wpath = Path(td) / "c.mid", Path(td) / "c.wav"
        write_midi(events, mpath, program)
        subprocess.run(["fluidsynth", "-ni", "-q", "-R", "0", "-C", "0", "-g", "1.0", "-r", str(SR),
                        "-F", str(wpath), sf2, str(mpath)], check=True, capture_output=True)
        audio, sr = sf.read(wpath, dtype="float64", always_2d=True)
    assert sr == SR, sr
    return audio.mean(axis=1)


def finish(audio: np.ndarray, length_s: float = LENGTH_S) -> np.ndarray:
    """Trim/pad to exactly length_s, fade the tail, scale to TARGET_RMS."""
    n = int(length_s * SR)
    out = np.zeros(n)
    out[: min(n, len(audio))] = audio[:n]
    fade = int(FADE_S * SR)
    out[-fade:] *= np.linspace(1.0, 0.0, fade)
    rms = np.sqrt(np.mean(out ** 2))
    if rms == 0:
        raise ValueError("rendered silence -- check fluidsynth / SoundFont")
    out *= TARGET_RMS / rms
    assert np.max(np.abs(out)) < 1.0, "clipping after loudness match"
    return out.astype(np.float32)


def main() -> None:
    import soundfile as sf

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="clips/controlled")
    p.add_argument("--sf2", default=DEFAULT_SF2)
    p.add_argument("--set", default="chords", choices=["chords", "tunes", "roughness"])
    args = p.parse_args()
    specs, length = {"chords": (clip_specs(), LENGTH_S), "tunes": (tune_specs(), TUNE_LENGTH_S),
                     "roughness": (roughness_specs(), ROUGH_LENGTH_S)}[args.set]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in specs:
        path = out / f"{s['clip_id']}.wav"
        audio = render(s["events"], args.sf2, s.get("program", 0))
        if "flutter_hz" in s:
            audio = flutter(audio[: int(length * SR)], s["flutter_hz"])
        sf.write(path, finish(audio, length), SR)
        rows.append({**{k: s[k] for k in ("clip_id", "label", "source", "recording", "key",
                                          "arrangement", "octave")}, "path": str(path)})
    silent = out / "silence.wav"
    sf.write(silent, np.zeros(int(length * SR), dtype=np.float32), SR)
    rows.append({"clip_id": "silence", "label": "silence", "source": "silence", "recording": "silence",
                 "key": "", "arrangement": "", "octave": "", "path": str(silent)})

    cols = ["clip_id", "path", "label", "source", "recording", "key", "arrangement", "octave"]
    with open(out / "clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} clips ({len(rows) - 1} chords + 1 silence) and {out / 'clips.csv'}")


if __name__ == "__main__":
    main()
