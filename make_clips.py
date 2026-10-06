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


def clip_specs() -> list[dict]:
    specs = []
    for k, key in enumerate(KEYS):
        for octave, base in OCTAVE_BASES.items():
            for arr in ARRANGEMENTS:
                for quality in ("major", "minor"):
                    cid = f"{key}_{octave}_{arr}_{quality}"
                    specs.append({"clip_id": cid, "label": quality, "source": key, "recording": cid,
                                  "key": key, "arrangement": arr, "octave": octave,
                                  "notes": chord_notes(base + k, quality, arr)})
    return specs


def write_midi(notes: list[int], path: Path) -> None:
    import mido

    mid = mido.MidiFile(ticks_per_beat=480)  # default tempo 500000 us/beat -> 960 ticks per second
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.Message("program_change", program=0, time=0))  # acoustic grand piano
    for n in notes:
        tr.append(mido.Message("note_on", note=n, velocity=VELOCITY, time=0))
    hold_ticks = int(round(HOLD_S * 960))
    for i, n in enumerate(notes):
        tr.append(mido.Message("note_off", note=n, velocity=0, time=hold_ticks if i == 0 else 0))
    mid.save(path)


def render(notes: list[int], sf2: str) -> np.ndarray:
    import soundfile as sf

    with tempfile.TemporaryDirectory() as td:
        mpath, wpath = Path(td) / "c.mid", Path(td) / "c.wav"
        write_midi(notes, mpath)
        subprocess.run(["fluidsynth", "-ni", "-q", "-R", "0", "-C", "0", "-g", "1.0", "-r", str(SR),
                        "-F", str(wpath), sf2, str(mpath)], check=True, capture_output=True)
        audio, sr = sf.read(wpath, dtype="float64", always_2d=True)
    assert sr == SR, sr
    return audio.mean(axis=1)


def finish(audio: np.ndarray) -> np.ndarray:
    """Trim/pad to exactly LENGTH_S, fade the tail, scale to TARGET_RMS."""
    n = int(LENGTH_S * SR)
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
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in clip_specs():
        path = out / f"{s['clip_id']}.wav"
        sf.write(path, finish(render(s["notes"], args.sf2)), SR)
        rows.append({**{k: s[k] for k in ("clip_id", "label", "source", "recording", "key",
                                          "arrangement", "octave")}, "path": str(path)})
    silent = out / "silence.wav"
    sf.write(silent, np.zeros(int(LENGTH_S * SR), dtype=np.float32), SR)
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
