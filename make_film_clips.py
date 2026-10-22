"""Build the clip lists for the real-music decoding step and the main experiment.

Source: the film-soundtrack set (Eerola & Vuoskoski 2011, OSF p6vkg). The audio is kept private
(clips/ is gitignored); only this script and the ratings-derived summary are public.

Two clip groups, no clip in both:
  - decoding: Set 1 clips whose intended emotion is anger, fear, happy, sad or tender, minus any
    Set 1 clip that is the same excerpt as a main-experiment clip (Set 2's `link` column), then an
    equal random subset per emotion, in random order.
  - main: the 25 Set 2 clips in the five *_HIGH groups (5 per emotion).

Also checks every main clip's `link` against the audio (loudness-envelope correlation), and
measures each clip's loudness from the file, so "loudness posing as emotion" can be checked.

Writes clips/film/decoding/clips.csv and clips/film/main/clips.csv
(clip_id,path,label,source,recording + ratings + rms_db). Both end with a silence row; the decoding
list also carries the other references: a flat tone and the rated-neutral clips (label "neutral").

Usage: python make_film_clips.py [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import random
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RAW = Path("clips/film/raw")
OUT = Path("clips/film")
EMOTIONS = ["anger", "fear", "happy", "sad", "tender"]
SR = 16_000


def load(path: Path) -> np.ndarray:
    x, sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1)
    g = gcd(sr, SR)
    return resample_poly(x, SR // g, sr // g)


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def envelope(x: np.ndarray, hop: int = 800) -> np.ndarray:
    n = len(x) // hop
    return np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))


def same_excerpt(a: np.ndarray, b: np.ndarray) -> float:
    """Best correlation of loudness envelopes over offsets up to ~7.5 s (Set 2 excerpts were re-cut
    from Set 1 at different start points, so the alignment shift can be several seconds)."""
    ea, eb = envelope(a), envelope(b)
    best = -1.0
    for lag in range(-150, 151):
        x, y = (ea[lag:], eb) if lag >= 0 else (ea, eb[-lag:])
        n = min(len(x), len(y))
        if n > 60:
            best = max(best, float(np.corrcoef(x[:n], y[:n])[0, 1]))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clear", type=int, default=15)
    ap.add_argument("--unclear", type=int, default=5)
    args = ap.parse_args()

    s1 = list(csv.DictReader(open(RAW / "mean_ratings_set1.csv", encoding="utf-8-sig")))
    s2 = list(csv.DictReader(open(RAW / "mean_ratings_set2.csv", encoding="utf-8-sig")))

    main_rows = [r for r in s2 if r["TARGET"].lower().removesuffix("_high") in EMOTIONS
                 and r["TARGET"].endswith("_HIGH")]
    assert len(main_rows) == 25, len(main_rows)
    main_links = {r["link"].zfill(3) for r in main_rows}

    # The audio must agree with the published link before the link is trusted for exclusion.
    print("link check (main clip -> Set 1 clip, envelope correlation):")
    worst = 1.0
    for r in main_rows:
        c = same_excerpt(load(RAW / "set2" / f"{r['Number'].zfill(3)}.mp3"),
                         load(RAW / "Set1" / f"{r['link'].zfill(3)}.mp3"))
        worst = min(worst, c)
        if c < 0.8:  # confirmed 2026-10-21: every linked clip is the best match among all 360 by a wide margin (>=0.85 vs <=0.68)
            print(f"  WARN set2 {r['Number']} vs set1 {r['link']}: {c:.3f}")
    print(f"  lowest of 25: {worst:.3f}")

    pool = {e: [r for r in s1 if r["TARGET"].lower() == e and r["number"].zfill(3) not in main_links]
            for e in EMOTIONS}
    # Daniel 2026-10-21: 20 per emotion, "mostly clear, but some unclear". Clear = listeners rated
    # the chosen emotion highest. 15 clear + 5 unclear per emotion (split: Claude).
    def clear(r):
        return max(EMOTIONS, key=lambda k: float(r[k])) == r["TARGET"].lower()
    rng = random.Random(args.seed)
    dec = []
    for e in EMOTIONS:
        ranked = sorted(pool[e], key=lambda r: r["number"])
        dec += rng.sample([r for r in ranked if clear(r)], args.clear)
        dec += rng.sample([r for r in ranked if not clear(r)], args.unclear)
    rng.shuffle(dec)

    def row(set_name, num, label, ratings, path):
        x = load(path)
        out = {"clip_id": f"{set_name}_{num}", "path": str(path), "label": label,
               "source": f"film_{set_name}", "recording": f"set1_{ratings['link']}" if "link" in ratings
               else f"set1_{num}", "rms_db": round(rms_db(x), 2), "seconds": round(len(x) / SR, 2)}
        for k in ["valence", "energy", "tension", *EMOTIONS]:
            out[k] = ratings.get(k, ratings.get(k.capitalize()))
        # The emotion listeners rated highest; "obvious" clips are the ones where this equals the label.
        out["listeners_top"] = max(EMOTIONS, key=lambda e: float(out[e]))
        return out

    silence = OUT / "silence.wav"
    sf.write(silence, np.zeros(SR * 15, dtype=np.float32), SR)
    sil = {"clip_id": "silence", "path": str(silence), "label": "silence", "source": "silence",
           "recording": "silence"}

    # References for building the emotion directions (Daniel 2026-10-21: silence, a flat tone and
    # neutral clips, checked against each other). Flat tone: steady 220 Hz sine, no rhythm, no
    # loudness change, 15 s, at the decoding clips' median loudness (pitch and level: Claude).
    # Neutral clips (Claude's rule, delegated): listeners rated all five emotions <= 4 of 9 and
    # pleasantness, energy and tension each 3.5-6.5; never a decoding or main clip.
    def is_neutral(r):
        mid = all(3.5 <= float(r[k]) <= 6.5 for k in ("valence", "energy", "tension"))
        return mid and max(float(r[e]) for e in EMOTIONS) <= 4.0
    used = {r["number"].zfill(3) for r in dec} | main_links
    neutral = [r for r in s1 if r["number"].zfill(3) not in used and is_neutral(r)]

    dec_rows = [row("set1", r["number"].zfill(3), r["TARGET"].lower(), r,
                    RAW / "Set1" / f"{r['number'].zfill(3)}.mp3") for r in dec]
    neu_rows = [row("set1", r["number"].zfill(3), "neutral", r,
                    RAW / "Set1" / f"{r['number'].zfill(3)}.mp3") for r in neutral]
    level = 10 ** (np.median([r["rms_db"] for r in dec_rows]) / 20)
    t = np.arange(SR * 15) / SR
    tone_path = OUT / "flat_tone.wav"
    sf.write(tone_path, (level * np.sqrt(2) * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR)
    tone = {"clip_id": "flat_tone", "path": str(tone_path), "label": "flat_tone", "source": "flat_tone",
            "recording": "flat_tone", "rms_db": round(rms_db(load(tone_path)), 2), "seconds": 15.0}

    groups = {
        "decoding": dec_rows,
        "main": [row("set2", r["Number"].zfill(3), r["TARGET"].lower().removesuffix("_high"), r,
                     RAW / "set2" / f"{r['Number'].zfill(3)}.mp3") for r in main_rows],
    }
    for name, rows in groups.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        cols = list(rows[0].keys())  # reference rows fill the columns they have
        with open(d / "clips.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, restval="")
            w.writeheader()
            refs = [sil] + ([tone] + neu_rows if name == "decoding" else [])
            w.writerows(rows + refs)
        print(f"\n{name}: {len(rows)} clips + {len(refs)} reference rows -> {d / 'clips.csv'}")
        print(f"  {'emotion':8} {'n':>3} {'own rating':>10} {'energy':>7} {'valence':>8} {'loudness dB':>12}")
        for e in EMOTIONS:
            g = [r for r in rows if r["label"] == e]
            m = lambda k: np.mean([float(r[k]) for r in g])
            print(f"  {e:8} {len(g):>3} {m(e):>10.2f} {m('energy'):>7.2f} {m('valence'):>8.2f} {m('rms_db'):>12.1f}")

    print(f"\ndecoding pool per emotion after removing main-clip excerpts: "
          f"{ {e: len(v) for e, v in pool.items()} } -> {args.clear} clear + {args.unclear} unclear each")


if __name__ == "__main__":
    main()
