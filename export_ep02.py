"""Chart data for Emotional Resonance EP02 (real-music decoding), from the saved readings.

Derived numbers only: no audio, no readings. Every figure the write-up already states is recomputed
here and checked against the stated value before anything is written (assert), so the charts cannot
drift from the text.

  .venv/bin/python export_ep02.py [--out runs/real-music/ep02-data.json] [--perm 1000]

Contents:
- luck: the full label-shuffle spread (seed 0, 1000 runs, best layer re-picked each run: the same
  call score_decoding.score_run makes), per check and model, as counts of clips right;
- loudness: loudness alone, each clip left out, nearest emotion-group average loudness wins;
- mixups: the five-way mix-up table at the primary layer (the ears);
- opposites: the comparison score for every pair, plus the random-group luck spread (opposites_luck
  method, seed 3, 2000 shuffles);
- clips: sample clips, listener ratings (1-9), whole-clip scores on the five arrows (each built
  without that clip) and the half-second timeline at the ears, for both models.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import score_decoding as sd

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs/real-music"
MODELS = {"qwen": "Qwen2-Audio", "af": "Audio Flamingo"}
CHECKS = {"happy_vs_sad": ("happy", "sad"), "happy_vs_fear": ("happy", "fear"),
          "tender_vs_sad": ("tender", "sad"), "all_five": tuple(sd.EMOTIONS)}
SAMPLE_CLIPS = ["set1_007", "set1_008", "set1_163"]
# Stated in the readout / write-up; the export refuses to run if a recomputation disagrees.
STATED = {
    "right": {"qwen": {"happy_vs_sad": 29, "happy_vs_fear": 31, "tender_vs_sad": 25, "all_five": 48},
              "af": {"happy_vs_sad": 26, "happy_vs_fear": 29, "tender_vs_sad": 24, "all_five": 50}},
    "loudness": {"happy_vs_sad": 30, "happy_vs_fear": 26, "tender_vs_sad": 19, "all_five": 32},
}


def load(model):
    man = json.loads((RUNS / f"{model}_film_decoding.manifest.json").read_text())
    labels = np.array([c["label"] for c in man["clips"]])
    ids = [c["clip_id"] for c in man["clips"]]
    names, mats = sd.layer_matrices(np.load(RUNS / f"{model}_film_decoding.npz"))
    return man, labels, ids, names, mats


def loudness_alone(rows, emotions):
    """Each clip left out: which emotion group's average loudness is it nearest? Returns clips right."""
    rows = [r for r in rows if r["label"] in emotions]
    right = 0
    for i, r in enumerate(rows):
        means = {e: np.mean([float(o["rms_db"]) for j, o in enumerate(rows) if j != i and o["label"] == e])
                 for e in emotions}
        right += min(means, key=lambda e: abs(means[e] - float(r["rms_db"]))) == r["label"]
    return right


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(RUNS / "ep02-data.json"))
    p.add_argument("--perm", type=int, default=1000)
    args = p.parse_args(argv)

    csv_rows = list(csv.DictReader(open(ROOT / "clips/film/decoding/clips.csv")))
    by_id = {r["clip_id"]: r for r in csv_rows}
    out = {"emotions": list(sd.EMOTIONS), "models": MODELS, "checks": {}, "luck": {}, "mixups": {},
           "opposites": {}, "clips": {}}

    loud = {}
    for chk, emos in CHECKS.items():
        loud[chk] = loudness_alone(csv_rows, emos)
        assert loud[chk] == STATED["loudness"][chk], (chk, loud[chk])
        n = 20 * len(emos)
        out["checks"][chk] = {"total": n, "guessing": n // len(emos), "loudness": loud[chk],
                              "mark": int(np.ceil(n * (sd.PASS_FIVE_WAY if chk == "all_five" else sd.PASS_TWO_WAY) - 1e-9))}

    for model in MODELS:
        man, labels, ids, names, mats = load(model)
        enc = mats[0]
        # Luck: the scorer's own label-shuffle call, kept whole instead of summarised.
        dist = sd.label_shuffle(mats, labels, "silence", args.perm, 0)
        out["luck"][model] = {}
        for chk, emos in CHECKS.items():
            n = out["checks"][chk]["total"]
            counts = np.rint(np.array(dist[chk]) * n).astype(int)
            right = sd.accuracy(enc, labels, "silence", emos)[0]
            assert right == STATED["right"][model][chk], (model, chk, right)
            out["luck"][model][chk] = {"right": right, "hist": np.bincount(counts, minlength=n + 1).tolist(),
                                       "typical": float(np.median(counts)),
                                       "top5": float(np.quantile(counts, 0.95)),
                                       "share_at_or_above": float((counts >= right).mean())}
        # Mix-ups at the ears.
        idx, S = sd.loo_scores(enc, labels, "silence")
        pick = np.array(sd.EMOTIONS)[S.argmax(1)]
        out["mixups"][model] = [[int(((labels[idx] == a) & (pick == b)).sum()) for b in sd.EMOTIONS]
                                for a in sd.EMOTIONS]
        # Opposites and their luck spread (opposites_luck.py method).
        out["opposites"][model] = {"pairs": sd.opposites(enc, labels)}
        rng = np.random.default_rng(3)
        emo = np.flatnonzero(np.isin(labels, sd.EMOTIONS))
        vals = []
        for _ in range(2000):
            sh = labels.copy()
            sh[emo] = rng.permutation(labels[emo])
            vals += list(sd.opposites(enc, sh).values())
        v = np.array(vals)
        out["opposites"][model]["luck"] = {"p01": float(np.quantile(v, .01)), "p50": float(np.median(v)),
                                           "p99": float(np.quantile(v, .99)), "min": float(v.min()),
                                           "max": float(v.max()),
                                           "hist_edges": [-0.2, 1.0], "hist": np.histogram(v, 48, (-0.2, 1.0))[0].tolist()}
        # Sample clips: whole-clip scores and the timeline at the ears.
        with np.load(RUNS / f"{model}_film_decoding.moments.npz") as m:
            W, wc = m["enc_windows"].astype(np.float64), m["window_clip"]
        seconds = man["moments_window_tokens"] / man["audio_tokens_per_second"]
        for cid in SAMPLE_CLIPS:
            i = ids.index(cid)
            rows = sd.timeline(W[wc == i], enc, labels, clip=i)
            whole = sd.timeline(enc[i:i + 1], enc, labels, clip=i)[0]
            if labels[i] in sd.EMOTIONS:      # same numbers the scorer used for the pick
                assert np.allclose(whole, S[list(idx).index(i)]), cid
            c = out["clips"].setdefault(cid, {
                "label": str(labels[i]) if labels[i] != "neutral" else "compound",
                "seconds": float(by_id[cid]["seconds"]),
                "listeners": [float(by_id[cid][e]) for e in sd.EMOTIONS], "models": {}})
            c["models"][model] = {"whole": [round(float(x), 3) for x in whole],
                                  "pick": sd.EMOTIONS[int(np.argmax(whole))],
                                  "window_seconds": seconds,
                                  "timeline": [[round(float(x), 3) for x in r] for r in rows]}
        print(model, "done")

    Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
