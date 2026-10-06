"""Score the controlled hearing test: can the model's internal numbers tell major from minor?

Two checks, both on the readings written by extract.py:

1. Does the sound register at all? For each layer: every chord's distance from the silent clip,
   compared with the typical distance between two chords. If chords sit no farther from silence
   than from each other, the sound isn't getting in -- stop before reading major vs minor.

2. Major vs minor on keys the detector never practised on. Leave-one-key-out: for each of the 12
   keys, build the detector (probe.fit: average major minus average minor) on the other 11 keys
   and grade it on the held-out key's 12 clips. Every clip is graded exactly once, by a detector
   that never saw its key. Pooled accuracy over all 144 clips, against the agreed pass mark.

Primary reading (declared before the run, 2026-10-06): the ENCODER (the ears). The 32 language-
model layers are reported as a profile; a single standout layer is a lead to confirm on fresh
clips, not the result (picking the best of 33 after looking flatters it).

Usage: python hearing_test.py runs/hearing_controlled [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from math import comb

import numpy as np

from probe import fit, ranking_score, load_run
import probe

FAIL_BELOW, PASS_ABOVE = probe.FAIL_BELOW, probe.PASS_ABOVE


def verdict(acc: float) -> str:
    if acc < FAIL_BELOW:
        return "fail"
    return "usable-more-clips" if acc <= PASS_ABOVE else "pass"

PRIMARY = "encoder"


def registers(X: np.ndarray, is_silence: np.ndarray) -> dict:
    """Chord-to-silence distances vs chord-to-chord distances (Euclidean)."""
    s = X[is_silence][0]
    C = X[~is_silence].astype(np.float64)
    to_sil = np.linalg.norm(C - s, axis=1)
    diffs = C[:, None, :] - C[None, :, :]
    cc = np.linalg.norm(diffs, axis=2)[np.triu_indices(len(C), 1)]
    return {"min_chord_to_silence": float(to_sil.min()), "median_chord_to_chord": float(np.median(cc)),
            "registers": bool(to_sil.min() > np.median(cc))}


def leave_one_key_out(X: np.ndarray, y: np.ndarray, keys: np.ndarray) -> dict:
    pred = np.zeros(len(y), dtype=bool)
    scores = np.zeros(len(y))
    for k in sorted(set(keys.tolist())):
        test = keys == k
        det = fit(X[~test & y], X[~test & ~y])
        pred[test] = det.predict(X[test])
        # scores from different folds aren't on one scale; centre each fold on its own cutoff
        scores[test] = det.scores(X[test]) - det.cutoff
    correct = int((pred == y).sum())
    n = len(y)
    acc = correct / n
    luck = sum(comb(n, i) for i in range(correct, n + 1)) / 2 ** n  # chance a coin-flipper does this well
    rank = float(np.mean([(s_p > s_n) + 0.5 * (s_p == s_n) for s_p in scores[y] for s_n in scores[~y]]))
    return {"n": n, "correct": correct, "accuracy": acc, "ranking": rank, "verdict": verdict(acc),
            "chance_of_doing_this_well_by_luck": luck}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--json")
    p.add_argument("--positive", default="minor", help="label scored as positive")
    p.add_argument("--negative", default="major")
    p.add_argument("--fail-below", type=float, default=probe.FAIL_BELOW)
    p.add_argument("--pass-above", type=float, default=probe.PASS_ABOVE)
    args = p.parse_args()
    global FAIL_BELOW, PASS_ABOVE
    FAIL_BELOW, PASS_ABOVE = args.fail_below, args.pass_above
    feats, labels, _, sources, man = load_run(args.run)
    labels, sources = np.asarray(labels), np.asarray(sources)
    is_sil = labels == "silence"
    chord = np.isin(labels, [args.positive, args.negative])
    y = labels[chord] == args.positive

    out = {"model": man["model"], "prompt": man["prompt"], "primary": PRIMARY,
           "repeat_check_identical": man.get("repeat_check_identical"), "layers": {}}
    for name, X in feats.items():
        X = np.asarray(X, dtype=np.float64)
        out["layers"][name] = {"sound": registers(X, is_sil),
                               "major_minor": leave_one_key_out(X[chord], y, sources[chord])}

    print(f"model {man['model']} | prompt {man['prompt']!r} | repeat check identical: "
          f"{man.get('repeat_check_identical')}")
    print(f"pass mark: below {FAIL_BELOW:.2f} fail, {FAIL_BELOW:.2f}-{PASS_ABOVE:.2f} usable with more "
          f"clips, above {PASS_ABOVE:.2f} pass. Primary reading: {PRIMARY}.\n")
    print(f"{'layer':<9} {'sound in?':>9} | {'held-out-key acc':>16} {'rank':>5} {'verdict':<18} {'luck':>8}")
    for name, r in out["layers"].items():
        mm = r["major_minor"]
        tag = " <- primary" if name == PRIMARY else ""
        print(f"{name:<9} {'yes' if r['sound']['registers'] else 'NO':>9} | {mm['accuracy']:>16.3f} "
              f"{mm['ranking']:>5.2f} {mm['verdict']:<18} {mm['chance_of_doing_this_well_by_luck']:>8.1e}{tag}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
