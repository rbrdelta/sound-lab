"""The fear detector from primer/probe-by-hand.md, run on every layer of real scratch work.

Per layer:
  direction = mean(positive clips) - mean(negative clips)      (worksheet Part A step 3)
  score     = clip . direction                                  (dot product)
  cutoff    = halfway between the two averages' scores          (Part A step 5)
  accuracy  = fraction of unseen clips put on the right side of the cutoff
  ranking   = fraction of (positive, negative) unseen pairs where the positive clip scores
              higher (ties count half) -- does the direction order clips correctly even if the
              cutoff is in the wrong place?

Splits are by recording, never by row: clips cut from the same recording never sit on both
sides. Optionally, whole sources (collections / recording sessions) are held out entirely as a
cross-source test -- the check that catches the worksheet's loudness trap (Part C).

Pass mark agreed 2026-10-05 (on unseen clips):
  below 2/3        -> fail: drop the model
  2/3 up to 0.80   -> usable, with the clip count raised (see simulate_clips.py)
  above 0.80       -> pass: use as is

Usage:
  python probe.py runs/qwen_hearing --positive fear --negative calm \
      [--test-fraction 0.3] [--cross-sources sessionB,sessionC] [--seed 0]

numpy only; no torch needed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

FAIL_BELOW = 2 / 3
PASS_ABOVE = 0.80


# ---------------------------------------------------------------------------------------------
# The detector itself
# ---------------------------------------------------------------------------------------------

@dataclass
class Detector:
    direction: np.ndarray
    cutoff: float

    def scores(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.direction

    def predict(self, X: np.ndarray) -> np.ndarray:
        """True = positive. Exactly-at-cutoff counts as negative (worksheet: 'above' = fear)."""
        return self.scores(X) > self.cutoff


def fit(X_pos: np.ndarray, X_neg: np.ndarray) -> Detector:
    mu_p = np.asarray(X_pos, dtype=np.float64).mean(0)
    mu_n = np.asarray(X_neg, dtype=np.float64).mean(0)
    d = mu_p - mu_n
    cutoff = 0.5 * (mu_p @ d + mu_n @ d)
    return Detector(direction=d, cutoff=float(cutoff))


def accuracy(det: Detector, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(det.predict(X) == np.asarray(y, dtype=bool)))


def ranking_score(det: Detector, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of positive/negative pairs ranked correctly (= area under the ROC curve)."""
    s = det.scores(X)
    y = np.asarray(y, dtype=bool)
    sp, sn = s[y], s[~y]
    if len(sp) == 0 or len(sn) == 0:
        return float("nan")
    diff = sp[:, None] - sn[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def verdict(acc: float) -> str:
    if np.isnan(acc):
        return "n/a"
    if acc < FAIL_BELOW:
        return "fail"
    if acc <= PASS_ABOVE:
        return "usable-more-clips"
    return "pass"


# ---------------------------------------------------------------------------------------------
# Splitting by recording / source
# ---------------------------------------------------------------------------------------------

def split_by_group(groups, test_fraction: float, seed: int, labels=None):
    """Return a boolean test mask. Whole groups go to test; deterministic for a given seed.

    If labels are given, groups are drawn separately within each label so both classes appear
    in the test set (a group that carries both labels is assigned by its first label).
    """
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq = sorted(set(groups.tolist()))
    if labels is None:
        strata = {None: uniq}
    else:
        labels = np.asarray(labels)
        strata: dict = {}
        for g in uniq:
            strata.setdefault(labels[groups == g][0], []).append(g)
    test_groups = set()
    for key in sorted(strata, key=str):
        gs = list(strata[key])
        rng.shuffle(gs)
        k = max(1, int(round(test_fraction * len(gs)))) if len(gs) > 1 else 0
        test_groups.update(gs[:k])
    return np.isin(groups, list(test_groups))


@dataclass
class LayerResult:
    layer: str
    heldout_accuracy: float
    heldout_ranking: float
    heldout_verdict: str
    cross_accuracy: float
    cross_ranking: float
    cross_verdict: str
    train_accuracy: float


def evaluate_layer(name, X, y, train, test, cross) -> LayerResult:
    det = fit(X[train & y], X[train & ~y])
    nan = float("nan")
    ha = accuracy(det, X[test], y[test]) if test.any() else nan
    hr = ranking_score(det, X[test], y[test]) if test.any() else nan
    ca = accuracy(det, X[cross], y[cross]) if cross.any() else nan
    cr = ranking_score(det, X[cross], y[cross]) if cross.any() else nan
    return LayerResult(name, ha, hr, verdict(ha), ca, cr, verdict(ca), accuracy(det, X[train], y[train]))


def run(features: dict, labels, recordings, sources, positive, negative,
        test_fraction=0.3, cross_sources=(), seed=0):
    """features: {layer_name: array [n_clips, d]}. Returns (list[LayerResult], split info)."""
    labels = np.asarray(labels)
    keep = np.isin(labels, [positive, negative])
    y = labels == positive
    sources = np.asarray(sources)
    cross = keep & np.isin(sources, list(cross_sources))
    pool = keep & ~cross
    rec = np.asarray(recordings)
    test = np.zeros(len(labels), dtype=bool)
    test[pool] = split_by_group(rec[pool], test_fraction, seed, labels[pool])
    train = pool & ~test
    if not (train & y).any() or not (train & ~y).any():
        raise ValueError("training split lacks one of the two labels")
    results = [evaluate_layer(name, np.asarray(X), y, train, test, cross) for name, X in features.items()]
    info = {"n_train": int(train.sum()), "n_test": int(test.sum()), "n_cross": int(cross.sum()),
            "train_recordings": sorted(set(rec[train].tolist())),
            "test_recordings": sorted(set(rec[test].tolist())),
            "cross_sources": sorted(set(cross_sources))}
    return results, info


def load_run(prefix: str):
    prefix = Path(prefix)
    data = np.load(prefix.with_suffix(".npz"))
    man = json.loads(prefix.with_suffix(".manifest.json").read_text())
    feats = {"encoder": data["encoder"]}
    for i in range(data["lm"].shape[1]):
        feats[f"lm_{i}"] = data["lm"][:, i, :]
    clips = man["clips"]
    return feats, [c["label"] for c in clips], [c["recording"] for c in clips], [c["source"] for c in clips], man


def _fmt(x):
    return "  -  " if np.isnan(x) else f"{x:.2f}"


def report(results, info) -> str:
    lines = [
        f"Built the detector on {info['n_train']} clips; tested on {info['n_test']} unseen clips from "
        f"different recordings" + (f" and {info['n_cross']} clips from sources it never saw "
                                   f"({', '.join(info['cross_sources'])})." if info["n_cross"] else "."),
        "Accuracy = share of unseen clips put on the right side of the cutoff. Ranking = share of "
        "fear/calm pairs where the fear clip scored higher (0.50 = coin flip).",
        f"Pass mark: below {FAIL_BELOW:.2f} fail, {FAIL_BELOW:.2f}-{PASS_ABOVE:.2f} usable with more clips, "
        f"above {PASS_ABOVE:.2f} pass.",
        "",
        f"{'layer':<9} {'train':>5} | {'unseen acc':>10} {'rank':>5} {'verdict':<18} | "
        f"{'other-source acc':>16} {'rank':>5} {'verdict':<18}",
    ]
    for r in results:
        lines.append(f"{r.layer:<9} {_fmt(r.train_accuracy):>5} | {_fmt(r.heldout_accuracy):>10} "
                     f"{_fmt(r.heldout_ranking):>5} {r.heldout_verdict:<18} | {_fmt(r.cross_accuracy):>16} "
                     f"{_fmt(r.cross_ranking):>5} {r.cross_verdict:<18}")
    lines += ["", "Caution: picking the best of many layers after looking flatters it. Treat a single "
              "standout layer as a lead to confirm on fresh clips, not as the result."]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="prefix written by extract.py (without .npz)")
    p.add_argument("--positive", required=True, help="label treated as positive (e.g. fear)")
    p.add_argument("--negative", required=True, help="label treated as negative (e.g. calm)")
    p.add_argument("--test-fraction", type=float, default=0.3, help="share of recordings held out")
    p.add_argument("--cross-sources", default="", help="comma-separated sources held out entirely")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="also write per-layer results here")
    a = p.parse_args(argv)
    feats, labels, recs, srcs, _ = load_run(a.run)
    cross = [s for s in a.cross_sources.split(",") if s]
    results, info = run(feats, labels, recs, srcs, a.positive, a.negative, a.test_fraction, cross, a.seed)
    print(report(results, info))
    if a.json:
        Path(a.json).write_text(json.dumps({"split": info, "layers": [asdict(r) for r in results]}, indent=2))


if __name__ == "__main__":
    main()
