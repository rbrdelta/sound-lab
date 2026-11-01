"""Score the real-music decoding step (REAL-MUSIC-DECODING-REGISTRATION.md, sections 6-11).

Plain version of what this does, per layer (the ears, then every language layer):
  1. Each emotion's direction = average reading of that emotion's clips minus the average reading
     of a reference (silence is primary; the flat tone and the 15 neutral clips are the checks).
  2. The "this is music" arrow = average of the 15 neutral clips minus silence. It is STRIPPED from
     each emotion direction: we remove exactly as much of that arrow as the direction contains
     (projection), and keep everything else. Not a straight subtraction.
  3. Every emotion clip is scored on all five directions, each built WITHOUT that clip. Its score
     on a direction = how far (clip minus the reference average) lies along the direction, with
     the direction scaled to length 1. The longest one is the model's answer; right if it matches
     the emotion the clip was chosen for.
  4. Checks in the plan's order: happiness vs sadness, happiness vs fear, tenderness vs sadness,
     then all five; the 75 clear clips are also reported on their own.
  5. The primary layer = the layer with the best all-five score. Picking the best of many layers
     can win by luck, so the luck level is measured the same way: shuffle the emotion labels, pick
     the best layer again, repeat (label-shuffle test).
  6. The three references are compared with each other, the sound check is run, and the five
     directions are compared pairwise (opposites and clusters).

Decisions the registration left open are listed in DECISIONS below and written into the output,
so Daniel reviews them before any result is read.

Usage:
  python score_decoding.py runs/real-music/qwen_film_decoding [--clips clips/film/decoding/clips.csv]
      [--perm 1000] [--seed 0]
Writes <prefix>.score.json. numpy only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

EMOTIONS = ("anger", "fear", "happy", "sad", "tender")
PAIR_CHECKS = (("happy", "sad"), ("happy", "fear"), ("tender", "sad"))  # plan section 8, steps 1-3
PASS_TWO_WAY = 2 / 3      # plan section 9
PASS_FIVE_WAY = 0.55      # plan section 9
REFERENCES = ("silence", "flat_tone", "neutral")

DECISIONS = [
    "Scoring centres each clip on the reference average before measuring it along a direction: "
    "score = (clip - reference average) . unit(direction). Section 7 says directions are built "
    "against the reference but not how a clip is measured; centring on the same reference is the "
    "natural reading. Alternative: centre on the average of all emotion clips (closer to the paper).",
    "Directions are scaled to length 1 before measuring, so a direction built from louder or more "
    "varied clips does not win just by being longer.",
    "Neutral reference: its directions are NOT stripped. Stripping the music arrow from "
    "(emotion - neutral) gives exactly the stripped silence direction (the arrow from neutral is "
    "the silence direction minus the music arrow, and stripping removes the music arrow entirely), "
    "so the comparison could never show a difference. Neutral therefore stands for 'subtract the "
    "music part', silence for 'strip it'; comparing them is the check Daniel asked for.",
    "Flat-tone reference: directions ARE stripped of the same music arrow (neutral minus silence), "
    "following 'removed from every direction' in section 7.",
    "The 75 clear clips are scored with directions built from all 100 decoding clips (each clip "
    "still left out of its own direction); only the reported accuracy is restricted to the 75. "
    "Alternative: build the directions from the 75 alone.",
    "Ties for the best layer go to the earliest layer.",
    "Two-way checks use only the two emotions' clips and directions, at the primary layer chosen "
    "on all five. Their luck level comes from the same label-shuffle runs: in each shuffle the "
    "layer is re-picked on all five, and the two-way score is read at that layer.",
    "'How often chance alone would produce this score' is given two ways: the coin-flip count "
    "for a fixed layer, and the label-shuffle count, which includes the luck of picking the best "
    "layer. The second is the honest one for the primary layer.",
    "Timeline windows and takeaway positions are centred on silence's whole-clip average (the "
    "same reference as the directions); silence's own windows are not used.",
    "No threshold for 'grayscale' (no clear primary) is set here: the plan's blur rule names the "
    "idea but not a number. The timeline reports the margin between the top two emotions; Daniel "
    "sets the reading rule before results are read.",
    "Sound check, as written in section 11, passes only when the jump from silence to the clips is "
    "larger than the clips' spread among themselves (mean distance silence-to-clip vs mean "
    "distance clip-to-clip). Found on synthetic data: clips that clearly carry a shared sound "
    "component can still fail it if they vary a lot from each other. Reported as a ratio, no stop "
    "is applied automatically; Daniel decides how to read it.",
    "Opposites check has a known pull toward zero: the music arrow is estimated from only 15 "
    "neutral clips, so stripping it leaves a little leftover music in every direction, shared by "
    "all five, which makes opposite pairs read less opposite. On synthetic data, a planted "
    "exactly-opposite pair read about -0.4 instead of about -0.8. A clearly negative score is "
    "still evidence for opposites; a score near zero is weaker evidence against than it looks. "
    "Not corrected here; options for Daniel: also report the comparison with the average of the "
    "five directions removed (the paper's approach), or estimate the leftover from the data.",
]


# --------------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------------

def unit(v: np.ndarray) -> np.ndarray:
    """Scale to length 1 along the last axis (zero vectors stay zero)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v, dtype=float), where=n > 0)


def strip(v: np.ndarray, arrow: np.ndarray) -> np.ndarray:
    """Remove exactly as much of `arrow` as each row of v contains (projection), keep the rest."""
    a = unit(arrow)
    return v - np.outer(v @ a, a).reshape(v.shape) if v.ndim > 1 else v - (v @ a) * a


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Comparison score: -1 exactly opposite, 0 unrelated, +1 same direction."""
    return float(unit(a) @ unit(b))


def reference_mean(X: np.ndarray, labels: np.ndarray, ref: str) -> np.ndarray:
    m = labels == ref
    if not m.any():
        raise ValueError(f"no clips labelled {ref!r}")
    return X[m].mean(0)


def music_arrow(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Average of the neutral clips minus silence's average."""
    return reference_mean(X, labels, "neutral") - reference_mean(X, labels, "silence")


def directions(X, labels, ref, emotions=EMOTIONS, members=None):
    """Full directions (no clip left out) for each emotion against `ref`, stripped or not per DECISIONS.

    members: optional boolean mask of which emotion clips count (default all). Returns [n_emo, d]."""
    r = reference_mean(X, labels, ref)
    keep = np.ones(len(labels), bool) if members is None else members
    D = np.stack([X[(labels == e) & keep].mean(0) - r for e in emotions])
    if ref != "neutral":
        D = strip(D, music_arrow(X, labels))
    return D


def loo_scores(X, labels, ref, emotions=EMOTIONS, members=None):
    """Score every emotion clip on every direction, each built without that clip.

    Only the clip's own emotion direction changes when it is left out (the reference and the music
    arrow use no emotion clips), so the own-class column is recomputed per clip and the others use
    the full directions. Returns (idx [n], S [n, n_emo]) for the clips in `emotions` (and members)."""
    keep = np.ones(len(labels), bool) if members is None else members
    idx = np.flatnonzero(np.isin(labels, emotions) & keep)
    r = reference_mean(X, labels, ref)
    arrow = music_arrow(X, labels) if ref != "neutral" else None
    sums = {e: X[(labels == e) & keep].sum(0) for e in emotions}
    counts = {e: int(((labels == e) & keep).sum()) for e in emotions}
    D = np.stack([sums[e] / counts[e] - r for e in emotions])
    if arrow is not None:
        D = strip(D, arrow)
    U = unit(D)
    C = X[idx] - r                       # clips centred on the reference
    S = C @ U.T
    own = np.array([emotions.index(labels[i]) for i in idx])
    for k, e in enumerate(emotions):
        rows = np.flatnonzero(own == k)
        if not len(rows):
            continue
        if counts[e] < 2:
            raise ValueError(f"need at least 2 {e!r} clips to leave one out")
        Dk = (sums[e] - X[idx[rows]]) / (counts[e] - 1) - r      # [n_k, d], one per left-out clip
        if arrow is not None:
            Dk = strip(Dk, arrow)
        S[rows, k] = np.einsum("ij,ij->i", C[rows], unit(Dk))
    return idx, S


def accuracy(X, labels, ref, emotions=EMOTIONS, members=None, report=None):
    """Fraction right (and counts) when the strongest direction is the answer.

    report: optional boolean mask restricting which scored clips are counted (e.g. the clear 75)."""
    idx, S = loo_scores(X, labels, ref, emotions, members)
    pred = np.array(emotions)[S.argmax(1)]
    right = pred == labels[idx]
    if report is not None:
        right = right[report[idx]]
    return int(right.sum()), int(right.size)


def binom_tail(k: int, n: int, p: float) -> float:
    """How often pure guessing at rate p gets k or more right out of n."""
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


# --------------------------------------------------------------------------------------------
# The full scoring pass
# --------------------------------------------------------------------------------------------

def layer_matrices(npz) -> tuple[list[str], list[np.ndarray]]:
    enc, lm = npz["encoder"].astype(np.float64), npz["lm"].astype(np.float64)
    names = ["encoder"] + [f"lm_{i}" for i in range(lm.shape[1])]
    return names, [enc] + [lm[:, i] for i in range(lm.shape[1])]


def checks_at(X, labels, ref, clear=None):
    """All the plan's checks at one layer: (right, total) per check."""
    out = {f"{a}_vs_{b}": accuracy(X, labels, ref, (a, b)) for a, b in PAIR_CHECKS}
    out["all_five"] = accuracy(X, labels, ref)
    if clear is not None:
        out["all_five_clear"] = accuracy(X, labels, ref, report=clear)
    return out


def best_layer(per_layer_all_five: list[tuple[int, int]]) -> int:
    acc = [r / n for r, n in per_layer_all_five]
    return int(np.argmax(acc))       # argmax returns the first (earliest) of any tie


def label_shuffle(mats, labels, ref, n_perm, seed, clear=None):
    """Luck level for the primary-layer scores: shuffle labels among the emotion clips, re-pick
    the best layer on all five, record every check at that layer. Returns {check: [fractions]}."""
    rng = np.random.default_rng(seed)
    emo = np.flatnonzero(np.isin(labels, EMOTIONS))
    dist: dict[str, list[float]] = {}
    for _ in range(n_perm):
        shuffled = labels.copy()
        shuffled[emo] = rng.permutation(labels[emo])
        all5 = [accuracy(X, shuffled, ref) for X in mats]
        X = mats[best_layer(all5)]
        for name, (r, n) in checks_at(X, shuffled, ref, clear).items():
            dist.setdefault(name, []).append(r / n)
    return dist


def sound_check(X, labels):
    """Is silence farther from the clips than the clips are from each other?"""
    clips = X[np.isin(labels, EMOTIONS)]
    s = reference_mean(X, labels, "silence")
    to_silence = float(np.linalg.norm(clips - s, axis=1).mean())
    sq = (clips ** 2).sum(1)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * clips @ clips.T, 0))  # pairwise distances
    np.fill_diagonal(d, 0)
    n = len(clips)
    between = float(d.sum() / (n * (n - 1)))
    return {"silence_to_clips": to_silence, "clips_to_each_other": between,
            "ratio": to_silence / between if between else float("inf")}


def compare_references(X, labels):
    """Per emotion: comparison score between the directions built from each pair of references."""
    D = {ref: directions(X, labels, ref) for ref in REFERENCES}
    out = {}
    for a, b in (("silence", "neutral"), ("silence", "flat_tone"), ("neutral", "flat_tone")):
        out[f"{a}_vs_{b}"] = {e: cosine(D[a][k], D[b][k]) for k, e in enumerate(EMOTIONS)}
    return out


def opposites(X, labels, ref="silence"):
    D = directions(X, labels, ref)
    return {f"{a}_vs_{b}": cosine(D[i], D[j])
            for i, a in enumerate(EMOTIONS) for j, b in enumerate(EMOTIONS) if i < j}


def score_run(names, mats, labels, clear=None, n_perm=1000, seed=0):
    result = {"decisions": DECISIONS, "layers": names,
              "pass_marks": {"two_way": PASS_TWO_WAY, "all_five": PASS_FIVE_WAY},
              "by_reference": {}}
    for ref in REFERENCES:
        per_layer = [checks_at(X, labels, ref, clear) for X in mats]
        result["by_reference"][ref] = {
            name: {chk: {"right": r, "total": n, "fraction": r / n} for chk, (r, n) in c.items()}
            for name, c in zip(names, per_layer)}
    # Primary: silence reference, layer chosen on all five.
    sil = result["by_reference"]["silence"]
    all5 = [(sil[n]["all_five"]["right"], sil[n]["all_five"]["total"]) for n in names]
    p = best_layer(all5)
    primary = names[p]
    chance = {chk: (0.5 if chk.count("_vs_") else 0.2) for chk in sil[primary]}
    summary = {}
    dist = label_shuffle(mats, labels, "silence", n_perm, seed, clear) if n_perm else {}
    for chk, v in sil[primary].items():
        mark = PASS_FIVE_WAY if chk.startswith("all_five") else PASS_TWO_WAY
        entry = {**v, "luck_rate": chance[chk], "pass_mark": mark,
                 "at_or_above_mark": v["fraction"] >= mark,
                 "guessing_gets_this_or_more": binom_tail(v["right"], v["total"], chance[chk])}
        if chk in dist:
            d = np.array(dist[chk])
            entry["label_shuffle"] = {
                "runs": len(d), "typical": float(np.median(d)), "top_5_percent": float(np.quantile(d, 0.95)),
                "share_at_or_above_observed": float((d >= v["fraction"]).mean())}
        summary[chk] = entry
    result["primary"] = {"reference": "silence", "layer": primary, "checks": summary}
    result["sound_check"] = {n: sound_check(X, labels) for n, X in zip(names, mats)}
    result["references_compared"] = {n: compare_references(X, labels) for n, X in zip(names, mats)}
    result["opposites_at_primary_layer"] = opposites(mats[p], labels)
    return result


# --------------------------------------------------------------------------------------------
# Moment by moment (timeline) -- the viewer is built separately
# --------------------------------------------------------------------------------------------

def timeline(rows: np.ndarray, X_whole: np.ndarray, labels: np.ndarray, ref="silence"):
    """Project each row (a half-second window, or a takeaway position) of ONE layer onto the five
    full directions. rows: [n_rows, d] for one clip at one layer; X_whole: [n_clips, d] whole-clip
    readings at the same layer (to build the directions). Returns [n_rows, 5]."""
    r = reference_mean(X_whole, labels, ref)
    U = unit(directions(X_whole, labels, ref))
    return (np.asarray(rows, dtype=np.float64) - r) @ U.T


def margin(scores: np.ndarray) -> np.ndarray:
    """Top emotion's score minus the second's, per row: how clear the primary is (no verdict)."""
    s = np.sort(np.atleast_2d(scores), axis=1)
    return s[:, -1] - s[:, -2]


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------

def load_clear(clips_csv: str | None, clip_ids: list[str], labels: np.ndarray):
    """Clear = listeners rated the clip's chosen emotion highest (listeners_top == label)."""
    if not clips_csv:
        return None
    top = {r["clip_id"]: r.get("listeners_top", "") for r in csv.DictReader(open(clips_csv))}
    return np.array([top.get(i, "") == lab for i, lab in zip(clip_ids, labels)])


def plain_summary(res: dict) -> str:
    pr = res["primary"]
    lines = [f"Primary layer (silence reference, best all-five score): {pr['layer']}"]
    names = {"happy_vs_sad": "Happiness vs sadness", "happy_vs_fear": "Happiness vs fear",
             "tender_vs_sad": "Tenderness vs sadness", "all_five": "All five",
             "all_five_clear": "All five, clear clips only"}
    for chk, v in pr["checks"].items():
        line = (f"  {names.get(chk, chk)}: {v['right']}/{v['total']} right ({v['fraction']:.0%}); "
                f"guessing alone gets {v['luck_rate']:.0%}; committed mark {v['pass_mark']:.0%}.")
        if "label_shuffle" in v:
            ls = v["label_shuffle"]
            line += (f" With labels shuffled and the best layer re-picked, a typical score is "
                     f"{ls['typical']:.0%}; {ls['share_at_or_above_observed']:.1%} of "
                     f"{ls['runs']} shuffles did this well or better.")
        lines.append(line)
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prefix", help="run prefix: reads <prefix>.npz and <prefix>.manifest.json")
    p.add_argument("--clips", default="clips/film/decoding/clips.csv",
                   help="clip list with listeners_top, for the clear-clips check ('' to skip)")
    p.add_argument("--perm", type=int, default=1000, help="label-shuffle runs (0 to skip)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    prefix = Path(args.prefix)
    man = json.loads(prefix.with_suffix(".manifest.json").read_text())
    labels = np.array([c["label"] for c in man["clips"]])
    ids = [c["clip_id"] for c in man["clips"]]
    clear = load_clear(args.clips, ids, labels)
    names, mats = layer_matrices(np.load(prefix.with_suffix(".npz")))
    res = score_run(names, mats, labels, clear, args.perm, args.seed)
    res.update({"run": str(prefix), "model": man.get("model"), "seed": args.seed})
    out = prefix.with_name(prefix.name + ".score.json")
    out.write_text(json.dumps(res, indent=2))
    print(plain_summary(res))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
