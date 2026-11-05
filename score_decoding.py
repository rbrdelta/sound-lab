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
  7. With the moment-by-moment file: which clips have a clear primary emotion on the whole-clip
     reading (the neutral bar), which show a wave inside the clip, which are blurred, and whether
     the takeaway positions beat the whole-clip average (the "after the climax" prediction).

Decisions the registration left open are listed in DECISIONS below and written into the output.
Daniel reviewed them on 2026-11-05, before any result was read.

Usage:
  python score_decoding.py runs/real-music/qwen_film_decoding [--clips clips/film/decoding/clips.csv]
      [--perm 1000] [--seed 0]
  python score_decoding.py --compare runs/real-music/qwen_film_decoding runs/real-music/af_film_decoding
Writes <prefix>.score.json. Reads <prefix>.moments.npz too when it exists. numpy only.
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
WAVE_BAR_SHARE = 0.05     # a wave's peak must rise above all but the top 5% of neutral moments
WAVE_SHOULDER = 0.25      # the windows beside the peak must be at least this far up it
SOUND_CHECK_SHARE = 0.95  # silence is outside the clips if this share of them lie past it
CLIMAX_WIN_CLIPS = 10     # takeaway "more clearly" = at least this many more clips right

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
    "Timeline windows are centred on silence's whole-clip average (the same reference as the "
    "directions); silence's own windows are not used. A clip's windows are always measured on "
    "arrows built without that clip: with the clip inside its own arrow, made-up data scored 70% "
    "instead of 35%.",
    "Takeaway positions are scored as their own set of readings: one average per clip over the "
    "positions after the audio, then the same arrows, stripping and five-way test as the whole-clip "
    "readings, silence's takeaway included. They are word positions, so measuring them on arrows "
    "built from sound positions gives every clip the same offset; on made-up data one emotion then "
    "took 58% of the picks.",
    "'After the climax' prediction (Daniel, 2026-11-05): the takeaway test beats the whole-clip "
    "test if it gets at least 10 more of the 100 clips right, best layer against best layer, read "
    "per model. For Audio Flamingo the takeaway positions are only the chat template's closing "
    "tokens, so its test is weaker.",
    "Clear primary, whole clip (Daniel, 2026-11-05: 'it has to be greater than the neutral'): a "
    "clip is clear if its gap between the top emotion and the runner-up is bigger than the biggest "
    "such gap among the 15 neutral clips.",
    "Wave (Daniel, 2026-11-05: 'a clear building up and a drop off', like 1, 2, 6, 3, 0): plain "
    "back-to-back windows, no overlap. An emotion's rise at a window = its score minus its usual "
    "level in that clip (the clip's median), minus what the five share at that moment (their "
    "median). A wave's peak rises above the top 5% of the neutral "
    "clips' rises on that arrow; the windows either side are at least a quarter of the way up it, "
    "so a lone spike does not count; a peak in the clip's first or last window does not count, "
    "because the build or the fall is cut off. A swell in all five together cancels out, because "
    "that points to energy; an explicit 'all five clear their bars' test was tried first and threw "
    "out genuine waves, because the arrows share a common part and a true wave lifts all five. "
    "Waves may overlap ('You can feel multiple "
    "things'). Rise, not the top-two gap, because two waves together shrink that gap. Rise, not raw "
    "score, because stripping puts the neutral clips' average at zero on every arrow, which would "
    "make the bar too easy for emotion clips. The quarter is Claude's number.",
    "Blurred clip: the whole-clip reading is not clear and the clip has at least one wave. Across "
    "clips the question is whether emotion clips show waves clearly more often than neutral clips "
    "(Daniel: 'the only test should be is there a clear pattern of it'). This replaces the plan's "
    "'every clip', which the neutral bar makes unreachable: about 1 clip in 16 clears it by luck. "
    "Only 15 neutral clips, so the comparison is rough.",
    "Cancelling what the five share is weak against loudness: arrows can point opposite ways along "
    "loudness, so a swell can raise anger and lower tenderness with no emotion involved. The "
    "waveform under the timeline is where a wave that merely follows a swell gets seen.",
    "Two models (Daniel, 2026-11-05): each is read on its own, then compared. Only both clear and "
    "disagreeing counts as a conflict; it stops nothing and feeds a hypothesis about which model "
    "suits the next step.",
    "Sound check, changed from section 11 by Daniel on 2026-11-05 before any reading was opened: "
    "silence must fall outside the spread of the clips. Measured along the line from silence to "
    "the average of the other clips, at least 95% of clips must lie past silence. The section 11 "
    "wording failed on made-up clips that clearly carried sound, and so does plain distance from "
    "the clips' average, because with thousands of numbers per reading clips sit far from their "
    "own average. Reported per layer; nothing stops automatically.",
    "Opposites check has a known pull toward zero: the music arrow is estimated from only 15 "
    "neutral clips, so stripping it leaves a little leftover music in every direction, shared by "
    "all five, which makes opposite pairs read less opposite. On synthetic data, a planted "
    "exactly-opposite pair read about -0.4 instead of about -0.8. A clearly negative score is "
    "still evidence for opposites; a score near zero is weaker evidence against than it looks. "
    "Kept as built (Daniel, 2026-11-05). Removing the average of the five directions was rejected: "
    "it forces the typical comparison to about -0.25, the 'apart by construction' trap in section 11.",
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
    """Does silence fall outside the spread of the clips? Along the line from silence to the
    average of the other clips, how many clips lie past silence? (Pure noise around silence: half.)"""
    clips = X[np.isin(labels, EMOTIONS)]
    s = reference_mean(X, labels, "silence")
    n = len(clips)
    line = unit((clips.sum(0) - clips) / (n - 1) - s)      # one line per clip, built without it
    along = np.einsum("ij,ij->i", clips - s, line)
    share = float((along > 0).mean())
    return {"share_past_silence": share, "outside_spread": share >= SOUND_CHECK_SHARE,
            "typical_distance_along_line": float(np.median(along))}


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

def timeline(rows: np.ndarray, X_whole: np.ndarray, labels: np.ndarray, ref="silence", clip=None):
    """Project each row (a half-second window) of ONE layer onto the five directions.
    rows: [n_rows, d] for one clip at one layer; X_whole: [n_clips, d] whole-clip readings at the
    same layer (to build the directions). clip: that clip's row in X_whole, so its arrows are built
    without it; always pass it when the rows come from a clip in X_whole. Returns [n_rows, 5]."""
    keep = np.arange(len(labels)) != clip if clip is not None else np.ones(len(labels), bool)
    r = reference_mean(X_whole[keep], labels[keep], ref)
    U = unit(directions(X_whole[keep], labels[keep], ref))
    return (np.asarray(rows, dtype=np.float64) - r) @ U.T


def margin(scores: np.ndarray) -> np.ndarray:
    """Top emotion's score minus the second's, per row: how clear the primary is (no verdict)."""
    s = np.sort(np.atleast_2d(scores), axis=1)
    return s[:, -1] - s[:, -2]


def whole_clip_clarity(X, labels, ref="silence"):
    """The neutral bar on the whole-clip reading. Returns (bar, {clip row: (pick, gap, clear)}) for
    the emotion clips; each clip is measured on arrows built without it."""
    gap = {}
    for i in np.flatnonzero(np.isin(labels, EMOTIONS + ("neutral",))):
        gap[int(i)] = timeline(X[i:i + 1], X, labels, ref, clip=i)[0]
    bar = max(float(margin(gap[i])[0]) for i in gap if labels[i] == "neutral")
    out = {}
    for i, s in gap.items():
        if labels[i] != "neutral":
            g = float(margin(s)[0])
            out[i] = (EMOTIONS[int(s.argmax())], g, g > bar)
    return bar, out


def moment_scores(W, window_clip, X, labels, ref="silence"):
    """{clip row: [n_windows, 5]} for the emotion and neutral clips at one layer."""
    return {int(i): timeline(W[window_clip == i], X, labels, ref, clip=i)
            for i in np.flatnonzero(np.isin(labels, EMOTIONS + ("neutral",)))}


def rises(scores: np.ndarray) -> np.ndarray:
    """Each emotion's score above its usual level in this clip (the clip's median), minus what the
    five share at that moment (their median), so a swell in all five counts for none of them."""
    r = scores - np.median(scores, axis=0)
    return r - np.median(r, axis=1, keepdims=True)


def wave_bars(neutral_scores: list[np.ndarray]) -> np.ndarray:
    """Per emotion: the rise that only the top 5% of neutral moments reach."""
    return np.quantile(np.vstack([rises(s) for s in neutral_scores]), 1 - WAVE_BAR_SHARE, axis=0)


def find_waves(scores: np.ndarray, bars: np.ndarray) -> list[dict]:
    """Waves in one clip: a build and a drop around a peak that clears the neutral bar."""
    r, waves = rises(scores), []
    for k, emotion in enumerate(EMOTIONS):
        above = np.flatnonzero(r[:, k] > bars[k])
        for run in np.split(above, np.flatnonzero(np.diff(above) > 1) + 1) if len(above) else []:
            peak = int(run[np.argmax(r[run, k])])
            if peak == 0 or peak == len(r) - 1:
                continue
            floor = WAVE_SHOULDER * r[peak, k]
            if r[peak - 1, k] >= floor and r[peak + 1, k] >= floor:
                waves.append({"emotion": emotion, "peak_window": peak, "rise": float(r[peak, k])})
    return waves


def group_tail(k: int, n1: int, K: int, N: int) -> float:
    """If K of N clips have something, how often would a group of n1 hold k or more of them by luck?"""
    return float(sum(math.comb(K, j) * math.comb(N - K, n1 - j) for j in range(k, min(K, n1) + 1))
                 / math.comb(N, n1))


def wave_report(W, window_clip, X, labels, ref="silence"):
    """The blur question at one layer: waves in the emotion clips against the neutral clips."""
    scores = moment_scores(W, window_clip, X, labels, ref)
    bars = wave_bars([s for i, s in scores.items() if labels[i] == "neutral"])
    waves = {i: find_waves(s, bars) for i, s in scores.items()}
    bar, whole = whole_clip_clarity(X, labels, ref)
    emo = [i for i in waves if labels[i] != "neutral"]
    neu = [i for i in waves if labels[i] == "neutral"]
    with_e, with_n = sum(bool(waves[i]) for i in emo), sum(bool(waves[i]) for i in neu)
    on_chosen = sum(w["emotion"] == labels[i] for i in emo for w in waves[i])
    total = sum(len(waves[i]) for i in emo)
    return {"neutral_bar_whole_clip": bar, "wave_bars": dict(zip(EMOTIONS, map(float, bars))),
            "clear_clips": sum(c for _, _, c in whole.values()), "emotion_clips": len(emo),
            "emotion_clips_with_a_wave": with_e, "neutral_clips": len(neu),
            "neutral_clips_with_a_wave": with_n,
            "this_split_or_stronger_by_luck": group_tail(with_e, len(emo), with_e + with_n, len(emo) + len(neu)),
            "blurred_clips": sum(bool(waves[i]) and not whole[i][2] for i in emo),
            "waves_in_emotion_clips": total, "waves_on_the_chosen_emotion": on_chosen}, whole, waves


def takeaway_matrices(tail, tail_clip, n_clips):
    """One reading per clip per language layer: the average of the positions after the audio."""
    names = [f"lm_{i}" for i in range(tail.shape[1])]
    per_clip = [tail[tail_clip == c].astype(np.float64).mean(0) for c in range(n_clips)]
    return names, list(np.stack(per_clip, axis=1))


def picks_right(X, labels, ref="silence"):
    """{clip row: was the five-way answer right}, each clip left out of its own direction."""
    idx, S = loo_scores(X, labels, ref)
    return dict(zip(idx.tolist(), (np.array(EMOTIONS)[S.argmax(1)] == labels[idx]).tolist()))


def climax_check(names, mats, t_names, t_mats, labels, ref="silence"):
    """'After the climax': best takeaway layer against best whole-clip layer, same clips."""
    def best(ns, ms):
        right = [picks_right(X, labels, ref) for X in ms]
        p = int(np.argmax([sum(r.values()) for r in right]))
        return ns[p], right[p]
    w_layer, w = best(names, mats)
    t_layer, t = best(t_names, t_mats)
    w_n, t_n = sum(w.values()), sum(t.values())
    return {"whole_clip_layer": w_layer, "whole_clip_right": w_n, "takeaway_layer": t_layer,
            "takeaway_right": t_n, "total": len(w),
            "right_only_on_takeaway": sum(t[i] and not w[i] for i in w),
            "right_only_on_whole_clip": sum(w[i] and not t[i] for i in w),
            "win_at": CLIMAX_WIN_CLIPS, "takeaway_wins": t_n - w_n >= CLIMAX_WIN_CLIPS}


def compare_models(a: dict, b: dict) -> dict:
    """Two score files: where are both models clear, and do they agree?"""
    ca, cb = ({c["clip_id"]: c for c in r["clips"]} for r in (a, b))
    both = [i for i in ca if i in cb and ca[i]["clear"] and cb[i]["clear"]]
    clash = [i for i in both if ca[i]["pick"] != cb[i]["pick"]]
    return {"both_clear": len(both), "both_clear_and_agree": len(both) - len(clash),
            "conflicts": [{"clip_id": i, a["model"]: ca[i]["pick"], b["model"]: cb[i]["pick"],
                           "listeners_chose": ca[i]["label"]} for i in clash]}


def add_moments(res, names, mats, labels, ids, moments_path):
    """Everything that needs the moment-by-moment file, at the primary layer."""
    p = names.index(res["primary"]["layer"])
    with np.load(moments_path) as m:
        t_names, t_mats = takeaway_matrices(m["lm_tail"], m["tail_clip"], len(labels))
        res["climax"] = climax_check(names, mats, t_names, t_mats, labels)
        del t_mats
        wc = m["window_clip"]
        W = (m["enc_windows"] if p == 0 else m["lm_windows"][:, p - 1]).astype(np.float64)
    res["blur"], whole, waves = wave_report(W, wc, mats[p], labels)
    res["clips"] = [{"clip_id": ids[i], "label": str(labels[i]), "pick": pick, "gap": gap,
                     "clear": bool(clear), "waves": waves[i]} for i, (pick, gap, clear) in whole.items()]


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
    sc = res["sound_check"][pr["layer"]]
    lines.append(f"Sound check at that layer: {sc['share_past_silence']:.0%} of clips lie past silence "
                 f"(silence is outside their spread at {SOUND_CHECK_SHARE:.0%}).")
    if "climax" in res:
        c = res["climax"]
        lines.append(f"After the climax: takeaway {c['takeaway_right']}/{c['total']} right at "
                     f"{c['takeaway_layer']}, whole clip {c['whole_clip_right']}/{c['total']} at "
                     f"{c['whole_clip_layer']}; a win needs {c['win_at']} more clips.")
    if "blur" in res:
        b = res["blur"]
        lines.append(f"Clear on the whole clip: {b['clear_clips']}/{b['emotion_clips']}. Clips with a "
                     f"wave: {b['emotion_clips_with_a_wave']}/{b['emotion_clips']} emotion, "
                     f"{b['neutral_clips_with_a_wave']}/{b['neutral_clips']} neutral. Blurred: "
                     f"{b['blurred_clips']}/{b['emotion_clips']}.")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prefix", nargs="?", help="run prefix: reads <prefix>.npz and <prefix>.manifest.json")
    p.add_argument("--compare", nargs=2, metavar="PREFIX",
                   help="compare two scored runs: where both models are clear, do they agree?")
    p.add_argument("--clips", default="clips/film/decoding/clips.csv",
                   help="clip list with listeners_top, for the clear-clips check ('' to skip)")
    p.add_argument("--perm", type=int, default=1000, help="label-shuffle runs (0 to skip)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.compare:
        a, b = (json.loads(Path(x + ".score.json").read_text()) for x in args.compare)
        print(json.dumps(compare_models(a, b), indent=2))
        return
    if not args.prefix:
        p.error("give a run prefix, or --compare with two")

    prefix = Path(args.prefix)
    man = json.loads(prefix.with_suffix(".manifest.json").read_text())
    labels = np.array([c["label"] for c in man["clips"]])
    ids = [c["clip_id"] for c in man["clips"]]
    clear = load_clear(args.clips, ids, labels)
    names, mats = layer_matrices(np.load(prefix.with_suffix(".npz")))
    res = score_run(names, mats, labels, clear, args.perm, args.seed)
    res.update({"run": str(prefix), "model": man.get("model"), "seed": args.seed})
    moments = prefix.with_suffix(".moments.npz")
    if moments.exists():
        add_moments(res, names, mats, labels, ids, moments)
    out = prefix.with_name(prefix.name + ".score.json")
    out.write_text(json.dumps(res, indent=2))
    print(plain_summary(res))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
