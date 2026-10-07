"""Build the Scratch Paper Viewer from saved readings: python viewer/build.py (from sound-lab/).

x = each clip's leave-one-key-out detector score (the graded score, cutoff at 0), standardised per
layer. Each row is one clip (Daniel, 2026-10-06: an unlabelled vertical axis implies meaning it
doesn't have): rows grouped by group label, then sorted by octave, key and variant.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe import load_run, fit

RUNS = [("Major vs minor — chords", "hearing_controlled", "Qwen2-Audio", "minor", "major", (2/3, 0.8)),
        ("Major vs minor — chords", "af_chords", "Audio Flamingo", "minor", "major", (2/3, 0.8)),
        ("Major vs minor — tunes", "hearing_tunes", "Qwen2-Audio", "minor", "major", (2/3, 0.8)),
        ("Major vs minor — tunes", "af_tunes", "Audio Flamingo", "minor", "major", (2/3, 0.8)),
        ("Roughness", "qwen_roughness", "Qwen2-Audio", "fear", "calm", (0.75, 0.9)),
        ("Roughness", "af_roughness", "Audio Flamingo", "fear", "calm", (0.75, 0.9)),
        ("Volume", "qwen_volume", "Qwen2-Audio", "fear", "calm", (2/3, 0.8)),
        ("Volume", "af_volume", "Audio Flamingo", "fear", "calm", (2/3, 0.8)),
        ("Tempo", "qwen_tempo", "Qwen2-Audio", "fear", "calm", (0.75, 0.9)),
        ("Tempo", "af_tempo", "Audio Flamingo", "fear", "calm", (0.75, 0.9))]

out = []
for test, run, model, pos, neg, (lo, hi) in RUNS:
    feats, labels, _, sources, man = load_run(f"runs/{run}")
    labels = np.array(labels)
    keep = np.isin(labels, [pos, neg])
    src = np.array(sources)[keep]
    y = labels[keep] == pos
    ids = list(np.array([c["clip_id"] for c in man["clips"]])[keep])
    res = json.load(open(f"runs/{run}.result.json"))["layers"]
    names = ["encoder"] + sorted([k for k in feats if k.startswith("lm_")], key=lambda s: int(s[3:]))
    layers = []
    for name in names:
        X = np.asarray(feats[name], dtype=np.float64)[keep]
        s = np.zeros(len(y))
        for k in sorted(set(src)):
            t = src == k
            d = fit(X[~t & y], X[~t & ~y])
            s[t] = d.scores(X[t]) - d.cutoff
        s = s / (np.std(s) or 1)
        layers.append({"name": "ears" if name == "encoder" else f"layer {int(name[3:]) + 1}",
                       "acc": round(res[name]["major_minor"]["accuracy"], 3),
                       "x": [round(v, 2) for v in s]})
    NAMES = {"Roughness": ("fast flutter", "slow flutter"), "Volume": ("jumping loudness", "gradual loudness"),
             "Tempo": ("fast tempo", "relaxed tempo")}
    pos_name, neg_name = NAMES.get(test, (pos, neg))
    KEYS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    def sortkey(i):
        p = ids[i].split("_")
        return (not y[i], p[1], KEYS.index(p[0]), p[2])
    order = sorted(range(len(ids)), key=sortkey)
    row = [0] * len(ids)
    for r_, i in enumerate(order):
        row[i] = r_
    out.append({"test": test, "model": model, "row": row, "pos": pos_name, "neg": neg_name, "lo": lo, "hi": hi,
                "ids": ids, "isPos": [int(v) for v in y], "layers": layers})

data = json.dumps(out, separators=(",", ":"))
walk = open("viewer/walkthrough.json").read()
page = open("viewer/template.html").read().replace("__DATA__", data).replace("__WALK__", walk)
open("viewer/scratch-paper-viewer.html", "w").write(page)
print("wrote viewer/scratch-paper-viewer.html")
