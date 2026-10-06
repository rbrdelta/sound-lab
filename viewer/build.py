"""Build the Scratch Paper Viewer from saved readings: python viewer/build.py (from sound-lab/).

x = each clip's leave-one-key-out detector score (the graded score, cutoff at 0); y = the layer's
biggest unlabelled variation (first principal component). Both standardised per layer.
"""
import json
import numpy as np
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
        Xc = X - X.mean(0)
        pc = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][0]
        pc = pc / (np.std(pc) or 1)
        layers.append({"name": "ears" if name == "encoder" else f"layer {int(name[3:]) + 1}",
                       "acc": round(res[name]["major_minor"]["accuracy"], 3),
                       "x": [round(v, 2) for v in s], "y": [round(v, 2) for v in pc]})
    out.append({"test": test, "model": model, "pos": pos, "neg": neg, "lo": lo, "hi": hi,
                "ids": ids, "isPos": [int(v) for v in y], "layers": layers})

data = json.dumps(out, separators=(",", ":"))
page = open("viewer/template.html").read().replace("__DATA__", data)
open("viewer/scratch-paper-viewer.html", "w").write(page)
print("wrote viewer/scratch-paper-viewer.html")
