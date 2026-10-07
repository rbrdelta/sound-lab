"""Data for the viewer's "Follow one clip" section: one slow-flutter and one fast-flutter clip
(roughness test, Qwen2-Audio), traced from waveform to spectrogram to the saved readings.

Waveform and spectrogram are recomputed here from the clip file. The per-time-slice readings were
not saved (extract.py keeps only the average over the clip's slices), so the walkthrough shows the
averaged ears reading (1,280 numbers) and every brain layer's averaged reading (4,096 numbers).
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe import load_run  # noqa: E402

RUN = "qwen_roughness"
CLIPS = [("C_oct4_rate1_calm", "slow flutter", "4.5 flutters a second"),
         ("C_oct4_rate1_fear", "fast flutter", "70 flutters a second")]


def spectrogram(x, sr=16000, n_fft=400, hop=160, n_mels=64):
    """Log-mel picture, same frame size and step as Whisper-style ears (25 ms window, 10 ms step)."""
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, n_fft // 2), n_fft)[::hop]
    mag = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    hz = np.linspace(0, sr / 2, mag.shape[1])
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    edges = np.interp(np.linspace(mel(0), mel(8000), n_mels + 2), mel(hz), hz)
    fb = np.zeros((n_mels, len(hz)))
    for i in range(n_mels):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.clip(np.minimum((hz - lo) / (c - lo + 1e-9), (hi - hz) / (hi - c + 1e-9)), 0, None)
    s = np.log10(fb @ mag.T + 1e-10)
    s = np.clip((s - (s.max() - 8)) / 8, 0, 1)
    return (s * 255).round().astype(int)


def q(v):
    """Quantise a reading to -99..99 for display (relative to its own largest value)."""
    v = np.asarray(v, dtype=np.float64)
    return (v / (np.abs(v).max() or 1) * 99).round().astype(int).tolist()


feats, labels, _, _, man = load_run(f"runs/{RUN}")
ids = [c["clip_id"] for c in man["clips"]]
lm = np.load(f"runs/{RUN}.npz")["lm"]
out = []
for cid, name, rate in CLIPS:
    i = ids.index(cid)
    x, sr = sf.read(f"clips/roughness/{cid}.wav", dtype="float64")
    wave = x[: len(x) // 1000 * 1000].reshape(1000, -1)
    out.append({"id": cid, "name": name, "rate": rate, "seconds": len(x) / sr,
                "wave": [[round(float(w.min()), 3), round(float(w.max()), 3)] for w in wave],
                "spec": spectrogram(x).tolist(),
                "ears": q(feats["encoder"][i]),
                "brain": [q(lm[i, layer]) for layer in range(lm.shape[1])]})
Path("viewer/walkthrough.json").write_text(json.dumps(out, separators=(",", ":")))
print("walkthrough.json", Path("viewer/walkthrough.json").stat().st_size // 1024, "KB")
