"""One clip's emotion timeline: a local page, one row per emotion over the waveform.

Registration section 12: each half-second window shaded pale to strong orange by how active that
emotion's arrow is, the waveform underneath. Waves (score_decoding.find_waves) are marked. The page
holds the film clip's shape, so it is written under runs/ and never committed or hosted.

Usage:
  python viewer/build_timeline.py runs/real-music/qwen_film_decoding set1_074 --layer lm_16
Writes runs/real-music/timeline-qwen_film_decoding-set1_074-lm_16.html
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score_decoding as sd  # noqa: E402

PAGE = """<!doctype html><meta charset="utf-8"><title>Timeline: __CLIP__</title>
<style>
/* A lab notebook page: emotion rows are the exhibit, the waveform sits on the same time axis. */
:root { --bg:#f6f7f4; --panel:#fff; --ink:#1d2320; --muted:#5f6b66; --rule:#d9ded9; --heat:194,65,45 }
@media (prefers-color-scheme: dark) { :root { --bg:#141816; --panel:#1c211f; --ink:#e7ece9;
  --muted:#9aa8a2; --rule:#2e3532; --heat:240,122,98 } }
body { background:var(--bg); color:var(--ink); font:16px/1.5 system-ui,sans-serif; margin:0 }
.wrap { max-width:1040px; margin:0 auto; padding:28px 16px 48px; display:grid; gap:16px }
h1 { font:600 26px/1.15 Georgia,serif; margin:0 }
p { color:var(--muted); margin:0; max-width:72ch }
.board { background:var(--panel); border:1px solid var(--rule); border-radius:10px; padding:16px;
  display:grid; grid-template-columns:76px 1fr; gap:4px 10px; align-items:center }
.name { font-size:14px; text-align:right; color:var(--muted) }
.row { display:grid; grid-auto-flow:column; grid-auto-columns:1fr; height:30px; gap:1px }
.cell { position:relative }
.cell.peak::after { content:""; position:absolute; inset:50% auto auto 50%; width:9px; height:9px;
  margin:-5px; border-radius:50%; background:var(--panel); border:2px solid var(--ink) }
svg { width:100%; height:70px; display:block }
svg path { fill:var(--muted) }
.axis { display:flex; justify-content:space-between; font:12px ui-monospace,monospace; color:var(--muted) }
button { font:14px system-ui,sans-serif; padding:4px 10px; border:1px solid var(--rule); border-radius:6px;
  background:var(--panel); color:var(--ink); cursor:pointer }
button[aria-pressed=true] { border-color:var(--ink); font-weight:600 }
#hint { font:13px ui-monospace,monospace; color:var(--muted); min-height:1.5em }
</style>
<div class="wrap">
  <h1>__CLIP__</h1>
  <p>__MODEL__, layer __LAYER__. Each square is __WINDOW__ seconds. Stronger orange means that emotion
  has risen further above its usual level in this clip. "As read" shows every rise, including small
  ones in the secondary emotions. "Net of the shared rise" takes out what all five share at that
  moment, which is what the wave rule uses. The "all five" row shows that shared rise on its own. A
  ring marks the peak of a wave. Inferred from the model's readings, not proof of anything felt.</p>
  <div><button id="raw" aria-pressed="true">As read</button>
  <button id="net" aria-pressed="false">Net of the shared rise</button></div>
  <div class="board" id="board"></div>
  <div id="hint">Point at a square for its numbers.</div>
</div>
<script>
const D = __DATA__;
const board = document.getElementById('board'), hint = document.getElementById('hint');
const cells = [];
const addRow = (label, id) => {
  board.insertAdjacentHTML('beforeend', `<div class="name">${label}</div><div class="row" id="${id}"></div>`);
  return document.getElementById(id);
};
D.emotions.forEach((e, k) => {
  const row = addRow(e, 'r' + k);
  D.rises.forEach((r, t) => {
    const c = document.createElement('div');
    c.className = 'cell' + (D.peaks.some(p => p.emotion === e && p.peak_window === t) ? ' peak' : '');
    c.onmouseenter = () => hint.textContent = `${e} at ${(t * D.window).toFixed(1)}s: rise as read ` +
      `${D.raw[t][k].toFixed(2)}, net ${r[k].toFixed(2)}, score ${D.scores[t][k].toFixed(2)}, wave bar ${D.bars[k].toFixed(2)}`;
    row.append(c); cells.push({ c, t, k });
  });
});
const sharedRow = addRow('all five', 'shared');
const sharedCells = D.shared.map((v, t) => {
  const c = document.createElement('div');
  c.className = 'cell';
  c.onmouseenter = () => hint.textContent = `all five at ${(t * D.window).toFixed(1)}s: shared rise ${v.toFixed(2)}`;
  sharedRow.append(c);
  return c;
});
const heat = (v, most) => `rgba(var(--heat),${(0.06 + 0.94 * Math.max(v, 0) / most).toFixed(3)})`;
function show(view) {
  const values = D[view], most = Math.max(...D.raw.flat(), ...D.rises.flat(), 1e-9);
  cells.forEach(({ c, t, k }) => c.style.background = heat(values[t][k], most));
  sharedCells.forEach((c, t) => c.style.background = heat(D.shared[t], most));
  document.getElementById('raw').setAttribute('aria-pressed', view === 'raw');
  document.getElementById('net').setAttribute('aria-pressed', view === 'rises');
}
document.getElementById('raw').onclick = () => show('raw');
document.getElementById('net').onclick = () => show('rises');
show('raw');
const n = D.envelope.length, up = D.envelope.map((v, i) => `${i},${50 - 48 * v}`);
const down = D.envelope.map((v, i) => `${i},${50 + 48 * v}`).reverse();
board.insertAdjacentHTML('beforeend', `<div class="name">sound</div>
  <svg viewBox="0 0 ${n} 100" preserveAspectRatio="none"><path d="M${up.concat(down).join('L')}Z"/></svg>
  <div></div><div class="axis"><span>0s</span><span>${(D.rises.length * D.window).toFixed(1)}s</span></div>`);
</script>
"""


def envelope(audio: np.ndarray, points: int = 600) -> list[float]:
    """The sound's outline: loudest sample in each slice, scaled to 1. Not the audio itself."""
    a = np.abs(np.asarray(audio, dtype=np.float64))
    if a.ndim > 1:
        a = a.mean(1)
    peaks = [float(c.max()) if len(c) else 0.0 for c in np.array_split(a, points)]
    top = max(peaks) or 1.0
    return [round(p / top, 3) for p in peaks]


def timeline_data(W, wc, X, labels, row, window_seconds, audio) -> dict:
    """Everything the page draws for the clip at `row` of the run."""
    scores = sd.moment_scores(W, wc, X, labels)
    bars = sd.wave_bars([s for i, s in scores.items() if labels[i] == "neutral"])
    s = scores[row]
    raw = s - np.median(s, axis=0)
    return {"emotions": list(sd.EMOTIONS), "rises": np.round(sd.rises(s), 3).tolist(),
            "raw": np.round(raw, 3).tolist(), "shared": np.round(np.median(raw, axis=1), 3).tolist(),
            "scores": np.round(s, 3).tolist(), "bars": np.round(bars, 3).tolist(),
            "peaks": sd.find_waves(s, bars), "window": window_seconds, "envelope": envelope(audio)}


def page(clip_id: str, model: str, layer: str, data: dict) -> str:
    return (PAGE.replace("__CLIP__", clip_id).replace("__MODEL__", model).replace("__LAYER__", layer)
            .replace("__WINDOW__", f"{data['window']:.2f}").replace("__DATA__", json.dumps(data)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prefix", help="run prefix, e.g. runs/real-music/qwen_film_decoding")
    p.add_argument("clip_id")
    p.add_argument("--layer", required=True, help="encoder or lm_<n>")
    p.add_argument("--clips", default="clips/film/decoding/clips.csv", help="clip list, for the audio path")
    args = p.parse_args(argv)

    import soundfile as sf

    prefix = Path(args.prefix)
    man = json.loads(prefix.with_suffix(".manifest.json").read_text())
    labels = np.array([c["label"] for c in man["clips"]])
    row = [c["clip_id"] for c in man["clips"]].index(args.clip_id)
    names, mats = sd.layer_matrices(np.load(prefix.with_suffix(".npz")))
    L = names.index(args.layer)
    with np.load(prefix.with_suffix(".moments.npz")) as m:
        wc = m["window_clip"]
        W = (m["enc_windows"] if L == 0 else m["lm_windows"][:, L - 1]).astype(np.float64)
    path = {r["clip_id"]: r["path"] for r in csv.DictReader(open(args.clips))}[args.clip_id]
    window = man["moments_window_tokens"] / man["audio_tokens_per_second"]
    data = timeline_data(W, wc, mats[L], labels, row, window, sf.read(path)[0])
    out = prefix.with_name(f"timeline-{prefix.name}-{args.clip_id}-{args.layer}.html")
    out.write_text(page(args.clip_id, man.get("model", ""), args.layer, data))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
