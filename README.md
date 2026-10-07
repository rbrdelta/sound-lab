# sound-lab

Can an open-weights audio-language model hear the sound cues people associate with fear? This repo
holds the hearing tests from a personal research program on functional emotions in AI models
(rowbyroh). The program's question is whether music or audio can create an internal state in a
model, fear vs calm, that changes its behavior. Before asking whether a model *feels* a sound, these
tests check whether it *hears* it: does the model's internal state (its activations) change
reliably when one property of the sound changes?

Every test was registered, with its prediction, before it ran. The registrations, predictions,
results and post-hoc checks are in [HEARING-TEST-REGISTRATION.md](HEARING-TEST-REGISTRATION.md).

## Results (2026-10-06)

144 clips per test, generated in code with everything held fixed except one property, spread
evenly across 12 keys or notes. A difference-of-means detector was graded leave-one-key-out, so
every clip was scored by a detector that never saw its key. The table shows the reading at the
audio encoder (the model's "ears").

| Test | What changes between the two groups | Qwen2-Audio-7B-Instruct | Audio Flamingo Next |
|---|---|---|---|
| Major vs minor, held chords | the middle note, a half-step | 39% | 42% |
| Major vs minor, 4 s tunes | the 3rd and 6th of the scale | 36% | 34% |
| Roughness | loudness flutter: 3–6 vs 40–100 per second | 100% | 100% |
| Volume | gradual vs jumping loudness, same levels | 100% | 100% |
| Tempo | 60–80 vs 160–200 repeats per minute | 99% | 100% |

Both models hear physical changes over time almost perfectly and do not hear mode. On major vs
minor the per-key change in the readings points a different way in each key, so a detector built
on 11 keys points backwards on the 12th (post-hoc; see the registration file). Shuffling the group
labels drops every passing test to about 50%, so the detector does not find splits in noise.

These are hearing results only. The group names in the data (`fear` / `calm`) are tags for the
fast and slow settings; nothing here measures fear. That is the next step, with human-rated real
music.

## Scratch Paper Viewer

An interactive page showing what happens inside the model for every clip: one row per clip, scored by the held-out detector, layer by layer. File: [`viewer/scratch-paper-viewer.html`](viewer/scratch-paper-viewer.html) (download and open in a browser). A hosted version will be linked here from rowbyroh.com once it is up.

## Files

| File | What it does |
|------|--------------|
| `make_clips.py` | Generates the clip sets: `--set chords`, `tunes`, `roughness`, `volume`, `tempo`. Chords, tunes and roughness use the FluidR3_GM SoundFont through fluidsynth; volume and tempo use a steady tone generated in code. |
| `extract.py` | Plays each clip through the model in one forward pass with a fixed prompt and no generation. Records the encoder output and every language-model layer, averaged over the audio positions, and checks that a repeated reading is byte-identical. |
| `hearing_test.py` | Scores a run: a sound-vs-silence check, then leave-one-key-out accuracy per layer against a pass mark. |
| `probe.py` | The detector: the average of one group minus the average of the other, with the cutoff halfway between. |
| `simulate_clips.py` | Estimates how many clips a later experiment needs, given a detector's accuracy. |
| `viewer/` | Builds the Scratch Paper Viewer: per-layer score plots for every test, plus one clip traced from waveform to readout. The built page is committed as `viewer/scratch-paper-viewer.html` (one self-contained file; download and open it in a browser). |
| `runpod/` | Pod setup (`bootstrap.sh`) and an idle watchdog that stops the pod. |
| `runs/*.result.json`, `runs/*.manifest.json` | Scores and run manifests for every test. The raw readings (`.npz`, about 570 MB) are not in the repo. |
| `REAL-MUSIC-SOURCES.md` | Emotion-labelled music datasets for the next step, and their licenses. |
| `tests/` | CPU tests: the worksheet numbers, a planted signal is recovered, determinism, and clip-set balance. |

## Reproduce

CPU checks:

    python3 -m venv .venv
    .venv/bin/pip install -r requirements-test.txt
    ./verify.sh

A full run needs one 24 GB GPU (an RTX 4090 was used):

    bash runpod/bootstrap.sh
    python make_clips.py --set roughness --out clips/roughness
    python extract.py --clips clips/roughness/clips.csv --out runs/qwen_roughness --prompt "Listen to this audio clip."
    python hearing_test.py runs/qwen_roughness --positive fear --negative calm --fail-below 0.75 --pass-above 0.90

The clips regenerate byte-identically from `make_clips.py`.

## Notes

- The models: Qwen2-Audio-7B-Instruct (Apache-2.0) and Audio Flamingo Next (NVIDIA OneWay
  Noncommercial License, academic use only). This work is unpaid personal research.
- Music from third-party datasets is never committed here.
- Design decisions were made by the program's author. Claude (Anthropic) wrote the code and ran
  the experiments under that direction.
- The research notebook (the thesis and session notes) is kept privately. References to
  `shadow-ledger/...` in the registration file point there.
