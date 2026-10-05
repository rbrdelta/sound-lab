# sound-lab — hearing-test setup for the sound arc

Setup code only, written so that GPU hours go on running and not on setting up. Nothing here has
spent money, downloaded a model, or run on a GPU.

**The clip set has NOT been written, and should not be yet.** It waits on a talk-through with
Daniel about what else could carry the signal (pitch, volume, tempo, instrument) and how each one
is held fixed (NEXT-SESSION.md, "Hearing test decisions — 2026-10-05"). There is deliberately no
code here that makes clips or picks stimulus settings.

## Files

| File | What it does |
|------|--------------|
| `extract.py` | Loads the model (Qwen2-Audio first, Audio Flamingo Next as fallback), plays each clip in one forward pass with a fixed prompt and no generation, and records the scratch work at the audio encoder (the ears) and at every language-model layer. Each one is averaged over the audio positions and written to `.npz` along with a manifest (clip, label, source, recording, model, versions, dtype, seed, prompt). Before it saves, it reads the first clip a second time and checks that both readings are identical. |
| `probe.py` | The worksheet's fear detector, run once per layer: the direction is the average of one label minus the average of the other, the cutoff sits halfway between them, and it reports accuracy plus a ranking score on clips it never saw. Clips are split by recording, never row by row. You can also hold out whole sources as a cross-source test, which is the check that catches the loudness trap. It applies the agreed pass mark: below 2/3 fails, 2/3 to 0.80 is usable with more clips, above 0.80 passes. |
| `simulate_clips.py` | Given a hearing accuracy and an assumed real effect, it estimates by simulation how many clips you need to catch that effect. |
| `tests/` | CPU tests. `test_probe.py` checks the exact worksheet numbers, that a planted direction is recovered, the loudness trap at scale, and that the same input always gives the same output. `test_extract.py` runs the hook and averaging code on tiny, randomly set-up copies of both model classes; nothing is downloaded. |
| `runpod/bootstrap.sh` | Run once on a fresh pod. It installs pinned versions, runs the tests and starts the watchdog. |
| `runpod/idle_watchdog.sh` | Stops the pod after N minutes with an idle GPU (default 20, with 15 minutes of grace after start). |
| `verify.sh` | Runs the CPU checks. |

The clip list for `extract.py` is a CSV with the columns `clip_id,path,label,source,recording`.

## Run the tests

    python3 -m venv sound-lab/.venv
    sound-lab/.venv/bin/pip install numpy pytest
    sound-lab/.venv/bin/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu   # optional
    sound-lab/.venv/bin/pip install transformers==5.6.2                                            # optional
    sound-lab/verify.sh

## Not yet verified (needs the pod)

- The real processors and checkpoints. The loading code and input building in `extract.py` follow the model cards and the transformers 5.6.2 source, but have not been run. One known change: transformers 5 renamed Qwen's `audios=` to `audio=`.
- Whether the readings are byte-identical on a GPU. `extract.py` checks this every time it runs and exits with an error if the two readings differ.
- On the pod: `runpodctl`, the `RUNPOD_POD_ID` setting, whether the CUDA 12.8 torch build works, and whether the watchdog really stops billing. Check the console the first time it fires.
- Clips longer than 30 seconds (Audio Flamingo cuts them into chunks). They have not been tested.
- The prompt wording is only a placeholder (`--prompt`). Choosing it is a design decision.
