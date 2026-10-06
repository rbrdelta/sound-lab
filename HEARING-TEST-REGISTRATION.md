# Hearing test, controlled phase — registration

Committed before any clip is rendered or read (2026-10-06). Design decided with Daniel in the
talk-through of 2026-10-06; decisions log in `shadow-ledger/NEXT-SESSION.md` ("Talk-through
decisions — 2026-10-06").

**Role:** instrumentation — a gate on the laboratory model, not an experiment about emotion.

1. **Throughline.** The sound arc's chain (Daniel, 08-08) asks whether audio can induce the state
   that carries strain failure. That is only testable if the laboratory model's ears carry
   musical information at all. This checks the simplest such information — major vs minor —
   under the cleanest conditions, before real-world sounds (a later, separate phase).
2. **Question (Daniel).** Can Qwen2-Audio-7B-Instruct's internal numbers tell a major chord from
   a minor chord, on keys the detector never practised on?
3. **The one variable that moves:** chord quality, major vs minor (the middle note a half-step
   lower). Spread evenly across both qualities so none lines up with the answer: key (12),
   arrangement (root position + two inversions), octave (two adjacent mid-range octaves).
   144 chord clips. (Pitch varying: Daniel. Arrangement and octave added for clip count: Claude,
   delegated by Daniel.)
4. **Held fixed:** instrument (sampled acoustic grand piano, FluidR3_GM, reverb/chorus off —
   Daniel chose piano), strike strength, length 2.0 s, loudness (RMS-matched), 16 kHz mono,
   prompt "Listen to this audio clip." (Daniel), model weights, bf16, eager attention, seed 0.
   No generation and no sampling: one forward pass per clip, read before any output, so there is
   no temperature. **Cannot be held fixed:** absolute pitch content differs between keys/octaves
   (that is why they are spread evenly, and why grading holds out whole keys).
5. **Prediction (Daniel, before any clip existed):** about 60% — below the 2/3 bar. "Given limited
   training, we think it sits around 60% accurate (below our threshold). So we're holding a higher
   bar if we're going to proceed."
6. **Interpretation key.** Primary reading = the encoder (the ears), declared now; the 32
   language-model layers are a profile, and a standout layer is a lead to confirm on fresh clips.
   Leave-one-key-out, pooled over all 144 clips.
   - Sound check first: if chords sit no farther from silence than from each other, the sound is
     not registering — stop; the major/minor number means nothing.
   - Below 2/3 → fail on this model; first check whether the generated sound is the problem
     (speech-trained ears, unfamiliar synthetic input) before dropping Qwen for Audio Flamingo.
   - 2/3 to 0.80 → usable, clip count raised per the simulation.
   - Above 0.80 → pass; clears the real-sounds phase.
   - Pass mark: floor Daniel, upper line Claude (agreed 10-05). It does not move after this.
   - Power: with 144 clips, a coin-flipper clears 2/3 by luck essentially never (<1 in 10,000);
     a true 75% hearer misses the bar ~1 in 100. Clips sharing a key are not independent, so
     the effective count is lower than 144.
7. **What the result changes.** Pass → real-sounds phase. Fail → diagnose the sound, then the
   fallback model. Either way the saved readings stay on file for the later fear detector
   (exploratory, labelled post-hoc).

**Exploratory, not primary (Daniel: "extra data points ... for the sake of documenting
everything"):** whether readings also separate by octave or arrangement; later, the saved
readings run through the fear detector.

**Definition-gate tally** (both = half each). Daniel: question, pitch-varies, three of the
held-fixed items (loudness, instrument, length), instrument choice, prompt, the sound check's
purpose, prediction, phasing, pass-mark floor = 11. Claude: arrangement + octave design,
held-out-key grading, silence method, primary-layer declaration, fail diagnosis = 5. Both:
throughline (his chain, Claude's framing), pass mark upper line (Claude proposed, Daniel agreed)
= 1 each. Daniel 12 / Claude 6 — above half.

Scripts: `make_clips.py` (clips), `extract.py` (readings), `hearing_test.py` (scoring).

## Result — 2026-10-06 (pod sound-lab-hearing, ~14 min, terminated after)

Checks: render identical across two passes (md5); repeat reading identical; every chord
registers vs silence at every layer.

**Primary (encoder): 0.389 held-out-key accuracy (56/144), ranking 0.32 → FAIL.** Every one of
the 32 language-model layers also fails (range 0.389–0.556; none reaches 2/3). Daniel predicted
~60%, below the bar: direction right, level lower than predicted.

Post-hoc, exploratory (not registered):
- Detector that has seen every key (hold out octave+arrangement instead): encoder 0.493,
  lm_8 0.549, lm_16 0.562, lm_31 0.514 — still at chance.
- Null by swapping major/minor within matched pairs (200 shuffles): encoder shuffled median
  0.500, middle 95% 0.431–0.556; real 0.389 sits below all 200. lm_31 likewise (0.389, 1%).
  lm_16 (0.535) is inside the shuffle range. So the encoder's below-chance score is systematic,
  not luck and not a method artifact — but it is not usable hearing and is unexplained. Lead
  only.

Files: readings `sound-lab/runs/hearing_controlled.npz` (laptop, gitignored, 65 MB — kept for
the later fear-detector pass), scored result `runs/hearing_controlled.result.json` (committed),
clips regenerate exactly from `make_clips.py`.
