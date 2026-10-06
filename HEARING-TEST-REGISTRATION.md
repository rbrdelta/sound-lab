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

Exploratory "what else is on the scratch paper" (registered as exploratory; same detector,
leave-one-key-out): octave (low vs high) encoder 0.86, LM layers 0.77–0.85; note order root vs
1st inversion ~0.57–0.61, root vs 2nd inversion ~0.66–0.68; major/minor 0.39–0.56. Inversions
also raise average pitch, so the most likely single reading is that the readings carry how HIGH
the notes are, not the intervals between them. Not tested further.

## Daniel's direction after the result (2026-10-06)

- He found the sample hard to tell apart by ear too ("that was really difficult for me too").
- Next: rerun on real audio, normalised so volume and length are fixed (key spread evenly as
  before — the notes themselves must differ for major/minor to exist).
- Then the same test on Audio Flamingo, to see whether the two models differ.
- Framing: major vs minor is one dimension of several that can elicit calm or fear; this is
  step one.

---

# Hearing test, controlled phase, second run: tunes instead of chords — registration

Committed before any tune is rendered or read (2026-10-06).

1. **Throughline.** The held-chord run failed (encoder 0.389), and Daniel found the pairs hard to
   tell apart by ear. Mode is normally heard in context, so the fail may reflect a weak stimulus,
   not missing hearing. This run changes one thing, held chord to 8-note tune, before any move to
   recorded music (later) or Audio Flamingo (after this run, Daniel).
2. **Question (same as run one).** Can the internal numbers tell major from minor on keys the
   detector never practised on?
3. **The one variable that moved from run one:** clip form, a held 2 s chord → a 4 s 8-note tune
   (0.5 s per note). Within the run, the variable is still major vs minor: the 3rd and 6th of the
   scale a half-step lower. Spread evenly: 12 keys, 2 octaves, 3 tunes (melody 1-2-3-5-6-5-3-1,
   arpeggio 1-3-5-8-8-5-3-1, scale 1-2-3-4-5-6-5-4) replacing the 3 arrangements. 144 clips +
   silence.
4. **Held fixed:** as run one (piano, velocity, RMS loudness, 16 kHz mono, prompt, model, bf16,
   eager attention, seed 0, single forward pass), with length now 4.0 s on every clip (Daniel).
   Cannot be held fixed: pitch content across keys/octaves (spread evenly; whole keys held out).
5. **Prediction (Daniel):** better than a coin flip, over 60%. His reasoning: "If it can
   understand how high the note it then it has a sense of pitch at least. So I do expect better
   than a coin flip." Given to him first: run one's scratch paper carried how high notes are,
   not the gaps between them, and major vs minor is entirely in the gaps.
6. **Interpretation key:** unchanged from run one (encoder primary; pass mark fixed; sound check
   first). Added: pass → recorded real music next on Qwen; fail → strong evidence Qwen's ears do
   not carry mode in a usable form, and Audio Flamingo runs the same tune set (Daniel's next step
   either way, after the license check).
7. **What it changes:** pass or fail, Audio Flamingo is next (Daniel). Pass also unlocks the
   recorded-music phase on Qwen.

**Definition-gate tally:** Daniel — 4 s length, loudness/length normalised, prediction, Audio
Flamingo next = 4. Claude — sequence over chord (Daniel asked which is the better test), tune
shapes, mood-note design = 3. Both — the question = 0.5 each. Daniel 4.5 / Claude 3.5.

## Result, run two (tunes) — 2026-10-06 (pod sound-lab-tunes, ~14 min, terminated after)

Checks: render identical across two passes; repeat reading identical; every tune registers vs
silence at every layer.

**Primary (encoder): 0.361 held-out-key accuracy (52/144), ranking 0.26 → FAIL.** All 32 LM
layers fail (0.361–0.500). Daniel predicted over 60%: not borne out. The wrong-way lean from
run one reproduced and grew (0.389 → 0.361).

Post-hoc, exploratory — most likely reading of the wrong-way lean (both runs): per key, the
average major→minor change in the readings is real but points a different way in each key, and
the 12 directions largely cancel (|mean| / mean |change| = 0.18–0.22 at encoder and last layer;
~0.29 would be expected from unrelated directions). Each held-out key's change points against
the other 11 keys' average (cosine −0.19 to −0.26; 10/12 keys negative at the encoder, 11/12 at
the last layer). A detector built on 11 keys therefore points backwards for the 12th. Reading:
the ears register *which note* moved, not a shared "went minor" quality. Mid layer (lm_16) shows
the pattern weakly (−0.04/−0.09). Not registered; a lead, not a finding.

Next (Daniel's sequence): the same tune set on Audio Flamingo, after the license check.

## Audio Flamingo license check — 2026-10-06

Read the license text itself (NVIDIA OneWay Noncommercial License, 22 Mar 2022, "academic"
variant, from the model repo). Use clause 3.3, verbatim: "The Work and any derivative works
thereof only may be used or intended for use non-commercially. ... As used herein,
'non-commercially' means for academic purposes only." "Academic" is not defined. The grant
(2.1) includes "publicly display", and nothing restricts publishing results.

Daniel's position: this is purely academic research; interest in a research role at Anthropic
does not change that. Claude's read agrees (not legal advice): unpaid personal research with
published findings is the use the clause allows; the job interest is a reason for doing
research, not a commercial use of the model. Boundary: never use Audio Flamingo or anything
derived from it in Partswatch or any paid product/service.

---

# Hearing test, Audio Flamingo on both clip sets — registration

Committed before Audio Flamingo reads any clip (2026-10-06).

1. **Throughline.** Qwen failed both sets (chords 0.389, tunes 0.361). Daniel's sequence: the
   same tests on the fallback model, to see whether a model trained on more music hears mode.
2. **Question:** same as before, per clip set.
3. **The one variable that moves vs the Qwen runs:** the model (Qwen2-Audio-7B-Instruct →
   nvidia/audio-flamingo-next-hf, the instruction-tuned checkpoint). Clip sets byte-identical
   to the Qwen runs (regenerated from make_clips.py; md5 compared against the Qwen-run renders).
4. **Held fixed:** both clip sets, prompt "Listen to this audio clip.", bf16, eager attention,
   seed 0, single forward pass, grading, pass mark. Cannot be held fixed: the model's own
   processor (input handling) and layer count differ — that is part of "the model".
5. **Predictions (Daniel), before any reading:**
   - Held chords: "for one chord it's hard either way, so it won't do much better than Qwen."
   - Tunes: "if it's trained on music, it should theoretically do better than a coin flip, so
     above 60%."
   Context given first: same Whisper-derived ear lineage, more music training (Music Flamingo
   data); chords/key/major-minor not mentioned in the readable paper text; key appears in the
   card's example caption prompt.
6. **Interpretation key:** encoder primary; pass mark unchanged. Chords fail + tunes pass →
   context is what lets this model hear mode; it becomes the laboratory-model candidate. Both
   fail → neither available open model carries mode in a usable form; bring back to Daniel
   before any further model search. Both pass → strong candidate.
7. **What it changes:** decides whether the sound arc has a laboratory model.

**Definition-gate tally:** Daniel — run on Audio Flamingo, two predictions, reasoning for each =
4 (counting each prediction with its reasoning as one: 2) → Daniel 3 (model choice, 2
predictions); Claude — run both sets = 1; license reading = mechanics. Daniel above half.

**License:** read and recorded above (non-commercial = academic purposes only).

## Result, Audio Flamingo on both sets — 2026-10-06 (pod sound-lab-flamingo, ~20 min, terminated)

Checks: clip sets byte-identical to the Qwen runs (md5); repeat reading identical for both sets;
every clip registers vs silence at every layer. Model has 28 LM layers (card config; paper says 36).
Note: AF's input format puts the prompt before the audio, so its LM layers (not the encoder)
can be shaped by the prompt where Qwen's cannot.

- **Held chords — encoder 0.424 (61/144), ranking 0.38 → FAIL.** All 28 LM layers 0.417–0.458.
  Daniel predicted "won't do much better than Qwen": borne out (Qwen 0.389).
- **Tunes — encoder 0.340 (49/144), ranking 0.26 → FAIL.** LM layers 0.326–0.451. Daniel
  predicted above 60%: not borne out.

Post-hoc, exploratory: keys-seen accuracy 0.465–0.618 (best: tunes, mid layer 14 at 0.618 — one
of several looks, a lead at most). The rotating-direction pattern found in Qwen repeats in both
AF sets: each key's major→minor change points against the other 11 keys' average (−0.13 to
−0.30; 10–12/12 keys negative) and the 12 changes largely cancel (0.17–0.23). Four runs, two
models, same shape: the change registers per key, not as a shared "minor" quality.

Per the interpretation key: both fail → neither available open model carries mode in a form
the registered detector can read; back to Daniel before any further model search.

---

# Hearing test, roughness (calm vs fearful flutter) — registration

Committed before any clip is rendered or read (2026-10-06).

1. **Throughline (Daniel).** Major vs minor is not detectable, so it is not a lever for inducing
   fear later. Fear is a combination of cues that music and voice share; Daniel's sequence tests
   them one at a time — roughness, then volume, then tempo — toward "can a model 'feel the
   sound' ... which elicit behavioral change?" This is the hearing test for the first cue.
2. **Question.** Can the internal numbers tell calm flutter from fearful flutter on notes the
   detector never practised on?
3. **The one variable that moves:** flutter rate — how many times a second the loudness swings.
   Calm 3 / 4.5 / 6 per second (speech-like); fear 40 / 70 / 100 per second (scream "roughness"
   range, 30–150). Slow vs fast (Daniel) rather than steady vs fast, so only the rate differs.
   Spread evenly: 12 notes x 2 octaves, three rates per class. 144 clips + silence.
4. **Held fixed:** carrier = one sustained organ note (General MIDI drawbar organ; chosen because
   a piano note fades and strings carry their own natural vibrato near the calm rates), flutter
   depth (loudness swings between 20% and 180% of average), length 3.0 s, average loudness
   (RMS-matched after flutter), 16 kHz mono, prompt, bf16, eager attention, seed 0, single
   forward pass. Graded leave-one-note-out (whole pitch classes held out). Cannot be held fixed:
   pitch across notes (spread evenly, held out).
5. **Prediction (Daniel), verbatim:** "it's not about the pitch, it's about the intonations, which
   are the tested rates should be very visible. So I expect clear improvements in scoring." (No
   number stated; not converted into one here.)
6. **Interpretation key — new pass mark set by Daniel for this cue:** below 75% → do not use;
   75%–90% → usable, with more clips; above 90% → use as is. Sound check (vs silence) first,
   as for every test (Daniel). Encoder primary; LM layers a profile.
7. **What it changes:** pass → roughness becomes a candidate fear lever, and the volume test
   follows; fail → it does not, and the volume test follows (Daniel's sequence either way).
   **Models:** both Qwen2-Audio and Audio Flamingo Next, each scored on its own — the comparison
   Daniel asked for in the last round.

**Definition-gate tally:** Daniel — throughline/reframe, cue choice (roughness), slow-vs-fast
contrast, sound check for each, prediction, pass-mark lines (75 / 90) = 6. Claude — carrier
choice, specific rates and depth, spread/grading = 3. Daniel above half.

## Result, roughness — 2026-10-06 (pod sound-lab-roughness, ~20 min, terminated)

Checks: render identical across two passes; repeat reading identical (both models); every clip
registers vs silence at every layer (both models). Carrier note's own loudness wobble ±26% over
the held part (drawbar organ), vs the ±80% flutter added — identical across classes, so it is not
a class difference, but it is not a perfectly steady carrier either.

- **Qwen2-Audio — encoder 1.000 (144/144) → PASS (above Daniel's 90% line).** Every LM layer
  0.944–1.000 (dip to ~0.95 across mid layers 8–16, back to 1.000 by layer 21).
- **Audio Flamingo Next — encoder 1.000 (144/144) → PASS.** Every one of 28 LM layers 1.000.
- Daniel's prediction ("very visible ... clear improvements in scoring"): borne out.

Scope: the classes were far apart by design (3–6 vs 40–100 flutters a second). This shows the
ears carry flutter rate cleanly; it does not show where between those rates the line sits, and
it is a hearing result, not evidence the model "feels" fear.

---

# Hearing test, volume (gradual vs abrupt loudness change) — registration

Committed before any model reads a clip (2026-10-06). Clips rendered locally first (no
SoundFont needed) to check they play correctly; the pod renders them again and the md5s must match.

1. **Throughline (Daniel).** Second cue in his sequence (roughness passed at 100% on both
   models). Still a "can the ears hear it" check; grey-scale belongs in the later fear-detection
   work.
2. **Question.** Can the internal numbers tell gradually changing loudness (calm) from jumping
   loudness (fear), on notes the detector never practised on?
3. **The one variable that moves (Daniel's correction):** how loudness changes — calm walks
   through the levels in order and glides between them; fear visits the same levels in a jumping
   order (every step at least 3 levels apart) and cuts instantly. **Range held fixed:** every
   clip visits the same 8 levels, −6 to +6 dB, once each. Daniel's first sketch moved range too;
   he fixed it to one variable. Spread: 3 orders per class, 12 notes x 2 octaves = 144 + silence.
4. **Held fixed:** carrier = a steady tone generated in code (a note plus its first 5 overtones,
   organ-like, zero wobble — instrument changed per Daniel because the SoundFont organ wobbles
   about ±2 dB on its own), level set, length 3.0 s, average loudness (RMS-matched), 16 kHz mono,
   prompt, bf16, eager attention, seed 0, single forward pass. Graded leave-one-note-out.
5. **Prediction (Daniel): 70%.** "Against a gradient this is more subtle."
6. **Interpretation key — pass mark set by Daniel for this cue:** below 2/3 fail; 2/3–80% usable
   with more clips; above 80% pass. Sound check first. Encoder primary.
7. **What it changes:** pass → volume joins roughness as a candidate fear cue; then tempo.
   Both models, scored separately.

**Definition-gate tally:** Daniel — cue, calm/fear concept (progressive vs jumps), fixing range
to one variable, instrument change, prediction, pass mark = 6. Claude — level set and orders,
glide/cut implementation, steady-tone carrier design = 3. Daniel above half.

## Result, volume — 2026-10-06 (pod sound-lab-volume, ~15 min, terminated)

Checks: pod renders byte-identical to the local renders (md5); repeat reading identical (both
models); every clip registers vs silence at every layer.

- **Qwen2-Audio — encoder 1.000 (144/144) → PASS (above 80%).** All 32 LM layers 1.000.
- **Audio Flamingo Next — encoder 1.000 (144/144) → PASS.** All 28 LM layers 1.000.
- Daniel predicted 70% ("against a gradient this is more subtle"): the result was higher — the
  ears separate gradual from jumping loudness completely, with range held fixed.

Scope: as with roughness, the classes were far apart by design (glides vs jumps of 3+ levels with
instant cuts). Instant cuts also add brief clicks/broadband energy at each jump — part of what
"abrupt" sounds like, but a separable sub-cue if the fear-detection work needs to attribute it.
