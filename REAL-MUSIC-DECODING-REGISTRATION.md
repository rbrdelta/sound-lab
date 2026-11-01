# Real music, decoding step — registration (approved 2026-11-01)

Rewritten 2026-10-21 at the end of the design talk. Every design answer from Daniel is on file
verbatim in the private notebook (`shadow-ledger/NEXT-SESSION.md`, from "Real-music test —
Daniel's first design inputs" onward). Who decided what is marked inline. Draft 3 (2026-11-01) records Daniel's answers to the three open questions and the
recognition risk; what changed is listed at the end. Becomes the registration when Daniel approves
it. Nothing has been run.

**Role:** instrumentation. The first of two clean parts (Daniel): **decoding**, then
**measuring** (the ten-word experiment). Decoding finds the emotion directions that the measuring
part, and later the switch-an-emotion-on step, depend on. It does not measure behavior.

## 1. Throughline

The hearing tests (2026-10-06) showed both models' ears pick up physical change over time (tempo,
roughness, loudness) almost perfectly and do not pick up major vs minor. So some cues people link
to emotion survive the conversion of sound into numbers, and at least one does not. Whether
emotion itself survives is unknown (Daniel). This step extends the map from sound properties to
emotion. Without emotion directions there is nothing to compare the ten words against and nothing
to switch on (Daniel: "that's the key step and open question").

## 2. Questions

1. **Detection (Daniel).** When a real music clip is played to the model, can a readout of its
   inside tell which of five emotions listeners rated it as? Start with the obvious ones "to see
   if it can even detect the emotion first."
2. **When and where (Daniel).** Which moment in the sound, and which layer, does the emotion
   register at? Does averaging over the whole clip blur a signal that only spikes at certain
   moments?
3. **Opposites and clusters (Daniel).** Built separately, do opposite emotions point in opposite
   directions (happiness vs sadness; anger vs fear)? Do related ones point the same way?

What differs from Anthropic's emotions paper (Daniel; notebook 2026-10-21): the input is sound,
not text; the question is detecting emotion in the input, so the early-to-middle layers matter
more than the layers that prepare the answer; and like the paper, the claim can only ever be
"directionally present," with stand-ins such as loudness removed as far as possible.

## 3. The one variable that moves

The emotion the clip was chosen for: anger, fear, happiness, sadness, tenderness (Daniel: five,
the film set's rated emotions). Equal clips per emotion, random order (Daniel).

## 4. Held fixed, and what can't be

**Held fixed:** prompt "Listen to this audio clip." (Daniel); no answer generated, one forward
pass per clip, read before any output (Daniel); 16 kHz mono; model weights, bf16, eager attention,
seed 0. Both models: Qwen2-Audio-7B-Instruct and Audio Flamingo Next (Daniel).

**Can't be held fixed:** everything else about real music (instruments, density, the film).
Spread across emotions only by the random draw.

**Loudness is kept as recorded** (Daniel: "a dimension we shouldn't eliminate"). It differs by
emotion (average level of the decoding clips): anger −20.4 dB, happiness −21.6, fear −26.9,
sadness −28.0, tenderness −28.6. Named risk: **loudness posing as emotion.** Handled by the
energy-matched checks (section 8) and, if results are inconclusive, by removing the loudness
direction from the readings and scoring again (ablation; Daniel), before any re-run with
leveled clips.

Second named risk: **recognition posing as hearing** (Daniel, 2026-11-01). If the model recognizes a
cue (the film, composer or artist), an emotion reading could come from what it learned about the
film in text rather than from the sound. The set's own readme calls the excerpts "not that well
known... although some could be identified by film aficionados." Not tested in this step. **Asking
the model to name the clip is ruled out: it is leading** (Daniel). The check, run only if emotions
are captured, is to read which words the model's inside points to for each clip and look for the
film, composer or artist; how to handle a hit is to be discussed then.

**What a positive result leads to (Daniel, 2026-11-01):** two follow-ups, neither run in this step.
(1) The recognition check above. (2) A listening check, clip by clip, half a second at a time:
verify that the emotion readings follow what happens in the sound, as evidence the model actually
listened. The moment-by-moment readings saved in this step are the data for it.

## 5. Clips

Film soundtracks, Eerola & Vuoskoski 2011 (OSF p6vkg). Audio kept private, never in this repo.
Built by `make_film_clips.py`.

- **Decoding: 100 clips**, 20 per emotion, from the larger first set, "mostly clear, but some
  unclear" (Daniel): 15 where listeners rated the chosen emotion highest, 5 where they rated
  another emotion highest (split: Claude). The 75 clear clips are also scored on their own.
- **Measuring part: 25 clips**, the second set's five "high" groups. **No clip is in both**
  (Claude's proposal; Daniel: use the clips "for whatever helps us bring the most value"). The
  second set's published links to the first set were confirmed against the audio for all 25.
- **References (Daniel: silence, a flat tone and neutral clips):**
  - silence, 15 s;
  - a flat tone: steady 220 Hz, no rhythm, no loudness change, at the decoding clips' median
    loudness (pitch and level: Claude);
  - 15 rated-neutral clips: listeners rated all five emotions 4 or under (of 9) and pleasantness,
    energy and tension each in the middle (3.5–6.5). Rule: Claude, delegated by Daniel. Caveat: 8
    of the 15 were picked by the authors as "high energy" clips, though listeners rated their
    energy middling.
- **Limitation:** the first set's track list is not published, so two excerpts of the same film
  cue could sit on both sides of the split.

## 6. What is read (one pass collects all of it)

Each clip is cut into time slices by the model itself; each slice is one audio position. At every
layer (the ears plus every language layer) we save:

1. **The whole-clip average**, as in the hearing test.
2. **Moment by moment:** averages over runs of consecutive audio positions covering **about half a
   second**. The number of positions is measured per model, not assumed (Daniel: "check the
   positions dynamically"): the first clip is read once to count that model's positions per
   second, and the window is half a second's worth, rounded. Both numbers are saved with the run.
3. **The takeaway positions:** every position after the audio. In the language layers a position
   only sees earlier ones, so these are the first positions that have heard the whole clip, up to
   the point where the model would start answering. For Qwen2-Audio the prompt comes after the
   audio; for Audio Flamingo the prompt comes before it, so its takeaway positions are only the
   chat template's closing tokens.
4. **The word positions before the audio**, unpooled. These have heard no sound, so they are pure
   word rows from the same pass. Saved for a later check of whether sound rows and word rows sit
   differently inside the model (Daniel's encoding question); **not analysed in this step**
   (saving them: Claude).

`extract.py --moments-seconds 0.5` saves items 2 to 4 (tested on CPU against tiny copies of both
model classes; not yet run on the real weights).

## 7. How the directions are built and scored

Same method as Anthropic's emotions paper (average many examples, subtract a reference), with the
reference changed:

- **Each emotion's direction is built separately** (Daniel): that emotion's average reading minus
  the reference's average reading.
- **The "this is music" part is stripped out** (Daniel): it is one arrow, the average of the 15
  neutral clips minus silence's average (one average: Claude proposed, Daniel chose). From each emotion's direction we
  remove exactly as much of that arrow as the direction contains, and keep everything else
  (stripping: Claude; Daniel agreed). Not a straight subtraction of the music arrow: that would
  make the silence reference identical to the neutral reference, and the comparison of
  references below could not show a difference. Cost, named in advance: any part of an emotion
  that genuinely points along "music" is stripped too; a large disagreement between the silence
  and neutral references is the sign of it.
- **Difference from the paper:** the paper also stripped rather than subtracted, but it removed
  several directions (the main ways emotionally neutral text varies, enough to cover half of its
  variation). We remove one, because directions taken from only 15 clips would mostly describe
  those clips rather than music in general. (Source: our notes on the paper's "Finding emotion
  vectors" section, 2026-10-21.)
- **Three references, checked against each other** (Daniel): silence, the flat tone, the neutral
  clips. If they give similar directions, the choice of reference does not drive the result. If
  they differ, that is reported. **Primary for the pass marks: silence**, compared against the
  neutral clips to see whether the result differs (Claude's proposal; Daniel confirmed 2026-11-01).
- **Scoring** (Daniel): every clip is scored on all five directions, each built without that clip;
  the strongest is the model's answer, correct only if it is the clip's chosen emotion.
- **Layers:** every layer gets a score. The best layer becomes the primary one, fixed for the
  measuring part (Claude; Daniel agreed). Picking the best of many layers can win by luck, so the
  luck level is measured the same way: shuffle the labels, pick the best layer again, repeat.

## 8. Order the results are read in (Claude proposed at Daniel's request)

1. Happiness vs sadness: does anything register at all? Fail → stop.
2. Happiness vs fear: both high-energy, so loudness cannot carry it.
3. Tenderness vs sadness: the same check at low energy.
4. All five together.
5. When and where: the timeline and layer profile (section 10).
6. Opposites and clusters: compare every pair of directions.

## 9. Pass marks (Daniel), fixed after approval

- Each two-emotion check: **2/3** (luck = 1/2).
- All five: **55%** (luck = 1/5). Daniel: "Don't stop the program too early."
- No pass mark for questions 2 and 3; they are read as descriptions.

## 10. Predictions (Daniel), before any reading

- **Detection:** over half right. "I assume the model understands fear as a concept already." If
  these models had Claude-like training, at least 70%.
- **When:** the overall emotional takeaway lands "after climax, after everything has revealed
  itself," so the takeaway positions should carry the emotion more clearly than the whole-clip
  average.
- **Opposites:** happiness and sadness, built separately, point in roughly opposite directions,
  "the same concept going in opposite directions." Also tested on anger vs fear. Similar
  directions (close to +1) are clusters.

The measuring part's prediction (near zero, against the silence run) is recorded with that part.

## 11. Interpretation key

- **Sound check:** if silence sits no farther from the clips than they sit from each other, the
  sound is not registering; stop.
- **Detection:**
  - Step 1 fails → no emotion signal found; stop.
  - Step 1 passes, steps 2 and 3 fail → loudness posing as emotion. That is still a finding.
  - Steps 2 and 3 pass → something beyond energy; step 4 shows how far.
  - In step 4, mistakes inside an energy group (fear called anger, sadness called tenderness)
    point to energy; correct picks inside a group point past it.
- **Blur (Daniel):** averaging blurred the signal if every clip's whole-clip reading comes out
  "grayscale" (two or three emotions weakly, no clear primary) while the moment-by-moment or
  takeaway readings show clear ones. If both are grayscale, there is no signal to blur.
- **Opposites:** the comparison score runs from −1 (exactly opposite) through 0 (unrelated) to +1
  (same direction). Clearly below 0 supports the opposites prediction. Two traps are named in
  advance: subtracting the other emotions would push directions apart by construction (avoided by
  building against a reference), and a silence reference would pull them together because all
  music shares "this is music" (handled by removing that part).

## 12. Visual

A timeline for each clip in the style of the emotions paper's highlighted text: one row per
emotion, each time slice shaded pale to strong orange by how active that direction is, with the
waveform underneath (Daniel). Built into the Scratch Paper Viewer. It stays on the laptop: the
film audio is private, so this version cannot be hosted publicly.

## 13. Definition-gate tally (draft)

**Daniel:** the detection question, the when-and-where question, the opposites question,
differences from the paper, five emotions, music only, prompt, no generation, both models, obvious
clips first, target-match scoring, equal counts in random order, 20 per emotion mostly clear,
loudness kept with ablation as the follow-up, two clean parts, happiness vs sadness first, inside
readout as primary, directions built separately, music part removed, three references compared,
the blur rule, the timeline visual, 2/3, 55%, three predictions, the recognition risk, its later
check by inside words rather than asking, the clip-by-clip listening check as the follow-up = 30.

**Both (half each; Claude proposed, Daniel decided):** silence as primary reference, one average
music arrow rather than several = 2.

**Claude:** clip split, order of the remaining steps, primary layer chosen on the decoding clips,
the loudness check by energy group, the 15/5 clear split, the neutral-clip rule (delegated), flat
tone pitch and level, half-second window counted per model, stripping rather than subtracting,
saving the pre-audio word positions = 10.

Daniel 31, Claude 11. Daniel above half.

## Changed in draft 3 (2026-11-01, Daniel's answers)

- Silence confirmed as the primary reference, compared against the neutral clips (section 7).
- Window: about half a second, positions counted per model from the first clip (section 6).
- "Music part removed" made exact: one average arrow from the 15 neutral clips, stripped, not
  subtracted; difference from the paper stated (section 7).
- Recognition named as a risk; asking the model ruled out as leading; inside-words check later,
  only if emotions are captured; clip-by-clip listening check as the second follow-up (section 4).
- Word positions before the audio saved for a later encoding check, not analysed here (section 6).

## Approval

Approved by Daniel, 2026-11-01: "ok, good to go!" Pass marks, predictions and the
interpretation key are fixed from this commit. Nothing had been run before it.
