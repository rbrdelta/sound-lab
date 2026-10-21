# Real music, decoding step — registration (DRAFT, not yet committed)

Draft written 2026-10-21 from Daniel's answers in that day's design talk; his words are on file
verbatim in the private notebook (`shadow-ledger/NEXT-SESSION.md`, "Real-music test — Daniel's
first design inputs" onward). Becomes the registration once Daniel has read it. Nothing has been
run.

**Role:** instrumentation. This is the first of two clean parts (Daniel): **decoding**, then
**measuring** (the ten-word experiment). It finds the emotion directions that the measuring part
and the later switch-an-emotion-on step depend on. It does not measure behavior.

1. **Throughline.** The hearing tests (2026-10-06) showed both models' ears pick up physical change
   over time (tempo, roughness, loudness) almost perfectly and do not pick up major vs minor.
   The next question is whether anything inside the model tracks the emotion people hear in real
   music. That is the bridge (Daniel): without emotion directions there is nothing to compare the
   ten words against and nothing to switch on.
2. **Question (Daniel).** When a real music clip is played to the model, can a readout of its
   inside tell which of five emotions listeners rated the clip as? Start with the obvious clips
   "to see if it can even detect the emotion first."
3. **The one variable that moves:** the emotion the clip was chosen for: anger, fear, happiness,
   sadness, tenderness (Daniel: five, the film set's rated emotions). Equal clips per emotion,
   random order (Daniel).
4. **Held fixed:** prompt "Listen to this audio clip." (Daniel); no answer generated, one forward
   pass per clip, read before any output, as in the hearing test (Daniel); 16 kHz mono; model
   weights, bf16, eager attention, seed 0. Both models, Qwen2-Audio-7B-Instruct and Audio
   Flamingo Next (Daniel). One silence clip for the sound check.
   **Cannot be held fixed:** everything else about real music: instruments, tempo, density, the
   film it came from. Spread across emotions only by the random draw.
   **Loudness differs by emotion** (measured from the files, decoding clips, average level):
   anger −20.4 dB, happiness −21.6, fear −26.9, sadness −28.0, tenderness −28.6. Named risk:
   *loudness posing as emotion.* **Kept as recorded** (Daniel: "Loudness is a dimension we
   shouldn't eliminate"). If the results are inconclusive, a follow-up run levels every clip to the
   same average loudness to see what loudness was contributing (Daniel).
5. **Clips.** Film soundtracks, Eerola & Vuoskoski 2011 (OSF p6vkg); audio kept private, never
   in this repo. Built by `make_film_clips.py`.
   - Decoding: the larger first set (360 clips, 30 chosen for each emotion), minus every clip that
     is the same excerpt as a measuring-part clip, then **20 per emotion = 100 clips, mostly clear
     but some unclear** (Daniel): 15 where listeners rated the chosen emotion highest, 5 where
     they rated another emotion highest (split: Claude). The clear-only 75 are also scored
     separately, as a side reading.
   - Measuring part: the 25 clips in the second set's five "high" groups (5 per emotion).
     **No clip is in both** (Claude's proposal; Daniel: use the clips "for whatever helps us
     bring the most value"). The second set's published `link` column was confirmed against the
     audio for all 25: each linked clip is the best match among all 360 by a wide margin.
   - Not checkable: the first set's track list is not published, so two different excerpts of
     the same film cue could sit on both sides. Stated as a limitation.
6. **Readout.** Music alone (Daniel). For each emotion, average the model's inside readings over
   its clips; a clip is called the emotion whose average it sits closest to. A clip counts as
   correct only if that is its chosen emotion (Daniel). Every clip is scored by a readout built
   without it. Primary signal = the inside readout (Daniel). Every layer gets a score, and the
   best layer becomes the primary one, fixed for the measuring part (Claude; Daniel agreed). Because
   choosing the best of many layers can win by luck, the luck level is measured the same way:
   shuffle the emotion labels, pick the best layer again, repeat.
7. **Order the results are read in** (Claude proposed at Daniel's request; one run collects
   everything):
   1. Happiness vs sadness: does anything register at all? Fail → stop.
   2. Happiness vs fear: both high-energy, so loudness cannot carry it.
   3. Tenderness vs sadness: the same check at low energy.
   4. All five together.
8. **Pass marks (Daniel).** Each two-emotion check: 2/3 (luck = 1/2). All five: 55% (luck = 1/5).
   Daniel: "don't stop the program too early." Fixed after this.
9. **Predictions (Daniel), before any reading.** Over half right on the readout: "I assume the
   model understands fear as a concept already." If these models had Claude-like training,
   at least 70%. (His prediction for the measuring part, near zero, is recorded there.)
10. **Interpretation key.**
    - Silence sits no farther from the clips than they sit from each other → sound not
      registering; stop.
    - Step 1 fails → no emotion signal found; stop.
    - Step 1 passes, steps 2 and 3 fail → loudness posing as emotion.
    - Steps 2 and 3 pass → something beyond energy. Step 4 shows how far.
    - In step 4, mistakes inside an energy group (fear called anger, sadness called tenderness)
      point to energy; correct picks inside a group point past it.
11. **Definition-gate tally (draft).** Daniel: question, five emotions, music only, prompt, no
    generation, both models, obvious clips first, target-match scoring, equal counts in random
    order, two clean parts, happiness vs sadness first, inside readout as primary, 2/3, 55%,
    prediction, 20 per emotion mostly clear, loudness kept with a leveled follow-up = 17. Claude:
    clip split, order of the remaining steps, primary layer chosen on the decoding clips, loudness
    check, 15/5 clear split = 5. Daniel above half.
