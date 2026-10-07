# Real-music sources with human emotion labels

Fact-finding, 2026-10-06. Nothing downloaded except two small text files (the film-soundtrack readme and one 9.6 KB ratings CSV) so the label counts below could be checked. "Verified" = read at the cited URL today. "Unverified" = from memory or a search snippet only.

What the next step needs: excerpts rated by listeners as (a) fearful/negative, (b) calm/peaceful, and (c) high-intensity but positive (happy, triumphant, exciting). Group (c) separates a detector that reads intensity (arousal) from one that reads positive-vs-negative (valence).

---

## 1. Film soundtracks set (Eerola & Vuoskoski 2011): exists, verified

- **Paper:** Eerola, T. & Vuoskoski, J. K. (2011). A comparison of the discrete and dimensional models of emotion in music. *Psychology of Music, 39(1)*, 18-49. https://doi.org/10.1177/0305735610362821
- **Hosting:** OSF project "Music and emotion stimulus sets consisting of film soundtracks", https://osf.io/p6vkg (metadata read through https://api.osf.io/v2/nodes/p6vkg/). The original jyu.fi page no longer responds. The readme says: "the page has become unresponsive and difficult to maintain. For this reason, we are now depositing the files to Open Science Framework."
- **Audio is included** (MP3, "moderate quality compression"), not just references. Files (sizes from the OSF file listing):
  - `Set1.zip` 148.6 MB: **360 excerpts**, about 15 s each, `001.mp3` to `360.mp3`. The readme calls it the piloting set with "fewer ratings per track".
  - `Set2.zip` 35.7 MB: **110 excerpts**, about 15 s each. The readme calls it the "more reliable and commonly utilised source".
  - `1min.zip` 29.3 MB: **16 excerpts**, 45 to 77 s long (mean 57 s): 4 scary, 4 happy, 4 sad, 4 tender, taken from Set 2.
  - `mean_ratings_set1.csv`, `mean_ratings_set2.csv`, `set1_tracklist.csv`, `set2_tracklist.csv`, `readme.md`, `visualise_ratings.R`.
- **Labels: confirmed as you remembered.** Mean listener ratings on a 1 (minimal) to 9 (maximal) scale.
  - Set 1 columns: `valence, energy, tension, anger, fear, happy, sad, tender, TARGET`.
  - Set 2 adds `Beauty, Liking, soundtrack, link`.
  - The ratings are means only. Per-listener ratings are not in the files I read.
- **How Set 2 is balanced (checked in the CSV):** 22 target groups of 5 excerpts each. Every discrete emotion (anger, fear, happy, sad, tender) has a HIGH and a MODERATE group. Each of valence, energy and tension has POS and NEG poles, each at HIGH and MODERATE.
- **Group means in Set 2, computed from the CSV** (V = valence, E = energy, T = tension, F = fear):
  - FEAR_HIGH: V 2.76, E 6.90, T 8.24, F 6.64. This is the fearful/negative group.
  - HAPPY_HIGH: V 7.89, E 8.14, T 4.73, F 1.11. This is high-intensity positive, the group that separates the two detectors.
  - ENERGY POS HIGH: V 7.10, E 7.86, F 1.21. Another high-intensity positive group.
  - TENDER_HIGH: V 7.23, E 3.63, T 2.30. This is calm/peaceful.
  - TENSION NEG HIGH (low-tension end): V 6.49, E 3.42, T 2.91. Also calm.
  - Counts across all 110: 19 excerpts with V >= 6 and E >= 6; 19 with F >= 5; 14 with V >= 5.5, E <= 4 and T <= 4.
  - The fear and happy groups are both high-energy (6.9 and 8.1), so they differ mainly in valence. That is the contrast the next step needs.
- **License:** OSF records the license as **"CC-By Attribution 4.0 International"** (https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/). The readme states the purpose: "provided here in order to share the stimulus materials for academic research." Caveat: the audio is excerpted from commercial film scores. The CC-BY tag was applied by the authors, and it is **unverified** whether they could license the underlying recordings. Using the clips privately for research is safe. Do not redistribute the audio (for example, in a public repo). Redistributing ratings plus track indices is fine.
- **Download:** OSF web UI, or `https://osf.io/download/<file-guid>/`. The readme is `4wzc9`, Set 2 ratings `d23z6`, Set 1 ratings `g2h8w`. Free, no account needed (the downloads worked without one).

## 2. Alternatives

| Dataset | Size / excerpt | Labels | Audio? | License (quoted) | Download |
|---|---|---|---|---|---|
| **EMOPIA** (2021) | 1,087 clips from 387 pop-piano songs; mean 32-41 s per quadrant | 4 valence/arousal quadrants (Q1 high V high A 250, Q2 low V high A 265, Q3 low V low A 253, Q4 high V low A 310), 4 annotators | **No.** "After the raw audio are crawled and put in audios/raw, you can use this script to get audio clips" (YouTube). MIDI included. | "Creative Commons Attribution 4.0 International" | https://zenodo.org/records/5257995 (32 MB, labels+MIDI) |
| **DEAM / MediaEval Emotion in Music** (2016) | 1,802: 1,744 x 45 s clips + 58 full songs | Valence + arousal only: per-second (-10 to 10) and whole-clip (9-point). At least 10 raters per clip (2013-14), 5 (2015). No discrete fear/calm. | **Yes**, 1.3 GB (Free Music Archive, Jamendo, MedleyDB) | Audio: "royalty-free (Creative Commons license enables us to redistribute the content)"; annotations: "released under Non Commercial Creative Commons (BY-NC)" | https://cvml.unige.ch/databases/DEAM/ ; paper https://pmc.ncbi.nlm.nih.gov/articles/PMC5345802 |
| **PMEmo** (2018) | 794 pop songs, chorus excerpts; 457 raters | Valence + arousal, whole-clip and per-0.5 s; skin-conductance (EDA) recordings too | **Yes**, chorus MP3s (~1.3 GB total) via Google Drive | Repo says MIT (applies to the repo). Licensing of the audio itself (commercial chart songs) is not stated, so **unverified**. Citation requested. | https://github.com/HuiZhangDB/PMEmo |
| **Emotify** (2015) | 400 x 1-min excerpts, 4 genres | 9 GEMS categories (a music-specific emotion scale): amazement, solemnity, tenderness, nostalgia, **calmness**, power, **joyful activation**, **tension**, sadness. Emotion the listener *felt*, not the emotion they heard in the music. No fear category. | **Yes**, emotifymusic.zip 363 MB | Only: "If you decide to use this dataset in your work we kindly ask you to cite the paper." No formal license. | https://www2.projects.science.uu.nl/memotion/emotifydata/ |
| **MERGE** (2024) | 3,554 audio x 30 s (balanced version 3,232) | 4 valence/arousal quadrants + continuous valence/arousal. Labels come from "User-generated mood tags extracted from the AllMusic platform", not listening tests, so they are weaker as human-listener labels. | **Yes**, 1.1-1.2 GB | "Creative Commons Attribution Non Commercial 4.0 International" | https://zenodo.org/records/13939205 ; https://arxiv.org/abs/2407.06060 |
| **MTG-Jamendo mood/theme** | 18,486 full tracks (at least 30 s) | 59 mood/theme tags, among them calm, relaxing, happy, epic, energetic. The tags come from uploaders, not listener ratings. A "scary"/"dark" tag was not confirmed on the page I read. | Yes, but **152 GB** (46 GB low quality) | Metadata "CC BY-NC-SA 4.0"; "made available solely for non-commercial research and academic use"; per-track CC audio licenses | https://github.com/MTG/mtg-jamendo-dataset |
| **Memo2496** (2025) | 2,496 instrumental pieces; 30 expert annotators | Valence/arousal | Unverified (figshare returned 403) | Unverified | https://figshare.com/articles/dataset/Memo2496/25827034 |
| **CAL500** | 500 songs | Has emotion tags, but I did not check them | Not checked | Not checked | Not researched. Old and small, low priority. |

## 3. Recommendation

The task is to find clean examples of three groups: fearful, calm, and high-intensity positive. All of them should come from human listener ratings and be usable for non-commercial research.

| Source | Fear/negative | Calm/peaceful | High-intensity positive | Listener ratings? | License fit | Verdict |
|---|---|---|---|---|---|---|
| **Film soundtracks Set 2** | Yes: FEAR_HIGH/MOD (10), V-NEG (10) | Yes: TENDER (10), low-tension and low-energy groups | Yes: HAPPY_HIGH (E 8.1, V 7.9), ENERGY POS | Yes, mean 1-9 ratings for each discrete emotion and each dimension | CC-BY 4.0 tag, academic purpose stated; keep audio private | **Best fit.** Purpose-built, balanced, explicit fear rating, 36 MB |
| Film soundtracks Set 1 | Yes (360 pool) | Yes | Yes | Yes, fewer raters | Same | Use to scale beyond 110 if needed |
| EMOPIA | Q2 = high-arousal negative (tense/angry, not specifically fear) | Q4 | Q1 | Yes (4 annotators, quadrant only) | CC-BY 4.0, but the audio has to be pulled from YouTube | Good second source, piano only, no fear scale |
| DEAM | Low V / high A region | Low A / high V | High A / high V | Yes, many raters, continuous | Audio CC, labels BY-NC | Good for a continuous valence/arousal check; no discrete fear |
| Emotify | No fear | Calmness | Joyful activation, power | Yes (felt emotion) | Cite only | Useful for calm and positive-intensity, not fear |
| MERGE / MTG-Jamendo | Weak | Weak | Weak | No (tags) | NC | Not recommended for this step |

**Bottom line:** Film soundtracks Set 2 (36 MB, 110 clips of 15 s) contains all three needed groups, labelled by listeners, with fear rated directly. Its fear and happy groups are both high-energy, so they differ mainly in valence. That separates a detector reading intensity from one reading positive-vs-negative. DEAM or EMOPIA are the natural second sources to check that a result holds on music that isn't film music.
