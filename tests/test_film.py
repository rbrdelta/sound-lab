"""CPU tests for make_film_clips.py's selection rules, on made-up rating rows (the film audio and
ratings are private and never needed here)."""
import numpy as np

import make_film_clips as fc

EMO = fc.EMOTIONS


def rating(number, target, top, top_score=7.0, others=2.0, **extra):
    r = {"number": str(number), "TARGET": target, "valence": "5", "energy": "5", "tension": "5"}
    for e in EMO:
        r[e] = str(top_score if e == top else others)
    r.update({k: str(v) for k, v in extra.items()})
    return r


def set1(n_clear=10, n_unclear=4):
    rows, n = [], 1
    for i, e in enumerate(EMO):
        for _ in range(n_clear):
            rows.append(rating(n, e.upper(), e)); n += 1
        for _ in range(n_unclear):
            rows.append(rating(n, e.upper(), EMO[(i + 1) % 5])); n += 1
    return rows


def test_clear_means_listeners_rated_the_chosen_emotion_highest():
    assert fc.clear(rating(1, "FEAR", "fear"))
    assert not fc.clear(rating(1, "FEAR", "anger"))


def test_decoding_pick_is_balanced_excludes_main_clips_and_is_seeded():
    s1 = set1()
    main_links = {"001", "002", "015"}  # two clear anger clips, one clear fear clip
    dec, pool = fc.pick_decoding(s1, main_links, 6, 2, seed=0)
    assert len(dec) == 40 and len(pool["anger"]) == 12 and len(pool["fear"]) == 13
    assert not {r["number"].zfill(3) for r in dec} & main_links
    for e in EMO:
        mine = [r for r in dec if r["TARGET"].lower() == e]
        assert sum(fc.clear(r) for r in mine) == 6 and sum(not fc.clear(r) for r in mine) == 2
    again, _ = fc.pick_decoding(s1, main_links, 6, 2, seed=0)
    other, _ = fc.pick_decoding(s1, main_links, 6, 2, seed=1)
    assert [r["number"] for r in again] == [r["number"] for r in dec]
    assert [r["number"] for r in other] != [r["number"] for r in dec]
    # emotions are interleaved, not left in blocks
    assert len({r["TARGET"] for r in dec[:10]}) > 1


def test_main_clips_are_the_five_high_groups_only():
    s2 = [{"TARGET": t, "Number": str(i), "link": str(i)} for i, t in enumerate(
        ["FEAR_HIGH", "FEAR_MODERATE", "HAPPY_HIGH", "SAD_HIGH", "TENDER_HIGH", "ANGER_HIGH", "SURPRISE_HIGH"])]
    assert [r["TARGET"] for r in fc.select_main(s2)] == ["FEAR_HIGH", "HAPPY_HIGH", "SAD_HIGH", "TENDER_HIGH", "ANGER_HIGH"]


def test_neutral_rule_boundaries():
    ok = rating(1, "FEAR", "fear", top_score=4.0, others=1.0, valence=3.5, energy=6.5, tension=5)
    assert fc.is_neutral(ok)
    assert not fc.is_neutral(rating(1, "FEAR", "fear", top_score=4.01, others=1.0))  # one emotion above 4
    assert not fc.is_neutral(rating(1, "FEAR", "fear", top_score=3.0, others=1.0, energy=6.6))  # energy not middling
    assert not fc.is_neutral(rating(1, "FEAR", "fear", top_score=3.0, others=1.0, valence=3.4))


def test_same_excerpt_finds_a_shifted_recut_and_rejects_another_clip():
    rng = np.random.default_rng(0)
    sr = fc.SR
    loud = np.repeat(rng.uniform(0.1, 1.0, 40), sr // 2)  # loudness changes every half second
    a = rng.standard_normal(20 * sr) * loud
    b = a[3 * sr: 18 * sr]  # the same music re-cut 3 s later
    c = rng.standard_normal(15 * sr) * np.repeat(rng.uniform(0.1, 1.0, 30), sr // 2)
    assert fc.same_excerpt(a, b) > 0.95
    assert fc.same_excerpt(a, c) < 0.8  # below the script's warning line


def test_rms_db_of_a_full_scale_sine():
    t = np.arange(fc.SR) / fc.SR
    assert abs(fc.rms_db(np.sin(2 * np.pi * 220 * t)) - (-3.01)) < 0.01
