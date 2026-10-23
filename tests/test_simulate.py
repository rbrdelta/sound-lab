"""CPU tests for simulate_clips.py: the clip-count estimate behaves the way its docstring says."""
import numpy as np

import simulate_clips as sim


def test_no_real_effect_fires_about_five_percent_of_the_time():
    rng = np.random.default_rng(0)
    rate = sim.catch_rate(200, 0.9, 0.5, 0.0, 4000, rng)
    assert 0.03 < rate < 0.07


def test_a_weaker_detector_needs_more_clips():
    perfect = sim.clips_needed(1.0, 0.5, 0.2, sims=1000)
    good = sim.clips_needed(0.9, 0.5, 0.2, sims=1000)
    weak = sim.clips_needed(0.75, 0.5, 0.2, sims=1000)
    assert perfect < good < weak
    # mistakes shrink the visible gap by (2 x accuracy - 1); the count grows with its square
    assert 2.5 < weak / perfect < 6.5


def test_a_coin_flip_detector_is_out_of_reach():
    assert sim.clips_needed(0.5, 0.5, 0.2, sims=200, max_n=300) is None
