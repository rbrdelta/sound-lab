"""How many clips do we need, given how well the detector hears? (Monte Carlo, numpy only.)

The picture: a later experiment compares two conditions (say, the model heard tense music vs
calm music before a task) and asks whether the "fear" state shows up more often in one of them.
The detector reads each trial and says fear / not fear -- but it is right only some of the time
(its hearing accuracy from probe.py). Every wrong reading pulls the two conditions' counts
toward each other, so a weaker detector needs more clips to see the same real shift.

Model of the world (stated so it can be argued with):
  - baseline: in condition A the state is truly present in `--base-rate` of trials;
    in condition B in `--base-rate + --effect` of trials.
  - the detector is equally good on both classes: right with probability = accuracy.
  - detection rule (a commitment device, fixed before looking, not a verdict on truth): a
    two-sided two-proportion test at the 5% level. "Caught" = that test fires.
  - we want to catch a real effect in `--power` (default 8 out of 10) of repeated experiments.

Usage:
  python simulate_clips.py                      # defaults: accuracies 0.67 0.75 0.80 0.90
  python simulate_clips.py --effect 0.15 --accuracies 0.7 0.85
"""
from __future__ import annotations

import argparse
from math import erf, sqrt

import numpy as np


_erf = np.vectorize(erf)


def _norm_two_sided_p(z: np.ndarray) -> np.ndarray:
    return 1 - _erf(np.abs(z) / sqrt(2))


def catch_rate(n: int, accuracy: float, base_rate: float, effect: float, sims: int, rng,
               alpha: float = 0.05) -> float:
    """Share of simulated experiments (n clips per condition) where the test fires."""
    p_a, p_b = base_rate, base_rate + effect
    # Simulate the truth, then the reading (right with prob=accuracy, flipped otherwise).
    # Expected reported rate = p*acc + (1-p)*(1-acc), so the visible gap is (2*acc-1)*effect.
    true_a = rng.random((sims, n)) < p_a
    true_b = rng.random((sims, n)) < p_b
    right_a = rng.random((sims, n)) < accuracy
    right_b = rng.random((sims, n)) < accuracy
    read_a = np.where(right_a, true_a, ~true_a).mean(1)
    read_b = np.where(right_b, true_b, ~true_b).mean(1)
    pooled = (read_a + read_b) / 2
    se = np.sqrt(np.maximum(pooled * (1 - pooled) * 2 / n, 1e-12))
    z = (read_b - read_a) / se
    p = _norm_two_sided_p(z)
    return float(np.mean(p < alpha))


def clips_needed(accuracy, base_rate, effect, power=0.8, sims=2000, seed=0, max_n=5000):
    """Smallest n per condition (on a doubling-then-bisect search) reaching the target catch rate."""
    rng = np.random.default_rng(seed)
    lo, hi = 2, 4
    while catch_rate(hi, accuracy, base_rate, effect, sims, rng) < power:
        lo, hi = hi, hi * 2
        if hi > max_n:
            return None
    while hi - lo > max(1, lo // 50):
        mid = (lo + hi) // 2
        if catch_rate(mid, accuracy, base_rate, effect, sims, rng) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accuracies", type=float, nargs="+", default=[0.67, 0.75, 0.80, 0.90])
    p.add_argument("--base-rate", type=float, default=0.5)
    p.add_argument("--effect", type=float, default=0.20, help="true shift in how often the state is present")
    p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--sims", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    print(f"Suppose the real effect is: the state shows up in {a.base_rate:.0%} of trials in one condition "
          f"and {a.base_rate + a.effect:.0%} in the other.")
    print(f"How many trials per condition to spot that difference in {a.power:.0%} of repeat experiments?\n")
    ref = clips_needed(1.0, a.base_rate, a.effect, a.power, a.sims, a.seed)
    for acc in [1.0] + list(a.accuracies):
        n = ref if acc == 1.0 else clips_needed(acc, a.base_rate, a.effect, a.power, a.sims, a.seed)
        seen = (2 * acc - 1) * a.effect
        tag = "a perfect detector (reference)" if acc == 1.0 else f"a detector right {acc:.0%} of the time"
        if n is None:
            print(f"  With {tag}: more than 5000 per condition -- effectively out of reach.")
            continue
        extra = "" if acc == 1.0 or not ref else f"  ({n / ref:.1f}x the perfect-detector count)"
        print(f"  With {tag}: about {n} per condition{extra}. "
              f"The {a.effect:.0%} real shift looks like {seen:.0%} through this detector.")
    print("\nThe catch: mistakes don't just add noise, they shrink the visible difference "
          "(by a factor of 2 x accuracy - 1), and the needed count grows roughly with the square of that.")


if __name__ == "__main__":
    main()
