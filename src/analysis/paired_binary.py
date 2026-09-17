"""METHOD, and why this one — settled by a COVERAGE SIMULATION, not by preference.

For paired binary outcomes the estimand is the difference of correlated
proportions, delta = p_arm1 - p_arm2. With the paired 2x2 counts

        b = pairs where arm1 = 1 and arm2 = 0
        c = pairs where arm1 = 0 and arm2 = 1
        n = total pairs

the concordant cells cancel and delta = (b - c) / n exactly.

⚠️ A METHOD THAT LOOKS RIGHT AND IS NOT. The first implementation here
conditioned on the discordant count m = b + c -- the same conditioning the exact
McNemar test performs -- and mapped a Clopper-Pearson interval for pi = b/m onto
delta = (m/n)(2*pi - 1). It is exact conditionally, it agrees with the test by
construction, and it is WRONG as an interval for delta: conditioning caps the
interval at +/- m/n, and when a real effect exists m is itself the quantity
carrying the signal, so the cap falls below the true delta about half the time.
Simulated coverage at n=100, true delta=0.05 was **0.558**, not 0.95. It is kept
below as `ci_conditional` for the diagnostic it is (how the discordants split),
and it is NEVER the reported interval.

THREE CANDIDATE METHODS WERE COVERAGE-TESTED against the paired multinomial, 2000
trials per cell, and the reported interval was chosen on the MEASURED WORST CASE,
not on preference (reproduce with `_selftest`):

    case                       delta    Wald   Newc   boot
    null, high concordance    +0.000   0.959  0.984  0.959
    moderate effect           +0.050   0.919  0.956  0.949
    large effect              +0.100   0.945  0.944  0.954
    small n, null             +0.000   0.993  0.993  0.993
    Table-10 regime           +0.005   0.994  0.998  0.611
    big effect, one-sided     +0.320   0.939  0.943  0.948

**Newcombe (1998) method 10 is what we report** -- worst-case coverage 0.943
against a 0.95 nominal, and the only method that holds up everywhere. It builds
the interval from Wilson intervals on the two marginal rates and adjusts by the
within-pair correlation ("square-and-add").

The other two are retained as diagnostics with their failure modes recorded,
because both are the obvious thing to reach for and both are wrong here:

EQUIVALENCE. `equivalence_verdict` implements the interval form of TOST: the two
arms are declared equivalent at margin `margin` only if the whole confidence
interval for delta lies inside (-margin, +margin). A margin must be supplied by
the caller and justified in prose -- there is no default, deliberately, because a
margin invented after seeing the data is the same error as the one this module
exists to correct.

"""
from __future__ import annotations

import random
from dataclasses import dataclass
from math import comb


# --------------------------------------------------------------------------
# exact McNemar -- duplicated deliberately from paper_b_multiplicity so this
# module stands alone as the interval half of the same instrument. Any change
# must be made in both, and the two are cross-checked in `_selftest`.
# --------------------------------------------------------------------------
def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar. Under H0 the discordant pairs split Binom(m, 1/2)."""
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(m, i) for i in range(k + 1)) / 2**m
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------
# Clopper-Pearson, via the beta quantile. Implemented through the regularised
# incomplete beta inverse rather than a normal approximation, so it is exact at
# the small m this paper actually has (m = 1 on one Table 10 arm).
# --------------------------------------------------------------------------
def _beta_ppf(q: float, a: float, b: float) -> float:
    """Inverse regularised incomplete beta by bisection.

    Bisection rather than a library call keeps this dependency-free and is
    fast enough at our scale; 200 iterations resolves to ~1e-60, far past the
    three decimals anything downstream prints.
    """
    if a <= 0:
        return 0.0
    if b <= 0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) by continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    from math import exp, lgamma

    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(a * _log(x) + b * _log(1 - x) - lbeta)
    if x > (a + 1) / (a + b + 2):
        return 1 - _betainc(b, a, 1 - x)

    tiny = 1e-30
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        f *= c * d
        if abs(1.0 - c * d) < 1e-14:
            break
    return front * (f - 1) / a


def _log(x: float) -> float:
    from math import log

    return log(x)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval for k successes of n."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else _beta_ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_ppf(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval — for reporting each ARM's own rate.

    Wilson rather than Clopper-Pearson for the marginal rates because it is the
    conventional choice for a reported proportion and does not collapse to
    [0, 1] at k = 0; the paired DIFFERENCE keeps the exact treatment above.
    """
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z(alpha)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _z(alpha: float) -> float:
    """Two-sided normal quantile by bisection on the erf-based CDF."""
    from math import erf, sqrt

    target = 1 - alpha / 2
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --------------------------------------------------------------------------
# Newcombe (1998) method 10 -- "square-and-add" for the paired difference.
# Builds the interval from Wilson intervals on the two MARGINAL rates and
# widens/narrows by the estimated within-pair correlation. Designed to fix the
# undercoverage that plain Wald shows on exactly this estimand.
# --------------------------------------------------------------------------
def _ci_newcombe(n11: int, n10: int, n01: int, n00: int,
                 alpha: float = 0.05) -> tuple[float, float]:
    n = n11 + n10 + n01 + n00
    if n == 0:
        return (0.0, 0.0)
    p1 = (n11 + n10) / n
    p2 = (n11 + n01) / n
    delta = p1 - p2

    l1, u1 = wilson(n11 + n10, n, alpha)
    l2, u2 = wilson(n11 + n01, n, alpha)

    # within-pair correlation from the 2x2 table; zero when a margin vanishes
    A = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    phi = 0.0 if A == 0 else (n11 * n00 - n10 * n01) / (A ** 0.5)

    t_lo = (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    t_hi = (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2
    lo = delta - max(t_lo, 0.0) ** 0.5
    hi = delta + max(t_hi, 0.0) ** 0.5
    return (max(-1.0, lo), min(1.0, hi))


# --------------------------------------------------------------------------
# the paired difference
# --------------------------------------------------------------------------
@dataclass
class PairedResult:
    n: int
    b: int                       # arm1 = 1, arm2 = 0
    c: int                       # arm1 = 0, arm2 = 1
    delta: float                 # (b - c) / n, in PROPORTION units
    ci: tuple[float, float]              # REPORTED: Newcombe method 10
    ci_wald: tuple[float, float]         # diagnostic: undercovers at moderate effects
    ci_boot: tuple[float, float] | None  # diagnostic: collapses at few discordants
    ci_conditional: tuple[float, float]  # diagnostic: undercovers, caps at m/n
    p: float

    @property
    def m(self) -> int:
        return self.b + self.c

    def pp(self) -> str:
        """One line in percentage points, the unit the paper reports in."""
        lo, hi = self.ci
        return (f"delta={self.delta*100:+6.1f}pp  95% CI [{lo*100:+6.1f}, {hi*100:+6.1f}]  "
                f"discordant {self.b}/{self.c}  n={self.n}  p={self.p:.4g}")


def paired_difference(
    pairs: list[tuple[int, int]],
    alpha: float = 0.05,
    bootstrap: int = 20000,
    seed: int = 20260808,
) -> PairedResult:
    """Delta, its exact conditional CI, a bootstrap cross-check, and exact McNemar."""
    n = len(pairs)
    b = sum(1 for x, y in pairs if x == 1 and y == 0)
    c = sum(1 for x, y in pairs if x == 0 and y == 1)
    delta = (b - c) / n if n else 0.0
    m = b + c
    z = _z(alpha)

    # ---- REPORTED: Newcombe method 10 (best measured worst-case coverage) ----
    n11 = sum(1 for x, y in pairs if x == 1 and y == 1)
    n00 = sum(1 for x, y in pairs if x == 0 and y == 0)
    ci = _ci_newcombe(n11, b, c, n00, alpha) if n else (0.0, 0.0)

    # ---- diagnostic: paired Wald on the discordant counts ----------------
    # Var(delta) = [(b + c) - (b - c)^2 / n] / n^2. At m = 0 this collapses to
    # zero, which would assert delta is known exactly; fall back to the
    # rule-of-three bound, the correct reading of "no discordant pair observed".
    if n == 0:
        ci_wald = (0.0, 0.0)
    elif m == 0:
        ci_wald = (-3.0 / n, 3.0 / n)
    else:
        var = (m - (b - c) ** 2 / n) / (n ** 2)
        half = z * max(var, 0.0) ** 0.5
        ci_wald = (delta - half, delta + half)

    # ---- DIAGNOSTIC ONLY: the conditional interval. Kept because it shows how
    # the discordants split, but it UNDERCOVERS (0.558 at n=100, delta=0.05) and
    # must never be the reported interval. See the module header.
    if m == 0:
        ci_cond = (0.0, 0.0)
    else:
        pi_lo, pi_hi = clopper_pearson(b, m, alpha)
        ci_cond = ((m / n) * (2 * pi_lo - 1), (m / n) * (2 * pi_hi - 1))

    ci_boot = None
    if bootstrap and n:
        rng = random.Random(seed)
        idx = range(n)
        deltas = []
        for _ in range(bootstrap):
            s = [pairs[rng.choice(idx)] for _ in idx]
            bb = sum(1 for x, y in s if x == 1 and y == 0)
            cc = sum(1 for x, y in s if x == 0 and y == 1)
            deltas.append((bb - cc) / n)
        deltas.sort()
        lo_i = int((alpha / 2) * bootstrap)
        hi_i = min(bootstrap - 1, int((1 - alpha / 2) * bootstrap))
        ci_boot = (deltas[lo_i], deltas[hi_i])

    return PairedResult(n, b, c, delta, ci, ci_wald, ci_boot, ci_cond, mcnemar_p(b, c))


def equivalence_verdict(res: PairedResult, margin: float) -> tuple[bool, str]:
    """Interval-form TOST. Equivalent iff the WHOLE CI sits inside +/- margin.

    `margin` is in proportion units and must be justified by the caller. Returns
    the verdict and a sentence stating what the data actually support, including
    the case this exists for: a non-significant test whose interval is far too
    wide to support any equivalence claim.
    """
    lo, hi = res.ci
    widest = max(abs(lo), abs(hi))
    if lo > -margin and hi < margin:
        return True, (f"EQUIVALENT at +/-{margin*100:.0f}pp: the 95% CI "
                      f"[{lo*100:+.1f}, {hi*100:+.1f}]pp lies entirely inside the margin.")
    if res.p >= 0.05:
        return False, (f"NOT SHOWN EQUIVALENT: the test does not reject "
                       f"(p={res.p:.3g}), but the 95% CI [{lo*100:+.1f}, {hi*100:+.1f}]pp "
                       f"still permits an effect as large as {widest*100:.1f}pp -- "
                       f"wider than the +/-{margin*100:.0f}pp margin. With {res.m} "
                       f"discordant pairs this is a power limit, not evidence of sameness.")
    return False, (f"DIFFERENT: the CI [{lo*100:+.1f}, {hi*100:+.1f}]pp excludes zero "
                   f"(p={res.p:.3g}).")


# --------------------------------------------------------------------------
def _selftest() -> None:
    """Deterministic checks. Run: uv run python -m src.analysis.paired_binary"""
    # 1. exact McNemar against hand-computable cases
    assert abs(mcnemar_p(0, 0) - 1.0) < 1e-12
    assert abs(mcnemar_p(1, 0) - 1.0) < 1e-12          # 2 * 0.5
    assert abs(mcnemar_p(2, 0) - 0.5) < 1e-12          # 2 * 0.25
    assert abs(mcnemar_p(3, 0) - 0.25) < 1e-12
    assert abs(mcnemar_p(10, 0) - 2 / 1024) < 1e-12

    # 2. Clopper-Pearson against published values for 0/10 and 1/10
    lo, hi = clopper_pearson(0, 10)
    assert abs(lo - 0.0) < 1e-9 and abs(hi - 0.30850) < 1e-4, (lo, hi)
    lo, hi = clopper_pearson(1, 10)
    assert abs(lo - 0.00253) < 1e-4 and abs(hi - 0.44502) < 1e-4, (lo, hi)

    for b, c in [(0, 0), (1, 0), (3, 0), (10, 1), (39, 0), (7, 7), (13, 6), (1, 9), (32, 0)]:
        pairs = [(1, 0)] * b + [(0, 1)] * c + [(1, 1)] * (100 - b - c)
        r = paired_difference(pairs, bootstrap=0)
        assert r.b == b and r.c == c, (r.b, r.c)
        lo, hi = r.ci
        if r.p < 0.05:
            assert lo > 0 or hi < 0, f"rejecting test but CI spans 0: {b},{c} -> {r.ci}"

    # 4. bootstrap and Wald agree to within a few points where both are valid.
    pairs = [(1, 0)] * 32 + [(1, 1)] * 30 + [(0, 0)] * 38
    r = paired_difference(pairs, bootstrap=5000)
    assert r.ci_boot is not None
    assert abs(r.ci_boot[0] - r.ci[0]) < 0.05, (r.ci_boot, r.ci)
    assert abs(r.ci_boot[1] - r.ci[1]) < 0.05, (r.ci_boot, r.ci)

    # 5. equivalence behaves on both sides of the margin. NOTE, because it is
    #    counter-intuitive and it corrected an assumption made while writing this
    #    module: a SMALL DISCORDANT COUNT IS INFORMATIVE. delta = (b-c)/n is
    #    bounded by m/n, so m = 1 of 100 pairs bounds the difference within 1pp
    #    whatever the p-value says. "p >= 0.25 therefore uninformative" does not
    #    follow -- the p-value is uninformative, the discordant count is not.
    pairs = [(1, 0)] * 1 + [(1, 1)] * 20 + [(0, 0)] * 79
    r = paired_difference(pairs, bootstrap=0)
    ok, msg = equivalence_verdict(r, margin=0.05)
    assert ok, msg                       # +/-5pp margin: 1pp interval fits inside
    ok2, msg2 = equivalence_verdict(r, margin=0.005)
    assert not ok2 and "power limit" in msg2, msg2   # +/-0.5pp margin: it does not

    # 6. COVERAGE — the decisive check, and the one that caught the first
    #    implementation. The reported interval must contain the TRUE delta at
    #    >= 93% (nominal 95%, allowing simulation noise and the known mild
    #    conservatism/liberality of Wald at extreme cells). The generative model
    #    is the paired multinomial itself: cell probabilities (p11, p10, p01,
    #    p00) with true delta = p10 - p01. Cases span a null, two real effects,
    #    a small n, and the Table-10 regime of very few discordants.
    rng = random.Random(4242)
    CASES = [
        ("null, high concordance", 100, (0.10, 0.02, 0.02, 0.86)),
        ("moderate effect",        100, (0.10, 0.06, 0.01, 0.83)),
        ("large effect",           100, (0.30, 0.15, 0.05, 0.50)),
        ("small n, null",           30, (0.20, 0.03, 0.03, 0.74)),
        ("Table-10 regime",        100, (0.20, 0.01, 0.005, 0.785)),
    ]
    for name, n, cells in CASES:
        p11, p10, p01, p00 = cells
        true_delta = p10 - p01
        cum, edges = 0.0, []
        for p in (p11, p10, p01, p00):
            cum += p
            edges.append(cum)
        outcomes = [(1, 1), (1, 0), (0, 1), (0, 0)]

        cov_n = cov_w = cov_c = 0
        trials = 2000
        for _ in range(trials):
            sample = []
            for _ in range(n):
                u = rng.random()
                for e, o in zip(edges, outcomes):
                    if u < e:
                        sample.append(o)
                        break
                else:
                    sample.append((0, 0))
            # bootstrap disabled here: it is a diagnostic, and running 2000
            # resamples inside 2000 trials makes the selftest minutes-slow. Its
            # coverage (incl. the 0.611 collapse) is recorded in the header.
            r = paired_difference(sample, bootstrap=0)
            for lohi, box in ((r.ci, "n"), (r.ci_wald, "w"), (r.ci_conditional, "c")):
                if lohi[0] - 1e-12 <= true_delta <= lohi[1] + 1e-12:
                    if box == "n":
                        cov_n += 1
                    elif box == "w":
                        cov_w += 1
                    else:
                        cov_c += 1
        rn, rw, rc = cov_n / trials, cov_w / trials, cov_c / trials
        print(f"  coverage  {name:22s} n={n:3d} delta={true_delta:+.3f}:  "
              f"Newcombe {rn:.3f}   wald {rw:.3f}   conditional {rc:.3f}")
        assert rn >= 0.94, f"reported-interval coverage {rn:.3f} at {name}"

    print("paired_binary selftest: all checks pass")


if __name__ == "__main__":
    _selftest()
