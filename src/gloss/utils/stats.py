"""Small-sample statistics: the Wilson interval every count in this benchmark is quoted with.

Copied from DJRHails/touchstone (``lab.evaluation.metrics.wilson_ci``). gloss reports rates
over tens of turns, where the normal approximation is wrong in the direction that flatters
the claim, so every binomial proportion carries a 95% Wilson interval.
"""

from __future__ import annotations

Z_95 = 1.96


def wilson_ci(successes: int, n: int) -> tuple[float, float, float]:
    """Wilson 95% interval for a binomial rate: ``(point, lo, hi)``, all NaN when ``n == 0``."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = successes / n
    denominator = 1 + Z_95 * Z_95 / n
    centre = (point + Z_95 * Z_95 / (2 * n)) / denominator
    half = Z_95 * ((point * (1 - point) / n + Z_95 * Z_95 / (4 * n * n)) ** 0.5) / denominator
    # At p == 0 or p == 1 the bound lands on p exactly in real arithmetic, but float rounding
    # can leave it a hair on the wrong side; clamp so the interval contains the point estimate.
    return (point, min(point, max(0.0, centre - half)), max(point, min(1.0, centre + half)))


def rate_with_ci(successes: int, n: int) -> str:
    """``'3/40 = 8% [2-20%]'`` — a count, its rate, and the 95% Wilson interval, for tables."""
    if n == 0:
        return f"{successes}/0 = n/a"
    point, low, high = wilson_ci(successes, n)
    return f"{successes}/{n} = {point:.0%} [{low:.0%}-{high:.0%}]"
