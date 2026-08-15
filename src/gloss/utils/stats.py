"""Interval estimates for the binomial rates gloss reports.

Wilson rather than normal-approximation: these rates sit near 0 and 1 at small n, exactly where
the Wald interval runs off the end of [0, 1] and reports zero width for a 0/n cell.
"""

from __future__ import annotations

import math

_Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_ci(successes: int, total: int, *, z: float = _Z_95) -> tuple[float, float, float]:
    """Point estimate and 95% Wilson score interval for ``successes``/``total``.

    Args:
        successes: number of trials that succeeded; must be in ``[0, total]``.
        total: number of trials. Zero returns all zeros — a no-data cell, which callers must
            report as such rather than as a rate of 0.
        z: normal quantile for the desired coverage.

    Returns:
        ``(point, low, high)``, each in ``[0, 1]``.

    Raises:
        ValueError: if ``successes`` is negative or exceeds ``total``, which would silently yield
            an interval outside [0, 1].
    """
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if not 0 <= successes <= max(total, 0):
        raise ValueError(f"successes={successes} out of range for total={total}")
    if total == 0:
        return 0.0, 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
    # The Wilson interval always contains the point estimate, so a bound on the wrong side of it is
    # float error, not statistics: at 10/10 the upper bound divides out to 0.9999999999999999. Clamp
    # to the unit interval AND to bracket the point, or a caller asserting low <= point <= high
    # (or rendering "1.00 [0.72, 1.00]") sees an impossible interval.
    low = min(max(0.0, (centre - spread) / denominator), proportion)
    high = max(min(1.0, (centre + spread) / denominator), proportion)
    return proportion, low, high
