"""Statistical helpers: Student-t confidence intervals and paired hypothesis tests."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import scipy.stats


class PairedDiffResult(NamedTuple):
    """Result of paired difference analysis with confidence intervals and p-values."""

    mean_diff: float
    ci_low: float
    ci_high: float
    n: int
    p_ttest: float
    p_wilcoxon: float


def mean_ci(
    values: Sequence[float],
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute sample mean and two-sided Student-t confidence interval.

    Notes
    -----
    NaN values are ignored prior to computation.
    If fewer than 2 finite values remain, returns (mean, nan, nan).
    If 0 finite values remain, returns (nan, nan, nan).

    Parameters
    ----------
    values : Sequence[float]
        Sample observations.
    level : float
        Confidence level, e.g. 0.95 for a 95% confidence interval.

    Returns
    -------
    tuple of (mean, ci_low, ci_high)
        Sample mean, lower CI bound, and upper CI bound.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    valid = arr[finite_mask]

    n = len(valid)
    if n == 0:
        return float(np.nan), float(np.nan), float(np.nan)
    m = float(np.mean(valid))
    if n < 2:
        return m, float(np.nan), float(np.nan)

    se = float(scipy.stats.sem(valid))
    if se == 0.0 or not np.isfinite(se):
        return m, m, m

    h = float(se * scipy.stats.t.ppf((1.0 + level) / 2.0, df=n - 1))
    return m, m - h, m + h


def paired_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    level: float = 0.95,
) -> PairedDiffResult:
    """Compute confidence interval and p-values for paired differences (a - b).

    Matched pairs where either observation is NaN or non-finite are dropped.

    Parameters
    ----------
    a : Sequence[float]
        First condition values (e.g. proposed router).
    b : Sequence[float]
        Second condition values (e.g. baseline router).
    level : float
        Confidence level for Student-t interval (default: 0.95).

    Returns
    -------
    PairedDiffResult
        NamedTuple containing:
        - mean_diff: Mean of (a - b).
        - ci_low: Lower bound of 95% CI on mean difference.
        - ci_high: Upper bound of 95% CI on mean difference.
        - n: Number of valid paired observations.
        - p_ttest: Two-sided paired Student-t test p-value.
        - p_wilcoxon: Two-sided Wilcoxon signed-rank test p-value.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    if len(arr_a) != len(arr_b):
        raise ValueError(f"Length mismatch: len(a)={len(arr_a)} != len(b)={len(arr_b)}")

    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    diffs = arr_a[mask] - arr_b[mask]
    n = len(diffs)

    if n == 0:
        return PairedDiffResult(
            mean_diff=float(np.nan),
            ci_low=float(np.nan),
            ci_high=float(np.nan),
            n=0,
            p_ttest=float(np.nan),
            p_wilcoxon=float(np.nan),
        )

    m = float(np.mean(diffs))

    if n < 2:
        return PairedDiffResult(
            mean_diff=m,
            ci_low=float(np.nan),
            ci_high=float(np.nan),
            n=n,
            p_ttest=float(np.nan),
            p_wilcoxon=float(np.nan),
        )

    se = float(scipy.stats.sem(diffs))
    if se == 0.0 or not np.isfinite(se):
        ci_low = m
        ci_high = m
        p_t = 1.0 if m == 0.0 else 0.0
    else:
        h = float(se * scipy.stats.t.ppf((1.0 + level) / 2.0, df=n - 1))
        ci_low = m - h
        ci_high = m + h
        t_res = scipy.stats.ttest_rel(arr_a[mask], arr_b[mask])
        p_t = float(t_res.pvalue)

    # Wilcoxon signed-rank test (guard against all zero or zero differences)
    non_zero = diffs[diffs != 0.0]
    if len(non_zero) == 0:
        p_w = 1.0
    else:
        try:
            w_res = scipy.stats.wilcoxon(diffs)
            p_w = float(w_res.pvalue)
        except Exception:
            p_w = float(np.nan)

    return PairedDiffResult(
        mean_diff=m,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        p_ttest=p_t,
        p_wilcoxon=p_w,
    )

