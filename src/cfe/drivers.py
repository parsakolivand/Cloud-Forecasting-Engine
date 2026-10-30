"""Dimension 1: what drives each service item's cost, and which bucket it belongs in.

For each service item we correlate its monthly cost with the overall monthly
requests. The comparison runs per calendar year (January to December), because
pooling several years mixes in price changes, commitments and growth that hide
the month-to-month relationship. An all-history figure is reported alongside
for comparison.

Each item then lands in one bucket, checked in this order:

1. Fixed: cost barely moves (support, tax, licences).
2. Seasonal very high / high / mid: cost follows requests (r >= 0.5).
3. Linear growing / declining: everything else, split by its direction over time.

The Pearson correlation r is also shown on a 0-10 scale (score = r x 10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

FIXED = "Fixed"
GROWING = "Linear: growing"
DECLINING = "Linear: declining"
VERY_HIGH = "Seasonal: very high"
HIGH = "Seasonal: high"
MID = "Seasonal: mid"
BUCKETS = [VERY_HIGH, HIGH, MID, GROWING, DECLINING, FIXED]
SEASONAL = [VERY_HIGH, HIGH, MID]
LINEAR = [GROWING, DECLINING]

ALL_HISTORY = "All history"
SIGNIFICANCE = 0.05  # p-value above this: the correlation could be chance
STEADY_TREND = 0.5  # linear items with a weaker time trend than this are irregular

RESULT_COLUMNS = [
    "service", "service_item", "window", "months", "total_cost", "variation",
    "correlation", "score", "p_value", "trend", "bucket",
]


@dataclass(frozen=True)
class Thresholds:
    """Bucket boundaries. Correlations are r (0-1), variation is std / mean."""

    very_high: float = 0.85
    high: float = 0.70
    mid: float = 0.50
    fixed_variation: float = 0.05


DEFAULT_THRESHOLDS = Thresholds()


def analyze(
    costs: pd.DataFrame,
    requests: pd.DataFrame,
    years: list[int],
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Correlate each service item's cost with requests, per calendar year and overall.

    ``costs`` should already be gap-filled (see ``quality.fill_missing_months``).
    ``years`` are the calendar years with 12 months of both cost and requests.
    """
    merged = costs.merge(requests, on="month", how="inner")
    rows = []
    for (service, item), group in merged.groupby(["service", "service_item"], sort=False):
        group = group.sort_values("month")
        for year in years:
            window = group[group["month"].dt.year == year]
            if len(window) == 12:
                rows.append(_row(service, item, str(year), window, thresholds))
        rows.append(_row(service, item, ALL_HISTORY, group, thresholds))
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _row(service: str, item: str, label: str, window: pd.DataFrame, t: Thresholds) -> dict:
    cost = window["cost"].to_numpy(dtype=float)
    reqs = window["requests"].to_numpy(dtype=float)
    r, p = _pearson(cost, reqs)
    trend, _ = _pearson(cost, np.arange(len(cost), dtype=float))
    variation = _variation(cost)
    return {
        "service": service,
        "service_item": item,
        "window": label,
        "months": len(window),
        "total_cost": float(cost.sum()),
        "variation": variation,
        "correlation": r,
        "score": r * 10,
        "p_value": p,
        "trend": trend,
        "bucket": classify(r, trend, variation, t),
    }


def classify(r: float, trend: float, variation: float, t: Thresholds = DEFAULT_THRESHOLDS) -> str:
    if variation <= t.fixed_variation:
        return FIXED
    if r >= t.very_high:
        return VERY_HIGH
    if r >= t.high:
        return HIGH
    if r >= t.mid:
        return MID
    # Not fixed and not request-driven: linear, by its direction over time.
    return DECLINING if trend < 0 else GROWING


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Items and cost per bucket, largest share of cost first."""
    summary = (
        results.groupby("bucket")
        .agg(items=("service_item", "count"), total_cost=("total_cost", "sum"))
        .reindex(BUCKETS)
        .dropna()
        .reset_index()
    )
    total = summary["total_cost"].sum()
    summary["share"] = summary["total_cost"] / total if total else 0.0
    summary["items"] = summary["items"].astype(int)
    return summary.sort_values("total_cost", ascending=False, ignore_index=True)


def _variation(cost: np.ndarray) -> float:
    """Coefficient of variation: how much cost swings around its average."""
    spread = float(np.std(cost))
    if spread == 0:
        return 0.0
    mean = abs(float(np.mean(cost)))
    return spread / mean if mean else float("inf")


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # A flat series has no correlation to measure.
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    result = stats.pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)
