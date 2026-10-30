"""Build a 12-month forecast from a baseline year.

Most methods start from E, the baseline year's December cost (the exit rate).

Seasonal items, one of four methods:

- Weights (request, manual, or the item's own cost): each month's weight is its
  share of the year (the twelve add up to 100%). December's cost divided by
  December's weight gives a yearly figure, which grows by ``growth`` and is
  spread over the months by their weights:

      month m = E / weight(December) * (1 + growth) * weight(m)

- Fixed + variable: a straight line fitted through the baseline year's monthly
  cost against requests (the same as LINEST in Excel or Google Sheets) gives a
  fixed cost per month and a cost per request. The forecast applies it to the
  forecast requests (baseline-year requests grown by the request growth):

      month m = fixed + cost per request * forecast requests(m)

Linear items (exit-rate method): add a constant step Z every month,

    Z = ((E * 12) * (1 + growth) - (E * 12)) / 78        78 = 1 + 2 + ... + 12
    month k (k = 1 for January) = (E + k * Z) / 30.4 * days in month k

where 30.4 (= 365 / 12) is the average month length. Fixed items use the same
step without the day scaling, since they are billed per month.

Forecasting further ahead (for example 2027 from a 2025 baseline) repeats the
method one year at a time: each year starts from the previous year's forecast
December, so growth is applied once per year. Fixed + variable instead grows
the requests once per year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfe import drivers

STEPS = 78  # 1 + 2 + ... + 12
AVERAGE_MONTH_DAYS = 365 / 12  # 30.4

EXIT_RATE = "Exit rate + growth"
EXIT_RATE_FLAT_DAYS = "Exit rate + growth (no day adjustment)"
REQUEST_WEIGHTS = "Request weights"
MANUAL_WEIGHTS = "Manual weights"
OWN_WEIGHTS = "Own cost weights"
FIXED_VARIABLE = "Fixed + variable"
SEASONAL_METHODS = [REQUEST_WEIGHTS, MANUAL_WEIGHTS, OWN_WEIGHTS, FIXED_VARIABLE]
FIXED_VARIABLE_BY_BUCKET = "Fixed + variable, per bucket"
ALL_SEASONAL = "All seasonal"
ACTUAL = "Actual"
NOT_CLASSIFIED = "Not classified"
TOTAL = "Total"

FORECAST_COLUMNS = ["service", "service_item", "bucket", "method", "month", "forecast"]


def monthly_increment(exit_rate: float, growth: float) -> float:
    """Z: the constant added each month so the year grows by ``growth``."""
    return ((exit_rate * 12) * (1 + growth) - (exit_rate * 12)) / STEPS


@dataclass
class Seasonal:
    """How seasonal items are forecast.

    ``weights``: month (1-12) -> share of the year, for request or manual weights.
    ``requests``: the requests table, needed for fixed + variable.
    ``request_growth``: growth of requests in the forecast year, for fixed + variable.
    """

    method: str = REQUEST_WEIGHTS
    weights: pd.Series | None = None
    requests: pd.DataFrame | None = None
    request_growth: float = 0.0


def exit_rate_months(
    exit_rate: float, growth: float, exit_month: pd.Timestamp, day_adjust: bool = True
) -> pd.DataFrame:
    """The 12 months after ``exit_month``, starting at ``exit_rate`` and adding Z each month.

    Each month builds on the one before (E + Z, E + 2Z, ...). With ``day_adjust``
    the value is then scaled from an average month (30.4 days) to the month's days.
    """
    z = monthly_increment(exit_rate, growth)
    months = forecast_months(exit_month)
    values = [exit_rate + k * z for k in range(1, 13)]
    if day_adjust:
        values = [v / AVERAGE_MONTH_DAYS * m.days_in_month for v, m in zip(values, months)]
    if exit_rate >= 0:
        values = [max(v, 0.0) for v in values]  # a declining cost stops at $0
    return pd.DataFrame({"month": months, "forecast": values})


def forecast_months(exit_month: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(exit_month + pd.offsets.MonthBegin(1), periods=12, freq="MS")


def request_weights(requests: pd.DataFrame, year: int) -> pd.Series | None:
    """Each month's share of the year's requests, indexed 1-12. None without a full year."""
    in_year = requests[requests["month"].dt.year == year]
    return _shares(in_year, "requests")


def own_weights(costs: pd.DataFrame, year: int) -> pd.Series | None:
    """Each month's share of one service item's cost in ``year``, indexed 1-12."""
    return _shares(costs[costs["month"].dt.year == year], "cost")


def _shares(rows: pd.DataFrame, field: str) -> pd.Series | None:
    if len(rows) != 12 or rows[field].sum() <= 0:
        return None
    values = rows.set_index(rows["month"].dt.month)[field]
    return (values / values.sum()).sort_index()


def seasonal_months(
    exit_rate: float, growth: float, weights: pd.Series, exit_month: pd.Timestamp
) -> pd.DataFrame:
    """The 12 months after ``exit_month``, shaped by ``weights``."""
    months = forecast_months(exit_month)
    yearly = exit_rate / weights[exit_month.month] * (1 + growth)
    return pd.DataFrame({"month": months, "forecast": [yearly * weights[m.month] for m in months]})


def fit_fixed_variable(requests: np.ndarray, cost: np.ndarray) -> tuple[float, float]:
    """(cost per request, fixed cost per month): a least-squares line, like LINEST."""
    slope, intercept = np.polyfit(np.asarray(requests, float), np.asarray(cost, float), 1)
    return float(slope), float(intercept)


def forecast_requests(
    requests: pd.DataFrame, baseline_year: int, growth: float, years_ahead: int = 1
) -> pd.Series:
    """Requests per month of year ``baseline_year + years_ahead``: the baseline year's
    requests grown by ``growth`` once per year."""
    in_year = requests[requests["month"].dt.year == baseline_year].sort_values("month")
    months = forecast_months(pd.Timestamp(year=baseline_year + years_ahead - 1, month=12, day=1))
    factor = (1 + growth) ** years_ahead
    return pd.Series(in_year["requests"].to_numpy(float) * factor, index=months)


def fixed_variable_months(
    cost: pd.Series, requests: pd.DataFrame, baseline_year: int, request_growth: float,
    years_ahead: int = 1,
) -> pd.DataFrame | None:
    """Fit cost (indexed by month) against requests in the baseline year, then forecast
    year ``baseline_year + years_ahead``."""
    in_year = cost[cost.index.year == baseline_year]
    reqs = requests.set_index("month")["requests"].reindex(in_year.index)
    if len(in_year) != 12 or reqs.isna().any() or reqs.nunique() < 2:
        return None
    slope, fixed = fit_fixed_variable(reqs.to_numpy(), in_year.to_numpy())
    future = forecast_requests(requests, baseline_year, request_growth, years_ahead)
    values = np.maximum(fixed + slope * future.to_numpy(), 0.0)
    return pd.DataFrame({"month": future.index, "forecast": values})


def method_for(bucket: str) -> str:
    # Fixed fees (support, licences) are billed per month, not per day.
    return EXIT_RATE_FLAT_DAYS if bucket == drivers.FIXED else EXIT_RATE


def build(
    costs: pd.DataFrame,
    baseline_year: int,
    buckets: dict[str, str],
    growth: dict[str, float],
    default_growth: float = 0.0,
    seasonal: Seasonal | None = None,
    years: int = 1,
) -> pd.DataFrame:
    """Forecast every service item for the ``years`` years after ``baseline_year``.

    ``buckets`` and ``growth`` map service item -> bucket / growth rate per year
    (0.07 = 7%). Items missing from ``growth`` use ``default_growth``. ``seasonal``
    sets the method for seasonal items; without what that method needs (weights,
    requests), they fall back to the exit-rate method.
    """
    baseline_december = pd.Timestamp(year=baseline_year, month=12, day=1)
    frames = []
    for (service, item), group in costs.groupby(["service", "service_item"], sort=False):
        december = group.loc[group["month"] == baseline_december, "cost"]
        if december.empty:
            continue
        bucket = buckets.get(item, NOT_CLASSIFIED)
        exit_rate, item_growth = float(december.iloc[0]), growth.get(item, default_growth)
        method = method_for(bucket)
        if bucket in drivers.SEASONAL and seasonal is not None:
            months = _seasonal_item(group, exit_rate, item_growth, baseline_year, seasonal, years)
            if months is not None:
                method = f"Seasonal: {seasonal.method.lower()}"
        if method in (EXIT_RATE, EXIT_RATE_FLAT_DAYS):
            months = _exit_rate_years(exit_rate, item_growth, baseline_year, years,
                                      day_adjust=method == EXIT_RATE)
        frames.append(months.assign(service=service, service_item=item, bucket=bucket,
                                    method=method))
    if not frames:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    return pd.concat(frames, ignore_index=True)[FORECAST_COLUMNS]


def _exit_rate_years(exit_rate, growth, baseline_year, years, day_adjust) -> pd.DataFrame:
    """The exit-rate method, one year at a time; each year starts where the line ended."""
    frames = []
    for n in range(years):
        exit_month = pd.Timestamp(year=baseline_year + n, month=12, day=1)
        frames.append(exit_rate_months(exit_rate, growth, exit_month, day_adjust))
        # The next year starts from December's line value, before day scaling.
        end = exit_rate + 12 * monthly_increment(exit_rate, growth)
        exit_rate = max(end, 0.0) if exit_rate >= 0 else end
    return pd.concat(frames, ignore_index=True)


def _seasonal_item(group, exit_rate, growth, baseline_year, seasonal: Seasonal, years: int):
    if seasonal.method == FIXED_VARIABLE:
        if seasonal.requests is None:
            return None
        cost = group.set_index("month")["cost"].sort_index()
        frames = [fixed_variable_months(cost, seasonal.requests, baseline_year,
                                        seasonal.request_growth, n) for n in range(1, years + 1)]
        return None if frames[0] is None else pd.concat(frames, ignore_index=True)
    weights = (own_weights(group, baseline_year) if seasonal.method == OWN_WEIGHTS
               else seasonal.weights)
    if weights is None or weights.get(12, 0) <= 0:
        return None
    frames = []
    for n in range(years):
        months = seasonal_months(exit_rate, growth, weights,
                                 pd.Timestamp(year=baseline_year + n, month=12, day=1))
        frames.append(months)
        exit_rate = months["forecast"].iloc[-1]  # December, = exit rate x (1 + growth)
    return pd.concat(frames, ignore_index=True)


def fixed_variable_by_bucket(
    costs: pd.DataFrame, requests: pd.DataFrame, baseline_year: int,
    buckets: dict[str, str], request_growth: float, years_ahead: int = 1,
) -> pd.Series:
    """Forecast total per seasonal bucket in year ``baseline_year + years_ahead``,
    fitting each bucket's total cost at once."""
    totals = {}
    for bucket in drivers.SEASONAL:
        items = [i for i, b in buckets.items() if b == bucket]
        if not items:
            continue
        cost = costs[costs["service_item"].isin(items)].groupby("month")["cost"].sum()
        months = fixed_variable_months(cost, requests, baseline_year, request_growth,
                                       years_ahead)
        if months is not None:
            totals[bucket] = months["forecast"].sum()
    return pd.Series(totals, dtype=float)


def compare_seasonal(
    costs: pd.DataFrame,
    baseline_year: int,
    buckets: dict[str, str],
    growth: dict[str, float],
    methods: dict[str, Seasonal],
    requests: pd.DataFrame | None = None,
    request_growth: float = 0.0,
    years_ahead: int = 1,
) -> pd.DataFrame:
    """Forecast totals of the seasonal buckets in year ``baseline_year + years_ahead``
    under each method, one row per method.

    Adds the fixed + variable fit per bucket when ``requests`` is given, and an
    Actual row with the error of each method when the forecast year has happened.
    """
    seasonal_items = {i: b for i, b in buckets.items() if b in drivers.SEASONAL}
    rows = {}
    target_year = baseline_year + years_ahead
    target = costs[costs["month"].dt.year == target_year]
    if target["month"].nunique() == 12:
        actual = target[target["service_item"].isin(seasonal_items)]
        rows[ACTUAL] = actual.groupby(actual["service_item"].map(seasonal_items))["cost"].sum()
    for label, seasonal in methods.items():
        result = build(costs, baseline_year, buckets, growth, seasonal=seasonal,
                       years=years_ahead)
        result = result[result["bucket"].isin(drivers.SEASONAL)
                        & (result["month"].dt.year == target_year)]
        rows[label] = result.groupby("bucket")["forecast"].sum()
    if requests is not None:
        rows[FIXED_VARIABLE_BY_BUCKET] = fixed_variable_by_bucket(
            costs, requests, baseline_year, buckets, request_growth, years_ahead)
    table = pd.DataFrame(rows).T
    table = table[[b for b in drivers.SEASONAL if b in table.columns]]
    table[ALL_SEASONAL] = table.sum(axis=1)
    if ACTUAL in table.index:
        table["Off by"] = _ratio(table[ALL_SEASONAL], pd.Series(
            table.loc[ACTUAL, ALL_SEASONAL], index=table.index))
        table.loc[ACTUAL, "Off by"] = np.nan
    return table.reset_index(names="method")


AVERAGE_MOM = "Average month over month"
JAN_TO_DEC = "January vs December"


def in_year_growth(costs: pd.DataFrame, years: list[int], how: str = AVERAGE_MOM) -> pd.DataFrame:
    """Growth within each year, per service item (0.012 = +1.2%).

    ``AVERAGE_MOM``: the average of the 11 month-over-month changes (Feb vs Jan ... Dec vs Nov).
    ``JAN_TO_DEC``: December vs January, the same as comparing them annualised (x 12).
    A change from a zero or negative month is meaningless, so it is left out.
    """
    rows = []
    for item, group in costs.groupby("service_item", sort=False):
        by_month = group.set_index("month")["cost"]
        row = {"service_item": item}
        for year in years:
            months = by_month[by_month.index.year == year].sort_index().to_numpy(dtype=float)
            row[str(year)] = _growth_within(months, how) if len(months) == 12 else None
        rows.append(row)
    return pd.DataFrame(rows, columns=["service_item", *[str(y) for y in years]])


def _growth_within(months: np.ndarray, how: str) -> float | None:
    if how == JAN_TO_DEC:
        return months[-1] / months[0] - 1 if months[0] > 0 else None
    previous, current = months[:-1], months[1:]
    valid = previous > 0
    if not valid.any():
        return None
    return float(np.mean(current[valid] / previous[valid] - 1))


def summarize(
    costs: pd.DataFrame, forecast: pd.DataFrame, baseline_year: int,
    target_year: int | None = None,
) -> pd.DataFrame:
    """Per service item: baseline actual, exit rate, forecast total for ``target_year``
    (default: the year after the baseline), and that year's actuals if they exist."""
    target_year = target_year or baseline_year + 1
    forecast = forecast[forecast["month"].dt.year == target_year]
    year = costs["month"].dt.year
    baseline = costs[year == baseline_year].groupby("service_item")["cost"].sum()
    actual_next = costs[year == target_year]
    exit_rate = costs[costs["month"] == pd.Timestamp(year=baseline_year, month=12, day=1)]
    summary = pd.DataFrame({
        "bucket": forecast.groupby("service_item")["bucket"].first(),
        "method": forecast.groupby("service_item")["method"].first(),
        "baseline_total": baseline,
        "exit_rate": exit_rate.set_index("service_item")["cost"],
        "forecast_total": forecast.groupby("service_item")["forecast"].sum(),
    })
    summary["change"] = summary["forecast_total"] / summary["baseline_total"] - 1
    if actual_next["month"].nunique() == 12:
        summary["actual_total"] = actual_next.groupby("service_item")["cost"].sum()
        summary["error"] = summary["forecast_total"] / summary["actual_total"] - 1
    return summary.dropna(subset=["forecast_total"]).reset_index(names="service_item")


def by_bucket(summary: pd.DataFrame) -> pd.DataFrame:
    """Totals per bucket plus a Total row, from the per-item ``summarize`` table."""
    money = [c for c in ("baseline_total", "exit_rate", "forecast_total", "actual_total")
             if c in summary]
    table = summary.groupby("bucket")[money].sum()
    table.insert(0, "items", summary.groupby("bucket").size())
    table = table.loc[_bucket_order(table.index)]
    table.loc[TOTAL] = table.sum()
    table["change"] = _ratio(table["forecast_total"], table["baseline_total"])
    if "actual_total" in table:
        table["error"] = _ratio(table["forecast_total"], table["actual_total"])
    return table.reset_index(names="bucket")


def compare(tables: dict[int, pd.DataFrame], years_ahead: int = 1) -> pd.DataFrame:
    """Side by side ``by_bucket`` tables, one set of columns per baseline year, each
    forecasting ``years_ahead`` years past its baseline."""
    frames = []
    for year, table in sorted(tables.items()):
        target = year + years_ahead
        columns = {"forecast_total": f"Forecast {target} (from {year})"}
        if "actual_total" in table:
            columns |= {"actual_total": f"Actual {target}", "error": f"Off by ({target})"}
        frames.append(table.set_index("bucket")[list(columns)].rename(columns=columns))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    return out.loc[_bucket_order(out.index)].reset_index(names="bucket")


def _bucket_order(present) -> list[str]:
    order = [*drivers.BUCKETS, NOT_CLASSIFIED, TOTAL]
    return [b for b in order if b in present]


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator - 1, empty where the denominator is zero."""
    return (numerator / denominator.replace(0, np.nan)) - 1
