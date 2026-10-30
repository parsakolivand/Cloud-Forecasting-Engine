"""Data-quality checks and gap filling for the canonical tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MIN_MONTHS = 12  # one full year is the minimum to see seasonality

# Spend with other vendors billed through the cloud provider (AWS, Azure and Google
# Cloud marketplaces). It does not follow cloud usage, so it is excluded by default.
THIRD_PARTY_WORDS = ("marketplace", "third party", "third-party", "3rd party")


@dataclass
class Issue:
    level: str  # "error" blocks the analysis, "warning" and "info" do not
    message: str


@dataclass
class QualityReport:
    first_month: pd.Timestamp | None
    last_month: pd.Timestamp | None
    months: int
    service_items: int
    complete_years: list[int]
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)


def fill_missing_months(costs: pd.DataFrame) -> pd.DataFrame:
    """Give every service item a row for every month the export covers.

    A month that exists in the export but has no row for an item means the item
    cost nothing that month, so it becomes $0. Months missing from the whole
    export are left out: that is a gap in the data, not a zero.
    """
    months = costs["month"].unique()
    items = costs[["service", "service_item"]].drop_duplicates()
    grid = items.merge(pd.DataFrame({"month": months}), how="cross")
    filled = grid.merge(costs, on=["service", "service_item", "month"], how="left")
    filled["cost"] = filled["cost"].fillna(0.0)
    return filled[costs.columns].sort_values(["service", "service_item", "month"], ignore_index=True)


def check(costs: pd.DataFrame, requests: pd.DataFrame | None = None) -> QualityReport:
    issues: list[Issue] = []
    if costs.empty:
        return QualityReport(None, None, 0, 0, [], [Issue("error", "The cost file has no rows.")])

    months = pd.DatetimeIndex(sorted(costs["month"].unique()))
    expected = pd.date_range(months[0], months[-1], freq="MS")
    gaps = expected.difference(months)

    if len(expected) < MIN_MONTHS:
        issues.append(Issue(
            "error",
            f"Only {len(expected)} months of cost history. At least {MIN_MONTHS} are needed "
            "to capture seasonality; more is better.",
        ))
    if len(gaps):
        issues.append(Issue(
            "warning",
            f"No cost data at all for {_month_list(gaps)}. Check that the export is complete.",
        ))

    negative = costs[costs["cost"] < 0]
    if not negative.empty:
        issues.append(Issue(
            "info",
            f"{len(negative)} rows have negative cost (credits or refunds). They are kept as-is.",
        ))

    complete_years = full_years(months)
    if requests is not None:
        req_months = pd.DatetimeIndex(requests["month"])
        missing = months.difference(req_months)
        if len(missing):
            issues.append(Issue(
                "warning",
                f"Requests data is missing for {_month_list(missing)}. Those months are "
                "skipped when comparing cost with requests.",
            ))
        complete_years = sorted(set(complete_years) & set(full_years(req_months)))
        if (requests["requests"] < 0).any():
            issues.append(Issue("error", "Requests data contains negative values."))

    return QualityReport(
        first_month=months[0],
        last_month=months[-1],
        months=len(months),
        service_items=costs["service_item"].nunique(),
        complete_years=complete_years,
        issues=issues,
    )


def third_party_items(costs: pd.DataFrame) -> list[str]:
    """Service items whose service or item name marks them as third-party spend."""
    names = costs[["service", "service_item"]].drop_duplicates()
    text = (names["service"] + " " + names["service_item"]).str.lower()
    flagged = text.apply(lambda t: any(word in t for word in THIRD_PARTY_WORDS))
    return sorted(names.loc[flagged, "service_item"].unique())


def full_years(months: pd.DatetimeIndex) -> list[int]:
    counts = pd.Series(months.year).value_counts()
    return sorted(int(y) for y, n in counts.items() if n == 12)


def _month_list(months: pd.DatetimeIndex, limit: int = 6) -> str:
    names = [m.strftime("%b %Y") for m in months[:limit]]
    more = len(months) - limit
    return ", ".join(names) + (f" and {more} more" if more > 0 else "")
