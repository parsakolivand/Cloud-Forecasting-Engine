import numpy as np
import pandas as pd
import pytest

from cfe import drivers
from cfe.drivers import ALL_HISTORY, Thresholds, analyze, classify, summarize

MONTHS = pd.date_range("2023-01-01", periods=24, freq="MS")
SEASON = np.tile([5, 6, 7, 6, 5, 3, 2, 5, 9, 9, 7, 5], 2).astype(float)


def table(item, cost):
    return pd.DataFrame({"month": MONTHS, "service": item.split(":")[0],
                         "service_item": item, "cost": cost})


def requests(values=SEASON * 1e9):
    return pd.DataFrame({"month": MONTHS, "requests": values})


def result(costs, years=(2023, 2024)):
    return analyze(costs, requests(), list(years)).set_index("window")


def test_cost_proportional_to_requests_scores_ten():
    res = result(table("EC2: Compute", SEASON * 100))
    assert res.loc["2023", "score"] == pytest.approx(10)
    assert res.loc["2023", "bucket"] == drivers.VERY_HIGH


def test_price_drop_between_years_hides_correlation_when_pooled():
    # Same seasonal shape each year, but year two traffic doubles while price halves,
    # so cost per month is identical across years and requests jump.
    reqs = SEASON * 1e9 * np.repeat([1, 2], 12)
    costs = table("EC2: Compute", SEASON * 100)
    res = analyze(costs, requests(reqs), [2023, 2024]).set_index("window")
    assert res.loc["2023", "score"] == pytest.approx(10)
    assert res.loc["2024", "score"] == pytest.approx(10)
    assert res.loc[ALL_HISTORY, "score"] < 9


def test_steady_growth_is_not_request_driven():
    res = result(table("S3: Storage", 1000 + 50 * np.arange(24)))
    assert res.loc["2023", "bucket"] == drivers.GROWING


def test_flat_cost_has_no_correlation():
    res = result(table("Support: Business", np.full(24, 30_000.0)))
    assert np.isnan(res.loc["2023", "score"])
    assert res.loc["2023", "bucket"] == drivers.FIXED


def test_incomplete_year_is_skipped():
    res = analyze(table("EC2: Compute", SEASON * 100).iloc[:18], requests(), [2023])
    assert set(res["window"]) == {"2023", ALL_HISTORY}


@pytest.mark.parametrize("r,trend,variation,bucket", [
    (0.90, 0.0, 0.2, drivers.VERY_HIGH),
    (0.85, 0.0, 0.2, drivers.VERY_HIGH),
    (0.80, 0.0, 0.2, drivers.HIGH),
    (0.60, 0.95, 0.2, drivers.MID),  # seasonal is checked before linear
    (0.50, 0.0, 0.2, drivers.MID),
    (0.49, 0.95, 0.2, drivers.GROWING),
    (0.30, -0.95, 0.2, drivers.DECLINING),
    (0.10, 0.20, 0.2, drivers.GROWING),  # irregular, but still linear by rule
    (-0.80, -0.1, 0.2, drivers.DECLINING),
    (float("nan"), 0.5, 0.2, drivers.GROWING),
    (0.90, 0.0, 0.01, drivers.FIXED),  # barely moves: fixed wins over a noisy correlation
    (float("nan"), float("nan"), 0.0, drivers.FIXED),
])
def test_classify(r, trend, variation, bucket):
    assert classify(r, trend, variation) == bucket


def test_custom_thresholds():
    assert classify(0.55, 0.3, 0.2, Thresholds(mid=0.6)) == drivers.GROWING
    assert classify(0.55, 0.3, 0.2, Thresholds(mid=0.5)) == drivers.MID


def test_summarize_shares_cost_by_bucket():
    results = analyze(
        pd.concat([table("EC2: Compute", SEASON * 300), table("Support: Business", np.full(24, 100.0))]),
        requests(), [2023],
    )
    summary = summarize(results[results["window"] == "2023"]).set_index("bucket")
    ec2, support = SEASON[:12].sum() * 300, 12 * 100.0
    assert summary.loc[drivers.VERY_HIGH, "share"] == pytest.approx(ec2 / (ec2 + support))
    assert summary["items"].sum() == 2
