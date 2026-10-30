import pandas as pd

from cfe.quality import check, fill_missing_months, third_party_items


def costs_for(months, item="EC2: Compute", cost=10.0):
    return pd.DataFrame({
        "month": pd.to_datetime(months),
        "service": item.split(":")[0],
        "service_item": item,
        "cost": cost,
    })


def test_short_history_is_an_error():
    report = check(costs_for(pd.date_range("2023-01-01", periods=6, freq="MS")))
    assert not report.ok
    assert "At least 12" in report.issues[0].message


def test_full_year_is_complete():
    report = check(costs_for(pd.date_range("2023-01-01", periods=12, freq="MS")))
    assert report.ok
    assert report.complete_years == [2023]


def test_gap_in_export_is_a_warning_and_breaks_the_year():
    months = pd.date_range("2023-01-01", periods=14, freq="MS").delete(3)
    report = check(costs_for(months))
    assert any("No cost data at all for Apr 2023" in i.message for i in report.issues)
    assert report.complete_years == []


def test_item_missing_a_month_is_filled_with_zero():
    months = pd.date_range("2023-01-01", periods=3, freq="MS")
    costs = pd.concat([costs_for(months, "EC2: Compute"), costs_for(months[:2], "S3: Storage")])
    filled = fill_missing_months(costs)
    s3 = filled[filled["service_item"] == "S3: Storage"]
    assert s3["cost"].tolist() == [10.0, 10.0, 0.0]


def test_requests_missing_months_limit_complete_years():
    costs = costs_for(pd.date_range("2023-01-01", periods=24, freq="MS"))
    requests = pd.DataFrame({
        "month": pd.date_range("2023-01-01", periods=18, freq="MS"),
        "requests": 1.0,
    })
    report = check(costs, requests)
    assert report.complete_years == [2023]
    assert any("Requests data is missing" in i.message for i in report.issues)


def test_third_party_items_are_flagged_by_name():
    costs = pd.concat([
        costs_for(["2023-01-01"], "Marketplace: Software"),
        costs_for(["2023-01-01"], "AWS Marketplace: Datadog"),
        costs_for(["2023-01-01"], "Azure: Third-Party Licences"),
        costs_for(["2023-01-01"], "EC2: Compute"),
    ])
    assert third_party_items(costs) == [
        "AWS Marketplace: Datadog", "Azure: Third-Party Licences", "Marketplace: Software",
    ]
