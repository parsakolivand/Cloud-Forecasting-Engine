import numpy as np
import pandas as pd
import pytest

from cfe import drivers, forecast

DEC_2025 = pd.Timestamp("2025-12-01")


def test_increment_matches_worked_example():
    # $100 exit rate, 7% growth: Z = (1200 * 1.07 - 1200) / 78 = $1.08
    assert forecast.monthly_increment(100, 0.07) == pytest.approx(1.0769, abs=1e-4)


def test_year_without_day_adjustment_grows_by_exactly_the_growth_rate():
    months = forecast.exit_rate_months(100, 0.07, DEC_2025, day_adjust=False)
    assert months["forecast"].iloc[0] == pytest.approx(101.08, abs=0.01)  # January
    assert months["forecast"].iloc[-1] == pytest.approx(112.92, abs=0.01)  # December
    assert months["forecast"].sum() == pytest.approx(100 * 12 * 1.07)
    assert months["month"].iloc[0] == pd.Timestamp("2026-01-01")


def test_day_adjustment_scales_from_an_average_month():
    months = forecast.exit_rate_months(100, 0.07, DEC_2025).set_index("month")["forecast"]
    z = forecast.monthly_increment(100, 0.07)
    # Each month builds on the previous one (E + kZ), then is scaled / 30.4 * days.
    assert months["2026-01-01"] == pytest.approx((100 + z) / (365 / 12) * 31)
    assert months["2026-02-01"] == pytest.approx((100 + 2 * z) / (365 / 12) * 28)
    assert months["2026-04-01"] == pytest.approx((100 + 4 * z) / (365 / 12) * 30)


def test_day_adjusted_year_still_grows_by_the_growth_rate():
    months = forecast.exit_rate_months(100, 0.07, DEC_2025)
    assert months["forecast"].sum() == pytest.approx(1200 * 1.07, rel=0.001)


def test_steep_decline_stops_at_zero():
    months = forecast.exit_rate_months(100, -2.0, DEC_2025, day_adjust=False)
    assert months["forecast"].min() == 0


def costs_table(item, values, start="2025-01-01"):
    return pd.DataFrame({
        "month": pd.date_range(start, periods=len(values), freq="MS"),
        "service": item.split(":")[0], "service_item": item, "cost": values,
    })


def test_build_uses_item_growth_and_skips_day_adjustment_for_fixed():
    costs = pd.concat([
        costs_table("S3: Storage", [100.0] * 12),
        costs_table("Support: Business", [500.0] * 12),
    ])
    result = forecast.build(
        costs, 2025,
        buckets={"S3: Storage": drivers.GROWING, "Support: Business": drivers.FIXED},
        growth={"S3: Storage": 0.07},
    )
    support = result[result["service_item"] == "Support: Business"]
    assert support["forecast"].tolist() == [500.0] * 12  # default growth 0, flat, no day scaling
    storage = result[result["service_item"] == "S3: Storage"].set_index("month")["forecast"]
    z = forecast.monthly_increment(100, 0.07)
    assert storage["2026-01-01"] == pytest.approx((100 + z) / (365 / 12) * 31)


def test_in_year_growth_average_month_over_month():
    # Feb +10%, then flat, then Dec +13.6%: the average of the 11 changes.
    costs = costs_table("S3: Storage", [100.0] + [110.0] * 10 + [125.0])
    growth = forecast.in_year_growth(costs, [2025]).set_index("service_item")
    assert growth.loc["S3: Storage", "2025"] == pytest.approx((0.10 + 125 / 110 - 1) / 11)


def test_in_year_growth_january_vs_december():
    costs = costs_table("S3: Storage", [100.0] + [110.0] * 10 + [125.0])
    growth = forecast.in_year_growth(costs, [2025], forecast.JAN_TO_DEC).set_index("service_item")
    assert growth.loc["S3: Storage", "2025"] == pytest.approx(0.25)


# Made-up monthly weights that add up to 100%.
WEIGHTS = pd.Series(
    [8.0, 9.0, 9.5, 9.0, 8.0, 5.0, 4.0, 7.0, 11.5, 11.5, 10.0, 7.5],
    index=range(1, 13),
) / 100


def test_request_weights_are_each_months_share():
    requests = pd.DataFrame({"month": pd.date_range("2025-01-01", periods=12, freq="MS"),
                             "requests": WEIGHTS.to_numpy() * 1e9})
    weights = forecast.request_weights(requests, 2025)
    assert weights.sum() == pytest.approx(1)
    assert weights[12] == pytest.approx(0.075)
    assert forecast.request_weights(requests, 2024) is None


def test_seasonal_month_is_exit_rate_over_december_weight_times_growth_times_weight():
    months = forecast.seasonal_months(100, 0.07, WEIGHTS, DEC_2025).set_index("month")["forecast"]
    assert months["2026-02-01"] == pytest.approx(100 / 0.075 * 1.07 * 0.09)
    assert months["2026-12-01"] == pytest.approx(107)  # December grows by exactly the growth
    assert months.sum() == pytest.approx(100 / 0.075 * 1.07)


def test_build_uses_seasonal_method_only_for_seasonal_items():
    costs = pd.concat([costs_table("EC2: Compute", [100.0] * 12),
                       costs_table("S3: Storage", [100.0] * 12)])
    buckets = {"EC2: Compute": drivers.VERY_HIGH, "S3: Storage": drivers.GROWING}
    seasonal = forecast.Seasonal(weights=WEIGHTS)
    result = forecast.build(costs, 2025, buckets, {}, default_growth=0.07, seasonal=seasonal)
    methods = result.groupby("service_item")["method"].first()
    assert methods["EC2: Compute"] == "Seasonal: request weights"
    assert methods["S3: Storage"] == forecast.EXIT_RATE
    # Without request weights, seasonal items fall back to the exit-rate method.
    fallback = forecast.build(costs, 2025, buckets, {}, default_growth=0.07)
    assert set(fallback["method"]) == {forecast.EXIT_RATE}


def test_summarize_compares_with_actuals_when_available():
    costs = costs_table("S3: Storage", [100.0] * 24, start="2024-01-01")
    result = forecast.build(costs, 2024, {}, {}, default_growth=0.0)
    summary = forecast.summarize(costs, result, 2024).set_index("service_item")
    assert summary.loc["S3: Storage", "baseline_total"] == 1200
    assert summary.loc["S3: Storage", "actual_total"] == 1200
    assert summary.loc["S3: Storage", "error"] == pytest.approx(0)  # 365 days / 30.4 = 12 months


def test_own_cost_weights_make_the_yearly_figure_last_years_total():
    values = [100.0, 90, 80, 70, 60, 50, 50, 60, 120, 130, 110, 90]
    costs = costs_table("EC2: Compute", values)
    result = forecast.build(costs, 2025, {"EC2: Compute": drivers.HIGH}, {}, default_growth=0.07,
                            seasonal=forecast.Seasonal(method=forecast.OWN_WEIGHTS))
    assert result["forecast"].sum() == pytest.approx(sum(values) * 1.07)
    assert result["forecast"].tolist() == pytest.approx([v * 1.07 for v in values])


def test_fixed_variable_matches_linest():
    requests = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], float)
    slope, fixed = forecast.fit_fixed_variable(requests, 500 + 2.5 * requests)
    assert slope == pytest.approx(2.5)
    assert fixed == pytest.approx(500)


def test_fixed_variable_forecast_uses_grown_requests():
    months = pd.date_range("2025-01-01", periods=12, freq="MS")
    reqs = np.arange(1, 13, dtype=float) * 1000
    requests = pd.DataFrame({"month": months, "requests": reqs})
    costs = costs_table("EC2: Compute", list(500 + 0.1 * reqs))
    seasonal = forecast.Seasonal(method=forecast.FIXED_VARIABLE, requests=requests,
                                 request_growth=0.10)
    result = forecast.build(costs, 2025, {"EC2: Compute": drivers.VERY_HIGH}, {},
                            seasonal=seasonal)
    assert result["forecast"].tolist() == pytest.approx(list(500 + 0.1 * reqs * 1.10))
    assert result["method"].iloc[0] == "Seasonal: fixed + variable"


def test_compare_seasonal_has_a_row_per_method_and_actuals():
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    reqs = np.tile(WEIGHTS.to_numpy(), 2) * 1e6
    requests = pd.DataFrame({"month": months, "requests": reqs})
    costs = costs_table("EC2: Compute", list(200 + 0.001 * reqs), start="2024-01-01")
    buckets = {"EC2: Compute": drivers.VERY_HIGH}
    methods = {
        forecast.REQUEST_WEIGHTS: forecast.Seasonal(weights=forecast.request_weights(requests, 2024)),
        forecast.OWN_WEIGHTS: forecast.Seasonal(method=forecast.OWN_WEIGHTS),
        forecast.FIXED_VARIABLE: forecast.Seasonal(method=forecast.FIXED_VARIABLE, requests=requests),
    }
    table = forecast.compare_seasonal(costs, 2024, buckets, {}, methods, requests).set_index("method")
    assert list(table.index) == [forecast.ACTUAL, *methods, forecast.FIXED_VARIABLE_BY_BUCKET]
    # Same pattern both years and no growth: own weights and fixed + variable are exact.
    assert table.loc[forecast.OWN_WEIGHTS, "Off by"] == pytest.approx(0, abs=1e-9)
    assert table.loc[forecast.FIXED_VARIABLE, "Off by"] == pytest.approx(0, abs=1e-9)
    assert table.loc[forecast.FIXED_VARIABLE_BY_BUCKET, "Off by"] == pytest.approx(0, abs=1e-9)


def test_linear_second_year_starts_where_the_first_year_line_ended():
    costs = costs_table("S3: Storage", [100.0] * 12)
    result = forecast.build(costs, 2025, {"S3: Storage": drivers.GROWING}, {},
                            default_growth=0.07, years=2)
    assert result["month"].min() == pd.Timestamp("2026-01-01")
    assert result["month"].max() == pd.Timestamp("2027-12-01")
    z1 = forecast.monthly_increment(100, 0.07)
    exit_2026 = 100 + 12 * z1  # December 2026 line value, before day scaling
    z2 = forecast.monthly_increment(exit_2026, 0.07)
    jan_2027 = result.set_index("month").loc["2027-01-01", "forecast"]
    assert jan_2027 == pytest.approx((exit_2026 + z2) / (365 / 12) * 31)


def test_fixed_two_years_ahead_grows_once_per_year():
    costs = costs_table("Support: Business", [500.0] * 12)
    result = forecast.build(costs, 2025, {"Support: Business": drivers.FIXED}, {},
                            default_growth=0.0, years=2)
    assert result["forecast"].tolist() == [500.0] * 24


def test_seasonal_weights_compound_each_year():
    costs = costs_table("EC2: Compute", [100.0] * 12)
    result = forecast.build(costs, 2025, {"EC2: Compute": drivers.VERY_HIGH}, {},
                            default_growth=0.07, seasonal=forecast.Seasonal(weights=WEIGHTS),
                            years=2)
    totals = result.groupby(result["month"].dt.year)["forecast"].sum()
    assert totals[2026] == pytest.approx(100 / 0.075 * 1.07)
    assert totals[2027] == pytest.approx(100 / 0.075 * 1.07 ** 2)


def test_fixed_variable_grows_requests_each_year():
    months = pd.date_range("2025-01-01", periods=12, freq="MS")
    reqs = np.arange(1, 13, dtype=float) * 1000
    requests = pd.DataFrame({"month": months, "requests": reqs})
    costs = costs_table("EC2: Compute", list(500 + 0.1 * reqs))
    seasonal = forecast.Seasonal(method=forecast.FIXED_VARIABLE, requests=requests,
                                 request_growth=0.10)
    result = forecast.build(costs, 2025, {"EC2: Compute": drivers.VERY_HIGH}, {},
                            seasonal=seasonal, years=2)
    year_2027 = result[result["month"].dt.year == 2027]["forecast"]
    assert year_2027.tolist() == pytest.approx(list(500 + 0.1 * reqs * 1.10 ** 2))


def test_summarize_reports_the_chosen_forecast_year():
    costs = costs_table("S3: Storage", [100.0] * 12)
    result = forecast.build(costs, 2025, {}, {}, default_growth=0.07, years=2)
    summary = forecast.summarize(costs, result, 2025, target_year=2027).set_index("service_item")
    in_2027 = result[result["month"].dt.year == 2027]["forecast"].sum()
    assert summary.loc["S3: Storage", "forecast_total"] == pytest.approx(in_2027)
    assert summary.loc["S3: Storage", "baseline_total"] == 1200
