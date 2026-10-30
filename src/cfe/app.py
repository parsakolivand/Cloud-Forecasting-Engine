"""Web interface. Run:  uv run streamlit run src/cfe/app.py"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from cfe import drivers, forecast, ingest, quality

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
TEMPLATES = ROOT / "templates"
REPO_URL = "https://github.com/parsakolivand/Cloud-Forecasting-Engine"
UPLOAD, AWS, SAMPLE = "Upload a CSV file", "Connect to AWS billing", "Try sample data"
YEAR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, safe together on a scatter
MUTED = "#898781"
# Axis labels like "$1.2M" and "150B" (d3's SI prefix for billions is G).
MONEY_AXIS = "'$' + replace(format(datum.value, '.3~s'), 'G', 'B')"
COUNT_AXIS = "replace(format(datum.value, '.3~s'), 'G', 'B')"


@st.cache_data
def read_costs(data: bytes, name: str = "") -> pd.DataFrame:
    return ingest.load_costs(io.BytesIO(data), name)


@st.cache_data
def read_requests(data: bytes, name: str = "") -> pd.DataFrame:
    return ingest.load_requests(io.BytesIO(data), name)


@st.cache_data(show_spinner="Fetching costs from AWS Cost Explorer...")
def read_aws(start: date, end: date, profile: str | None, dimension: str, metric: str,
             include_marketplace: bool):
    from cfe.connectors import aws

    return aws.fetch_costs(start, end, profile=profile, item_dimension=dimension, metric=metric,
                           include_marketplace=include_marketplace)


def sidebar() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    costs = requests = None
    with st.sidebar:
        st.subheader(":material/payments: Cost data")
        source = st.radio("Where is your cost data?", [UPLOAD, AWS, SAMPLE], key="source")
        if source == UPLOAD:
            upload = st.file_uploader("Monthly cost per service item", type=ingest.FILE_TYPES)
            template_button("cost")
            if upload:
                costs = read_costs(upload.getvalue(), upload.name)
        elif source == AWS:
            costs = aws_form()
        else:
            costs = read_costs((EXAMPLES / "sample_costs.csv").read_bytes())
            requests = read_requests((EXAMPLES / "sample_requests.csv").read_bytes())

        st.subheader(":material/bar_chart: Requests data")
        if source == SAMPLE:
            st.caption("Sample requests data is loaded.")
        else:
            upload = st.file_uploader("Total requests (API calls) per month",
                                      type=ingest.FILE_TYPES)
            template_button("requests")
            if upload:
                requests = read_requests(upload.getvalue(), upload.name)

        st.divider()
        st.caption(f"Your files are only used in this session and are not stored. "
                   f"[Source code on GitHub]({REPO_URL})")
    return costs, requests


def template_button(kind: str) -> None:
    name = f"{kind}_template.csv"
    st.download_button(
        f"Download the {kind} template", (TEMPLATES / name).read_bytes(), file_name=name,
        mime="text/csv", icon=":material/download:", type="tertiary",
    )


def aws_form() -> pd.DataFrame | None:
    st.caption(
        "Uses your local AWS credentials (profile or SSO). Keys are never entered here. "
        "Needs the ce:GetCostAndUsage permission; AWS charges $0.01 per API page."
    )
    with st.form("aws"):
        profile = st.text_input("AWS profile (blank = default)")
        today = date.today().replace(day=1)
        start = st.date_input("From", date(today.year - 3, 1, 1))
        end = st.date_input("To (exclusive)", today)
        dimension = st.selectbox("Service item = service +", ["USAGE_TYPE", "OPERATION", "INSTANCE_TYPE"])
        metric = st.selectbox("Cost type", ["UnblendedCost", "AmortizedCost", "NetAmortizedCost"])
        marketplace = st.checkbox(
            "Include AWS Marketplace", value=False,
            help="Marketplace purchases are spend with other vendors, billed through AWS. "
                 "They do not follow your cloud usage.",
        )
        submitted = st.form_submit_button("Fetch costs")
    if submitted:
        st.session_state["aws_query"] = (
            start.replace(day=1), end.replace(day=1), profile or None, dimension, metric,
            marketplace,
        )
    if "aws_query" in st.session_state:
        try:
            return read_aws(*st.session_state["aws_query"])
        except ImportError:
            st.error("Install the AWS extra first:  uv sync --extra aws")
        except Exception as exc:  # noqa: BLE001 - credentials, permissions, throttling: show AWS's message
            st.error(f"AWS request failed: {exc}")
    return None


STEPS = [
    (":material/upload_file:", "1. Add your data",
     ("Monthly cost per service item and total requests per month, for at least 12 "
      "months. Or try the sample data.")),
    (":material/hub:", "2. Find the cost drivers",
     "Each service item lands in a bucket: seasonal (follows requests), linear or fixed."),
    (":material/trending_up:", "3. Forecast next year",
     "Pick a baseline year and growth, and get a 12-month forecast by bucket and item."),
]


def welcome() -> None:
    st.markdown("#### Forecast next year's cloud cost in three steps")
    for column, (icon, title, text) in zip(st.columns(3), STEPS):
        with column.container(border=True, height="stretch"):
            st.markdown(f"### {icon}\n**{title}**\n\n{text}")
    st.info("Choose your cost data in the sidebar to begin, or pick **Try sample data**.",
            icon=":material/arrow_back:")

    st.markdown("#### Your files")
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Cost file**: one row per service item per month")
        st.dataframe(pd.read_csv(TEMPLATES / "cost_template.csv").head(4), hide_index=True)
        template_button("cost")
    with right.container(border=True):
        st.markdown("**Requests file**: total requests (API calls) per month")
        st.dataframe(pd.read_csv(TEMPLATES / "requests_template.csv"), hide_index=True)
        template_button("requests")
    st.caption(
        "Use at least 12 months; more is better. Name service items 'Service: Item' "
        "(e.g. 'EC2: Compute Instance'). Values like $1,234.50 and column names like 'date' "
        "or 'amount' are understood, and tab-separated and Excel files work too."
    )


def data_check(costs: pd.DataFrame, report: quality.QualityReport) -> list[str]:
    """Show the data checks; return the service items the user excluded."""
    a, b, c, d = st.columns(4)
    a.metric("Months of history", report.months, border=True)
    b.metric("Service items", report.service_items, border=True)
    c.metric("Total cost", f"${costs['cost'].sum():,.0f}", border=True)
    d.metric("Complete calendar years", len(report.complete_years), border=True)
    st.caption(f"{report.first_month:%B %Y} to {report.last_month:%B %Y}")

    for issue in report.issues:
        show = {"error": st.error, "warning": st.warning, "info": st.info}[issue.level]
        show(issue.message, icon={"error": "🛑", "warning": "⚠️", "info": "ℹ️"}[issue.level])
    if not report.issues:
        st.success("No problems found in the data.", icon="✅")

    excluded = exclusions(costs)

    st.subheader("Clean monthly cost")
    wide = costs.pivot_table(index="service_item", columns="month", values="cost", aggfunc="sum")
    wide.columns = [f"{m:%b %Y}" for m in wide.columns]
    st.dataframe(wide.style.format("${:,.0f}"), height=360)
    st.download_button(
        "Download clean data (CSV)",
        costs.to_csv(index=False).encode(),
        file_name="clean_costs.csv",
        mime="text/csv",
    )
    return excluded


def exclusions(costs: pd.DataFrame) -> list[str]:
    items = sorted(costs["service_item"].unique())
    flagged = quality.third_party_items(costs)
    st.subheader("Excluded service items")
    st.caption(
        "Excluded items are left out of the cost drivers and the forecast. Marketplace and "
        "other third-party charges are excluded by default: they are spend with other "
        "vendors, billed through your cloud provider, and do not follow your cloud usage. "
        "You can exclude other items too, such as one-off charges."
    )
    excluded = st.multiselect(
        "Excluded from cost drivers and forecast", items, default=flagged,
        key=f"excluded-{hash(tuple(items))}",
    )
    if flagged:
        st.info("Flagged as third-party (marketplace) spend: " + ", ".join(flagged), icon="🏷️")
    return excluded


def cost_drivers(costs: pd.DataFrame, requests: pd.DataFrame | None, report) -> None:
    if requests is None:
        st.info("Upload the requests file in the sidebar to see which costs are driven by requests.")
        return
    if not report.complete_years:
        st.warning(
            "No calendar year (January to December) has all 12 months of both cost and "
            "requests data, so the comparison cannot run yet."
        )
        return

    windows = [str(y) for y in reversed(report.complete_years)] + [drivers.ALL_HISTORY]
    window = st.segmented_control("Time window", windows, default=windows[0]) or windows[0]
    st.caption(
        "Compare one calendar year at a time. Across several years, price changes, "
        "commitments and growth blur the month-to-month relationship."
    )
    thresholds = bucket_rules()
    st.session_state["thresholds"] = thresholds
    results = drivers.analyze(costs, requests, report.complete_years, thresholds)
    view = apply_overrides(results[results["window"] == window])

    st.subheader("Where your spend sits")
    summary = drivers.summarize(view)
    mix = [("Seasonal", drivers.SEASONAL, "Cost that follows requests"),
           ("Linear", drivers.LINEAR, "Cost that grows or shrinks over time"),
           ("Fixed", [drivers.FIXED], "Cost that barely moves")]
    for column, (name, buckets, help_text) in zip(st.columns(3), mix):
        share = summary.loc[summary["bucket"].isin(buckets), "share"].sum()
        column.metric(name, f"{share:.0%} of cost", help=help_text, border=True)
    st.dataframe(
        summary,
        hide_index=True,
        column_config={
            "bucket": "Bucket",
            "items": "Service items",
            "total_cost": st.column_config.NumberColumn("Cost in window", format="dollar"),
            "share": st.column_config.ProgressColumn(
                "Share of cost", min_value=0, max_value=1, format="percent"
            ),
        },
    )

    st.subheader("Service items")
    st.caption("Disagree with a bucket? Pick a different one in the Bucket column. "
               "Your choice applies to every time window.")
    bucket_editor(view, window)

    st.subheader("Look closer at one service item")
    item = st.selectbox("Service item", view["service_item"])
    detail = costs[costs["service_item"] == item].merge(requests, on="month")
    if window != drivers.ALL_HISTORY:
        detail = detail[detail["month"].dt.year == int(window)]
    st.altair_chart(scatter(detail, item), width="stretch")
    left, right = st.columns(2)
    left.altair_chart(over_time(detail, "cost", "Cost per month", MONEY_AXIS, "$,.0f"),
                      width="stretch")
    right.altair_chart(over_time(detail, "requests", "Requests per month", COUNT_AXIS, ",.0f"),
                       width="stretch")
    st.altair_chart(overlay(detail), width="stretch")
    st.caption(
        "Cost and requests are on different scales, so both are shown as a % of their own "
        "average month (100 = average). Lines that rise and fall together mean cost follows requests."
    )


def bucket_rules() -> drivers.Thresholds:
    d = drivers.Thresholds()
    with st.expander("Bucket rules"):
        st.markdown(
            "Each service item lands in the first bucket that fits, in this order:\n"
            "1. **Fixed**: cost barely moves month to month (support, tax, licences).\n"
            "2. **Seasonal: very high / high / mid**: cost follows requests, and so their "
            "seasonal pattern. Grouped by how closely cost follows "
            "requests. Correlation r runs 0 to 1; the request score is r x 10.\n"
            "3. **Linear: growing / declining**: everything else, split by whether cost "
            "went up or down over the window. The *time trend score* is the correlation "
            "between cost and the month number (1, 2, 3 ...), times 10: 10 = a perfectly "
            "straight rise, -10 = a straight fall, 0 = no steady direction. Linear items "
            "scoring between -5 and 5 are marked 'not a steady line' for a closer look."
        )
        a, b, c = st.columns(3)
        very_high = a.number_input("Very high from r =", 0.0, 1.0, d.very_high, 0.05)
        high = b.number_input("High from r =", 0.0, 1.0, d.high, 0.05)
        mid = c.number_input("Mid from r =", 0.0, 1.0, d.mid, 0.05)
        a, _, _ = st.columns(3)
        fixed = a.number_input("Fixed if monthly swing below (%)", 0.0, 50.0,
                               d.fixed_variation * 100, 1.0,
                               help="Standard deviation as a % of the average monthly cost.")
        st.caption(
            "With 12 months of data, a correlation below about 0.58 could be chance. "
            "Such items are marked 'could be chance' in the table."
        )
    if not very_high > high > mid:
        st.error("Bucket rules must satisfy very high > high > mid. Using the defaults.")
        return d
    return drivers.Thresholds(very_high, high, mid, fixed / 100)


def apply_overrides(view: pd.DataFrame) -> pd.DataFrame:
    overrides = st.session_state.setdefault("overrides", {})
    view = view.sort_values("total_cost", ascending=False).copy()
    view["engine_bucket"] = view["bucket"]
    view["bucket"] = [overrides.get(i, b) for i, b in zip(view["service_item"], view["bucket"])]
    view["set_by"] = view["service_item"].isin(overrides).map({True: "You", False: "Engine"})
    chance = view["bucket"].isin(drivers.SEASONAL) & (view["p_value"] > drivers.SIGNIFICANCE)
    irregular = view["bucket"].isin(drivers.LINEAR) & ~(view["trend"].abs() >= drivers.STEADY_TREND)
    view["note"] = ""
    view.loc[chance, "note"] = "⚠️ could be chance"
    view.loc[irregular, "note"] = "⚠️ not a steady line"
    return view


def bucket_editor(view: pd.DataFrame, window: str) -> None:
    overrides = st.session_state["overrides"]
    version = st.session_state.setdefault("override_version", 0)
    columns = ["service_item", "total_cost", "score", "time_score", "bucket", "set_by", "note"]
    edited = st.data_editor(
        view.assign(time_score=view["trend"] * 10)[columns],
        key=f"buckets-{window}-{version}",
        hide_index=True,
        disabled=[c for c in columns if c != "bucket"],
        column_config={
            "service_item": "Service item",
            "total_cost": st.column_config.NumberColumn("Cost in window", format="dollar"),
            "score": st.column_config.ProgressColumn(
                "Request score (0-10)", min_value=0, max_value=10, format="%.1f",
                help="How closely cost follows requests month by month. 10 = moves exactly together.",
            ),
            "time_score": st.column_config.NumberColumn(
                "Time trend score", format="%.1f",
                help="How steadily cost moves in one direction month after month. "
                     "10 = straight rise, -10 = straight fall, 0 = no steady direction.",
            ),
            "bucket": st.column_config.SelectboxColumn(
                "Bucket", options=drivers.BUCKETS, required=True
            ),
            "set_by": "Set by",
            "note": "Check",
        },
    )
    changed = edited["bucket"].to_numpy() != view["bucket"].to_numpy()
    if changed.any():
        for item, bucket, engine in zip(
            view["service_item"][changed], edited["bucket"][changed], view["engine_bucket"][changed]
        ):
            if bucket == engine:
                overrides.pop(item, None)
            else:
                overrides[item] = bucket
        st.session_state["override_version"] = version + 1
        st.rerun()
    if overrides and st.button("Reset my bucket changes"):
        overrides.clear()
        st.session_state["override_version"] = version + 1
        st.rerun()


def scatter(detail: pd.DataFrame, item: str) -> alt.Chart:
    years = sorted(detail["month"].dt.year.unique())[-len(YEAR_COLORS):]
    detail = detail[detail["month"].dt.year.isin(years)].assign(year=lambda d: d["month"].dt.year.astype(str))
    color = (
        alt.Color("year:N", scale=alt.Scale(domain=[str(y) for y in years], range=YEAR_COLORS),
                  legend=alt.Legend(title="Year", orient="top"))
        if len(years) > 1 else alt.value(YEAR_COLORS[0])
    )
    return (
        alt.Chart(detail, title=f"{item}: cost vs requests, one dot per month")
        .mark_circle(size=90, opacity=0.9)
        .encode(
            x=alt.X("requests:Q", title="Requests", axis=alt.Axis(labelExpr=COUNT_AXIS)),
            y=alt.Y("cost:Q", title="Cost", scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelExpr=MONEY_AXIS)),
            color=color,
            tooltip=[
                alt.Tooltip("month:T", title="Month", format="%b %Y"),
                alt.Tooltip("cost:Q", title="Cost", format="$,.0f"),
                alt.Tooltip("requests:Q", title="Requests", format=",.0f"),
            ],
        )
        .properties(height=340)
    )


def over_time(detail: pd.DataFrame, field: str, title: str, axis: str, fmt: str) -> alt.Chart:
    hover = alt.selection_point(fields=["month"], nearest=True, on="pointerover",
                                empty=False, clear="pointerout")
    base = alt.Chart(detail, title=title).encode(
        x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %Y")),
        y=alt.Y(f"{field}:Q", title=None, scale=alt.Scale(zero=False),
                axis=alt.Axis(labelExpr=axis)),
    )
    line = base.mark_line(color=YEAR_COLORS[0], strokeWidth=2)
    targets = base.mark_point(size=400, opacity=0).encode(
        tooltip=[alt.Tooltip("month:T", title="Month", format="%b %Y"),
                 alt.Tooltip(f"{field}:Q", title=title.split(" per")[0], format=fmt)],
    ).add_params(hover)
    rule = alt.Chart(detail).mark_rule(color=MUTED).encode(x="month:T").transform_filter(hover)
    dot = base.mark_circle(size=80, color=YEAR_COLORS[0]).transform_filter(hover)
    return (line + rule + dot + targets).properties(height=240)


def overlay(detail: pd.DataFrame) -> alt.Chart:
    """Cost and requests on one axis, each indexed to its own average month (100)."""
    wide = detail[["month", "cost", "requests"]].copy()
    for field in ("cost", "requests"):
        mean = wide[field].mean()
        wide[f"{field}_index"] = wide[field] / mean * 100 if mean else float("nan")
    long = wide.melt(id_vars="month", value_vars=["cost_index", "requests_index"],
                     var_name="series", value_name="index")
    long["series"] = long["series"].map({"cost_index": "Cost", "requests_index": "Requests"})

    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["Cost", "Requests"], range=YEAR_COLORS[:2]))
    x = alt.X("month:T", title=None, axis=alt.Axis(format="%b %Y"))
    hover = alt.selection_point(fields=["month"], nearest=True, on="pointerover",
                                empty=False, clear="pointerout")

    baseline = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(
        color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=x, y=alt.Y("index:Q", title="% of average month", scale=alt.Scale(zero=False)),
        color=color,
    )
    labels = alt.Chart(end_labels(long)).mark_text(
        align="left", dx=8, color=MUTED).encode(x=x, y="label_y:Q", text="series:N")
    dots = alt.Chart(long).mark_circle(size=80).encode(
        x=x, y="index:Q", color=color).transform_filter(hover)
    rule = alt.Chart(wide).mark_rule(color=MUTED).encode(x=x).transform_filter(hover)
    targets = alt.Chart(wide).mark_rule(strokeWidth=20, opacity=0).encode(
        x=x,
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%b %Y"),
            alt.Tooltip("cost:Q", title="Cost", format="$,.0f"),
            alt.Tooltip("cost_index:Q", title="Cost, % of average", format=".0f"),
            alt.Tooltip("requests:Q", title="Requests", format=",.0f"),
            alt.Tooltip("requests_index:Q", title="Requests, % of average", format=".0f"),
        ],
    ).add_params(hover)
    return alt.layer(baseline, lines, labels, rule, dots, targets).properties(
        title="Cost and requests together", height=300,
    ).configure_view(clip=False)


def end_labels(long: pd.DataFrame, field: str = "index") -> pd.DataFrame:
    """Each series' name at its last point, pushed apart when two lines end close together."""
    ends = long.dropna(subset=[field]).sort_values("month").groupby("series").tail(1)
    ends = ends.sort_values(field).copy()
    gap = (long[field].max() - long[field].min()) * 0.07
    ends["label_y"] = ends[field]
    same_month = ends["month"].nunique() == 1
    if len(ends) == 2 and same_month and ends[field].diff().iloc[1] < gap:
        middle = ends[field].mean()
        ends["label_y"] = [middle - gap / 2, middle + gap / 2]
    return ends


MONEY = "dollar"
BUCKET_COLUMNS = {
    "bucket": "Bucket",
    "items": "Service items",
    "service_item": "Service item",
    "method": "Method",
    "exit_rate": st.column_config.NumberColumn("Exit rate (Dec)", format=MONEY),
    "change": st.column_config.NumberColumn("Change", format="percent"),
    "error": st.column_config.NumberColumn("Forecast off by", format="percent"),
}


def forecast_tab(costs: pd.DataFrame, requests: pd.DataFrame | None, report) -> None:
    years = quality.full_years(pd.DatetimeIndex(costs["month"].unique()))
    if not years:
        st.warning("A baseline needs one full calendar year (January to December) of cost data.")
        return
    options = [str(y) for y in reversed(years)]
    with st.container(border=True):
        st.markdown("**:material/tune: Forecast settings**")
        left, right = st.columns(2)
        baseline = int(left.segmented_control("Baseline year", options, default=options[0],
                                              key="baseline") or options[0])
        target = baseline + 1
        defaults = st.session_state.setdefault("default_growth", {})
        defaults[baseline] = right.number_input(
            f"Growth for {target} (%)",
            min_value=-100.0, max_value=1000.0, value=defaults.get(baseline, 0.0), step=1.0,
            key=f"default-growth-{baseline}",
            help="Applied to every service item you have not set yourself below.",
        )
        st.caption(f"Forecast for {target}, starting from the December {baseline} exit rate. "
                   "Each baseline year keeps its own settings.")
        seasonal_controls(costs, requests, report, baseline, target)
        forecast_methods(baseline)
    _, growth, result, summary = scenario(costs, requests, report, baseline)
    if result.empty:
        st.warning(f"No service item has a cost in December {baseline} to start from.")
        return

    a, b, c, d = st.columns(4)
    base_total, fc_total = summary["baseline_total"].sum(), summary["forecast_total"].sum()
    a.metric(f"Actual {baseline}", f"${base_total:,.0f}", border=True)
    b.metric(f"Forecast {target}", f"${fc_total:,.0f}",
             delta=f"{fc_total / base_total - 1:+.1%} vs {baseline}" if base_total else None,
             border=True)
    exit_total = summary["exit_rate"].sum()
    c.metric(f"Exit rate (Dec {baseline})", f"${exit_total:,.0f}", border=True)
    if "actual_total" in summary:
        actual = summary["actual_total"].sum()
        d.metric(f"Actual {target}", f"${actual:,.0f}",
                 delta=f"forecast off by {fc_total / actual - 1:+.1%}", delta_color="off",
                 border=True)

    items = costs["service_item"].unique()
    choice = st.selectbox("Show", ["All service items", *items], key="forecast_item")
    picked = costs if choice == "All service items" else costs[costs["service_item"] == choice]
    picked_fc = result if choice == "All service items" else result[result["service_item"] == choice]
    st.altair_chart(forecast_chart(picked, picked_fc, baseline), width="stretch")

    bucket_breakdown(summary, baseline, target)

    seasonal_tab, growth_tab, baselines_tab, monthly_tab = st.tabs([
        "Compare seasonal methods", "Growth per service item", "Compare baselines",
        f"Monthly forecast {target}",
    ])
    with seasonal_tab:
        compare_seasonal_methods(costs, requests, report, baseline, target)
    with growth_tab:
        growth_section(costs, summary, growth, years, baseline)
    with baselines_tab:
        compare_baselines(costs, requests, report, years)
    with monthly_tab:
        monthly_table(result, target)


def growth_section(costs, summary, growth, years: list[int], baseline: int) -> None:
    how = st.segmented_control(
        "Growth within each past year",
        [forecast.AVERAGE_MOM, forecast.JAN_TO_DEC],
        default=forecast.AVERAGE_MOM, key="growth-indicator",
    ) or forecast.AVERAGE_MOM
    st.caption(
        "Type a growth % for any service item to play with the forecast. "
        + ("The year columns show the average of the 11 month-over-month changes "
           "(Feb vs Jan ... Dec vs Nov)." if how == forecast.AVERAGE_MOM else
           "The year columns compare December with January. Annualising both (x 12) "
           "gives the same %.")
    )
    custom = st.session_state.setdefault("growth", {}).setdefault(baseline, {})
    label = "Avg MoM" if how == forecast.AVERAGE_MOM else "Jan vs Dec"
    growth_editor(summary, growth, custom, forecast.in_year_growth(costs, years, how),
                  baseline, label)


def monthly_table(result: pd.DataFrame, target: int) -> None:
    monthly = result.pivot_table(index="service_item", columns="month", values="forecast",
                                 aggfunc="sum")
    monthly.loc["Total"] = monthly.sum()
    monthly.columns = [f"{m:%b %Y}" for m in monthly.columns]
    monthly["Year"] = monthly.sum(axis=1)
    st.dataframe(monthly.style.format("${:,.0f}"))
    st.download_button(
        "Download forecast (CSV)",
        result.to_csv(index=False).encode(),
        file_name=f"forecast_{target}.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def forecast_methods(baseline: int) -> None:
    with st.expander("How each bucket is forecast"):
        st.markdown(
            f"The **exit rate** is the cost in December {baseline}.\n\n"
            "- **Seasonal**, with weights (request, manual or own cost): exit rate ÷ "
            "December's weight gives a yearly figure. Each month = yearly figure × "
            "(1 + growth) × that month's weight.\n"
            "- **Seasonal**, fixed + variable: a fixed cost per month plus a cost per "
            f"request, fitted on {baseline} (like LINEST in Excel or Google Sheets). Each "
            "month = fixed + cost per request × forecast requests. Forecast requests = "
            f"{baseline} requests × (1 + request growth).\n"
            "- **Linear**: each month builds on the previous one, adding Z = ((exit rate × 12) "
            "× (1 + growth) − (exit rate × 12)) ÷ 78. Each month is then ÷ 30.4 "
            "(an average month) × its number of days.\n"
            "- **Fixed**: the same as linear, without the day scaling (billed per month)."
        )


def seasonal_controls(costs, requests, report, baseline: int, target: int) -> None:
    if requests is None:
        return
    method = st.segmented_control(
        "Forecast seasonal items with", forecast.SEASONAL_METHODS,
        default=forecast.REQUEST_WEIGHTS, key="seasonal-method",
    ) or forecast.REQUEST_WEIGHTS
    if method == forecast.REQUEST_WEIGHTS:
        weights = forecast.request_weights(requests, baseline)
        st.caption(f"Calculated from your requests file: each month's share of {baseline} "
                   "requests. The same % apply to every seasonal service item.")
        if weights is not None:
            weights_row(weights)
    elif method == forecast.MANUAL_WEIGHTS:
        st.caption("Type your own % for each month. They apply to every seasonal service "
                   f"item. They start as the {baseline} request weights.")
        manual_weights_editor(requests, baseline)
    elif method == forecast.OWN_WEIGHTS:
        st.caption(
            f"Each service item uses its own monthly share of its {baseline} cost, so the % "
            f"differ per item. December ÷ its own December weight = its {baseline} total, so "
            "growth is applied to last year's actual."
        )
        per_item_table(costs, requests, report, baseline, own_weight_rows)
    else:
        rates = st.session_state.setdefault("request_growth", {})
        rates[baseline] = st.number_input(
            f"Request growth for {target} (%)", min_value=-100.0, max_value=1000.0,
            value=rates.get(baseline, 0.0), step=1.0, key=f"request-growth-{baseline}",
            help=f"Forecast requests = {baseline} requests × (1 + request growth).",
        )
        st.caption("Growth comes from requests, so the per-item growth % below does not "
                   f"apply to seasonal items. Fixed and per-request cost, fitted on {baseline}:")
        per_item_table(costs, requests, report, baseline, fixed_variable_rows)


MONTH_NAMES = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)]


def weights_row(weights: pd.Series) -> None:
    row = dict(zip(MONTH_NAMES, weights.to_numpy())) | {"FY": weights.sum()}
    st.dataframe(pd.DataFrame([row]), hide_index=True, column_config={
        c: st.column_config.NumberColumn(c, format="percent") for c in row})


def manual_weights(requests, baseline: int) -> pd.Series | None:
    saved = st.session_state.get("manual_weights", {}).get(baseline)
    return saved if saved is not None else forecast.request_weights(requests, baseline)


def manual_weights_editor(requests, baseline: int) -> None:
    weights = manual_weights(requests, baseline)
    if weights is None:
        weights = pd.Series(1 / 12, index=range(1, 13))
    version = st.session_state.setdefault("manual_version", 0)
    shown = pd.DataFrame([dict(zip(MONTH_NAMES, (weights * 100).round(2).to_numpy()))])
    edited = st.data_editor(
        shown, hide_index=True, key=f"manual-weights-{baseline}-{version}",
        column_config={m: st.column_config.NumberColumn(m, format="%.2f%%", min_value=0)
                       for m in MONTH_NAMES},
    )
    if not edited.equals(shown):
        values = edited.iloc[0].fillna(0).to_numpy(dtype=float) / 100
        st.session_state.setdefault("manual_weights", {})[baseline] = pd.Series(
            values, index=range(1, 13))
        st.session_state["manual_version"] = version + 1
        st.rerun()
    total = weights.sum() * 100
    if abs(total - 100) > 0.01:
        st.warning(f"The months add up to {total:.2f}%, not 100%, so the year's total will "
                   f"be {total:.2f}% of the yearly figure.")
    else:
        st.caption("FY: 100%")


def own_weight_rows(costs, requests, buckets, baseline: int) -> pd.DataFrame:
    rows = []
    for item, bucket in buckets.items():
        if bucket in drivers.SEASONAL:
            weights = forecast.own_weights(costs[costs["service_item"] == item], baseline)
            if weights is not None:
                rows.append({"service_item": item, **dict(zip(MONTH_NAMES, weights))})
    return pd.DataFrame(rows)


def fixed_variable_rows(costs, requests, buckets, baseline: int) -> pd.DataFrame:
    reqs = requests[requests["month"].dt.year == baseline].set_index("month")["requests"]
    rows = []
    for item, bucket in buckets.items():
        if bucket not in drivers.SEASONAL:
            continue
        cost = costs[costs["service_item"] == item].set_index("month")["cost"]
        cost = cost[cost.index.year == baseline]
        if len(cost) == 12 and len(reqs) == 12:
            per_request, fixed = forecast.fit_fixed_variable(reqs.reindex(cost.index), cost)
            rows.append({"service_item": item, "fixed": fixed, "per_request": per_request})
    return pd.DataFrame(rows)


def per_item_table(costs, requests, report, baseline: int, rows_for) -> None:
    table = rows_for(costs, requests, item_buckets(costs, requests, report, baseline), baseline)
    if table.empty:
        st.caption("No seasonal service items in this baseline year.")
        return
    config = {"service_item": "Service item",
              "fixed": st.column_config.NumberColumn("Fixed per month", format=MONEY),
              "per_request": st.column_config.NumberColumn("Cost per request", format="%.12f")}
    config |= {m: st.column_config.NumberColumn(m, format="percent") for m in MONTH_NAMES}
    st.dataframe(table, hide_index=True, column_config=config)


def seasonal_settings(requests, baseline: int, method: str | None = None):
    """The seasonal method and its inputs saved for ``baseline``."""
    if requests is None:
        return None
    method = method or st.session_state.get("seasonal-method") or forecast.REQUEST_WEIGHTS
    weights = (manual_weights(requests, baseline) if method == forecast.MANUAL_WEIGHTS
               else forecast.request_weights(requests, baseline))
    growth = st.session_state.get("request_growth", {}).get(baseline, 0.0) / 100
    return forecast.Seasonal(method, weights, requests, growth)


def growth_for(costs: pd.DataFrame, baseline: int) -> dict[str, float]:
    default = st.session_state.get("default_growth", {}).get(baseline, 0.0) / 100
    custom = st.session_state.get("growth", {}).get(baseline, {})
    return {item: custom.get(item, default) for item in costs["service_item"].unique()}


def scenario(costs, requests, report, baseline: int):
    """Buckets, growth, monthly forecast and per-item summary for one baseline year,
    using the settings saved for that year."""
    buckets = item_buckets(costs, requests, report, baseline)
    growth = growth_for(costs, baseline)
    result = forecast.build(costs, baseline, buckets, growth,
                            seasonal=seasonal_settings(requests, baseline))
    return buckets, growth, result, forecast.summarize(costs, result, baseline)


def compare_seasonal_methods(costs, requests, report, baseline: int, target: int) -> None:
    if requests is None:
        return
    buckets = item_buckets(costs, requests, report, baseline)
    methods = {m: seasonal_settings(requests, baseline, m) for m in forecast.SEASONAL_METHODS}
    table = forecast.compare_seasonal(
        costs, baseline, buckets, growth_for(costs, baseline), methods, requests,
        st.session_state.get("request_growth", {}).get(baseline, 0.0) / 100,
    )
    if table.empty or forecast.ALL_SEASONAL not in table:
        st.caption("No seasonal service items in this baseline year.")
        return
    st.caption(
        f"The {target} forecast for the seasonal buckets under each method, with the "
        "growth and settings you have entered. Fixed + variable per bucket fits one line to "
        "each bucket's total cost instead of to each item."
        + (f" The Actual row is what {target} really cost." if "Off by" in table else "")
    )
    config = {"method": "Method", "Off by": st.column_config.NumberColumn(format="percent")}
    config |= {c: st.column_config.NumberColumn(c, format=MONEY)
               for c in table.columns if c not in config}
    st.dataframe(table, hide_index=True, column_config=config)


def money_columns(baseline: int, target: int) -> dict:
    return {
        "baseline_total": st.column_config.NumberColumn(f"Actual {baseline}", format=MONEY),
        "forecast_total": st.column_config.NumberColumn(f"Forecast {target}", format=MONEY),
        "actual_total": st.column_config.NumberColumn(f"Actual {target}", format=MONEY),
    }


def bucket_breakdown(summary: pd.DataFrame, baseline: int, target: int) -> None:
    st.subheader("By bucket")
    st.caption("Click a bucket to see the service items in it.")
    table = forecast.by_bucket(summary)
    config = BUCKET_COLUMNS | money_columns(baseline, target)
    event = st.dataframe(
        table, hide_index=True, column_config=config, key=f"bucket-table-{baseline}",
        on_select="rerun", selection_mode="single-row",
    )
    rows = event.selection.rows
    if not rows or table.loc[rows[0], "bucket"] == forecast.TOTAL:
        return
    bucket = table.loc[rows[0], "bucket"]
    items = summary[summary["bucket"] == bucket].sort_values("forecast_total", ascending=False)
    columns = [c for c in ("service_item", "method", "baseline_total", "exit_rate", "forecast_total",
                           "change", "actual_total", "error") if c in items]
    st.markdown(f"**{bucket}**: {len(items)} service item{'s' if len(items) != 1 else ''}")
    st.dataframe(items[columns], hide_index=True, column_config=config)


def compare_baselines(costs, requests, report, years: list[int]) -> None:
    tables, notes = {}, []
    for year in years:
        *_, summary = scenario(costs, requests, report, year)
        if summary.empty:
            continue
        tables[year] = forecast.by_bucket(summary)
        default = st.session_state.get("default_growth", {}).get(year, 0.0)
        custom = len(st.session_state.get("growth", {}).get(year, {}))
        notes.append(f"{year}: {default:g}%" + (f" (+{custom} items set by you)" if custom else ""))
    st.caption(
        "The forecast from each baseline year, with the growth saved for it ("
        + "; ".join(notes) + "). Where the forecast year has already happened, the actual "
        "cost and the error show which baseline forecast best."
    )
    table = forecast.compare(tables)
    config = {"bucket": "Bucket"} | {
        c: st.column_config.NumberColumn(c, format="percent" if c.startswith("Off by") else MONEY)
        for c in table.columns if c != "bucket"
    }
    st.dataframe(table, hide_index=True, column_config=config)


def item_buckets(costs, requests, report, baseline: int) -> dict[str, str]:
    """Each item's bucket in the baseline year, including manual changes."""
    if requests is None or baseline not in report.complete_years:
        return {}
    thresholds = st.session_state.get("thresholds", drivers.DEFAULT_THRESHOLDS)
    results = drivers.analyze(costs, requests, [baseline], thresholds)
    view = apply_overrides(results[results["window"] == str(baseline)])
    return dict(zip(view["service_item"], view["bucket"]))


def growth_editor(summary, growth, custom, history, baseline: int, label: str) -> None:
    target = baseline + 1
    table = summary.merge(history, on="service_item", how="left")
    table["growth"] = table["service_item"].map(growth) * 100
    table["set_by"] = table["service_item"].isin(custom).map({True: "You", False: "Default"})
    table = table.sort_values("baseline_total", ascending=False, ignore_index=True)

    years = [c for c in history.columns if c != "service_item"]
    columns = ["service_item", "bucket", *years, "baseline_total", "exit_rate",
               "growth", "set_by", "forecast_total", "change"]
    config = {
        "service_item": "Service item",
        "bucket": "Bucket",
        **{y: st.column_config.NumberColumn(f"{label} {y}", format="percent") for y in years},
        "baseline_total": st.column_config.NumberColumn("Baseline year", format="dollar"),
        "exit_rate": st.column_config.NumberColumn("Exit rate (Dec)", format="dollar"),
        "growth": st.column_config.NumberColumn(f"Growth {target} (%)", format="%.1f",
                                                min_value=-100, step=0.5),
        "set_by": "Set by",
        "forecast_total": st.column_config.NumberColumn(f"Forecast {target}", format="dollar"),
        "change": st.column_config.NumberColumn("vs baseline year", format="percent"),
    }
    if "actual_total" in table:
        columns += ["actual_total", "error"]
        config["actual_total"] = st.column_config.NumberColumn(f"Actual {target}", format="dollar")
        config["error"] = st.column_config.NumberColumn("Forecast off by", format="percent")

    version = st.session_state.setdefault("growth_version", 0)
    edited = st.data_editor(
        table[columns], key=f"growth-{baseline}-{version}", hide_index=True,
        disabled=[c for c in columns if c != "growth"], column_config=config,
    )
    changed = edited["growth"].to_numpy() != table["growth"].to_numpy()
    if changed.any():
        for item, value in zip(table["service_item"][changed], edited["growth"][changed]):
            custom[item] = (value or 0.0) / 100
        st.session_state["growth_version"] = version + 1
        st.rerun()
    if custom and st.button("Reset my growth changes"):
        custom.clear()
        st.session_state["growth_version"] = version + 1
        st.rerun()


def forecast_chart(costs: pd.DataFrame, result: pd.DataFrame, baseline: int) -> alt.Chart:
    years = costs["month"].dt.year
    actual = costs[years.isin([baseline, baseline + 1])].groupby("month")["cost"].sum()
    predicted = result.groupby("month")["forecast"].sum()
    exit_month = pd.Timestamp(year=baseline, month=12, day=1)
    # Start the forecast line at the exit rate so the two lines join up.
    predicted = pd.concat([actual.loc[[exit_month]], predicted])
    wide = pd.DataFrame({"actual": actual, "forecast": predicted}).rename_axis("month").reset_index()
    long = wide.melt(id_vars="month", var_name="series", value_name="cost").dropna()
    long["series"] = long["series"].map({"actual": "Actual", "forecast": "Forecast"})

    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["Actual", "Forecast"], range=YEAR_COLORS[:2]))
    dash = alt.StrokeDash("series:N", legend=None,
                          scale=alt.Scale(domain=["Actual", "Forecast"], range=[[1, 0], [6, 4]]))
    x = alt.X("month:T", title=None, axis=alt.Axis(format="%b %Y"))
    y = alt.Y("cost:Q", title=None, scale=alt.Scale(zero=False), axis=alt.Axis(labelExpr=MONEY_AXIS))
    hover = alt.selection_point(fields=["month"], nearest=True, on="pointerover",
                                empty=False, clear="pointerout")

    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(x=x, y=y, color=color, strokeDash=dash)
    labels = alt.Chart(end_labels(long, "cost")).mark_text(
        align="left", dx=8, color=MUTED).encode(x=x, y=alt.Y("label_y:Q"), text="series:N")
    dots = alt.Chart(long).mark_circle(size=80).encode(
        x=x, y=y, color=color).transform_filter(hover)
    rule = alt.Chart(wide).mark_rule(color=MUTED).encode(x=x).transform_filter(hover)
    targets = alt.Chart(wide).mark_rule(strokeWidth=20, opacity=0).encode(
        x=x,
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%b %Y"),
            alt.Tooltip("actual:Q", title="Actual", format="$,.0f"),
            alt.Tooltip("forecast:Q", title="Forecast", format="$,.0f"),
        ],
    ).add_params(hover)
    return alt.layer(lines, labels, rule, dots, targets).properties(
        title="Monthly cost: actual and forecast", height=320,
    ).configure_view(clip=False)


def main() -> None:
    st.set_page_config(page_title="Cloud Forecasting Engine", page_icon="☁️", layout="wide")
    st.title(":material/cloud: Cloud Forecasting Engine")
    st.caption("An open forecasting engine for FinOps teams: clean your cloud cost data, "
               "find what drives each cost, and forecast the next 12 months.")

    try:
        costs, requests = sidebar()
    except ingest.DataError as exc:
        st.error(f"We could not read that file: {exc}")
        return
    if costs is None or costs.empty:
        welcome()
        return

    report = quality.check(costs, requests)
    costs = quality.fill_missing_months(costs)
    check_tab, drivers_tab, forecast_tab_ = st.tabs([
        ":material/fact_check: Data check", ":material/hub: Cost drivers",
        ":material/trending_up: Forecast",
    ])
    with check_tab:
        excluded = data_check(costs, report)
    costs = costs[~costs["service_item"].isin(excluded)]
    if not report.ok or costs.empty:
        message = ("Fix the problems on the Data check tab first." if not report.ok
                   else "Every service item is excluded. Include some on the Data check tab.")
        for tab in (drivers_tab, forecast_tab_):
            tab.error(message)
        return
    with drivers_tab:
        cost_drivers(costs, requests, report)
    with forecast_tab_:
        forecast_tab(costs, requests, report)


if __name__ == "__main__":
    main()
