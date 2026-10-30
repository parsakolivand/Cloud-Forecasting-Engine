from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "src" / "cfe" / "app.py")


def test_welcome_screen_without_data():
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert any("Choose your cost data" in i.value for i in at.info)


def test_sample_data_runs_end_to_end():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Try sample data").run()
    assert not at.exception
    assert at.metric[0].value == "36"  # months of history
    assert at.metric[3].value == "3"  # complete calendar years

    at.selectbox[0].set_value("S3: Storage").run()
    assert not at.exception


def test_bucket_override_applies_and_resets():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Try sample data").run()
    assert "Reset my bucket changes" not in [b.label for b in at.button]

    # What the table widget reports when a user changes row 0's bucket to Fixed.
    at.session_state["buckets-2025-0"] = {
        "edited_rows": {"0": {"bucket": "Fixed"}}, "added_rows": [], "deleted_rows": [],
    }
    at.run()
    assert not at.exception
    assert at.session_state["overrides"] == {"EC2: Compute Instance": "Fixed"}

    next(b for b in at.button if b.label == "Reset my bucket changes").click()
    at.run()
    assert at.session_state["overrides"] == {}


def test_forecast_tab_growth_and_backtest():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Try sample data").run()
    labels = [m.label for m in at.metric]
    assert "Forecast 2026" in labels

    at.number_input(key="default-growth-2025").set_value(7.0).run()
    at.session_state["growth-2025-0"] = {
        "edited_rows": {"0": {"growth": 20.0}}, "added_rows": [], "deleted_rows": [],
    }
    at.run()
    assert not at.exception
    assert at.session_state["growth"] == {2025: {"EC2: Compute Instance": 0.2}}

    # A past baseline can be checked against what actually happened.
    at.segmented_control(key="baseline").set_value("2024").run()
    assert not at.exception
    assert "Actual 2025" in [m.label for m in at.metric]
    assert at.number_input(key="default-growth-2024").value == 0.0

    # Going back to 2025 restores the growth set for it.
    at.segmented_control(key="baseline").set_value("2025").run()
    assert at.number_input(key="default-growth-2025").value == 7.0
    assert at.session_state["growth"][2025] == {"EC2: Compute Instance": 0.2}


def test_clicking_a_bucket_shows_its_items():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Try sample data").run()
    at.session_state["bucket-table-2025"] = {"selection": {"rows": [0], "columns": [], "cells": []}}
    at.run()
    assert not at.exception
    assert any(m.value.startswith("**Seasonal: very high**: 3 service items")
               for m in at.markdown)
    compare = next(d.value for d in at.dataframe if "Forecast 2026 (from 2025)" in d.value.columns)
    assert {"Forecast 2024 (from 2023)", "Actual 2024", "Off by (2024)"} <= set(compare.columns)


def test_marketplace_is_excluded_by_default():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Try sample data").run()
    assert at.multiselect[0].value == ["Marketplace: Software"]
    items = {i for d in at.dataframe if "service_item" in d.value.columns
             for i in d.value["service_item"]}
    assert "Marketplace: Software" not in items

    at.multiselect[0].set_value([]).run()  # include it again
    assert not at.exception
    items = {i for d in at.dataframe if "service_item" in d.value.columns
             for i in d.value["service_item"]}
    assert "Marketplace: Software" in items


def test_root_entry_point_runs():
    at = AppTest.from_file(str(Path(__file__).parents[1] / "streamlit_app.py")).run()
    assert not at.exception


def test_forecast_two_years_ahead():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Try sample data").run()
    at.selectbox(key="target-2025").set_value(2027).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Forecast 2027" in labels and "Actual 2025" in labels

    # From 2023, two years ahead is 2025, which can be checked against actuals.
    at.segmented_control(key="baseline").set_value("2023").run()
    at.selectbox(key="target-2023").set_value(2025).run()
    assert not at.exception
    assert "Actual 2025" in [m.label for m in at.metric]
    compare = next(d.value for d in at.dataframe if "Forecast 2025 (from 2023)" in d.value.columns)
    assert "Forecast 2027 (from 2025)" in compare.columns

    # Going back to 2025 remembers 2027.
    at.segmented_control(key="baseline").set_value("2025").run()
    assert at.selectbox(key="target-2025").value == 2027
