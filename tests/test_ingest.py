import io

import pandas as pd
import pytest

from cfe.ingest import DataError, load_costs, load_requests


def csv(text: str) -> io.StringIO:
    return io.StringIO(text.strip())


def test_long_costs_with_aliases_and_currency_strings():
    df = load_costs(csv("""
Billing Month,Service Name,Usage Type,Amount
2023-01-15,EC2,Compute Instance,"$1,000.50"
2023-01-20,EC2,Compute Instance,$500
2023-02-01,EC2,Compute Instance,(25.00)
"""))
    assert list(df.columns) == ["month", "service", "service_item", "cost"]
    assert df["month"].tolist() == [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")]
    assert df["cost"].tolist() == [1500.50, -25.0]  # daily rows summed, (x) is negative


def test_wide_costs_split_service_from_item():
    df = load_costs(csv("""
service_item,2023-01-01,2023-02-01
EC2: Compute Instance,"$812,345.67","$698,765.43"
S3: Storage,100,200
"""))
    assert set(df["service"]) == {"EC2", "S3"}
    ec2 = df[df["service_item"] == "EC2: Compute Instance"]
    assert ec2["cost"].tolist() == [812345.67, 698765.43]


def test_wide_requests_single_row():
    df = load_requests(csv("""
,2023-01-01,2023-02-01
requests," 123,456,789,012 "," 134,567,890,123 "
"""))
    assert df["requests"].tolist() == [123456789012, 134567890123]


def test_unreadable_month_is_reported():
    with pytest.raises(DataError, match="month we could not read"):
        load_costs(csv("""
month,service_item,cost
not-a-date,EC2: Compute,10
"""))


def test_non_numeric_cost_is_reported():
    with pytest.raises(DataError, match="not numbers"):
        load_costs(csv("""
month,service_item,cost
2023-01-01,EC2: Compute,ten dollars
"""))


REQUESTS_ROWS = [("2023-01-01", " 123,456,789,012 "), ("2023-02-01", " 134,567,890,123 ")]
EXPECTED = [123456789012, 134567890123]


@pytest.mark.parametrize("sep", ["\t", ";", ","])
def test_requests_with_any_separator(sep):
    quote = '"' if sep == "," else ""
    text = f"Month{sep}Requests\n" + "\n".join(
        f"{m}{sep}{quote}{r}{quote}" for m, r in REQUESTS_ROWS)
    df = load_requests(io.BytesIO(text.encode()))
    assert df["requests"].tolist() == EXPECTED


def test_byte_order_mark_from_excel_is_ignored():
    text = "﻿Month,Requests\n" + "\n".join(f'{m},"{r}"' for m, r in REQUESTS_ROWS)
    assert load_requests(io.BytesIO(text.encode("utf-8"))).columns[0] == "month"


def test_excel_file(tmp_path):
    path = tmp_path / "requests.xlsx"
    pd.DataFrame(REQUESTS_ROWS, columns=["Month", "Requests"]).to_excel(path, index=False)
    assert load_requests(str(path))["requests"].tolist() == EXPECTED
    with open(path, "rb") as stream:  # an upload: a stream plus its file name
        assert load_requests(stream, "requests.xlsx")["requests"].tolist() == EXPECTED
