"""Load cost and request data from CSV files into one clean, canonical shape.

Canonical cost table:      month | service | service_item | cost
Canonical requests table:  month | requests

``month`` is always the first day of the month. Both "long" files (one row per
month) and "wide" spreadsheets (one column per month) are accepted.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import IO

import pandas as pd

COST_COLUMNS = ["month", "service", "service_item", "cost"]
REQUEST_COLUMNS = ["month", "requests"]

# Column names we recognise, after lower-casing and stripping spaces/underscores/dashes.
ALIASES = {
    "month": {
        "month", "date", "period", "billingperiod", "billingmonth", "usagemonth",
        "billingperiodstart", "startdate", "usagestartdate", "chargeperiodstart",
    },
    "service": {"service", "servicename", "product", "productname"},
    "service_item": {
        "serviceitem", "item", "subservice", "usagetype", "usagetypegroup",
        "servicesubcategory", "subcategory",
    },
    "cost": {
        "cost", "amount", "spend", "unblendedcost", "amortizedcost", "billedcost",
        "effectivecost", "netcost",
    },
    "requests": {"requests", "requestcount", "apicalls", "calls", "count", "volume"},
}


class DataError(ValueError):
    """Raised when an input file cannot be turned into clean data."""


EXCEL = (".xlsx", ".xls")
FILE_TYPES = ["csv", "tsv", "txt", "xlsx", "xls"]


def load_costs(source: str | IO, name: str | None = None) -> pd.DataFrame:
    """Read a cost file (long or wide) and return the canonical cost table."""
    return normalize_costs(read_table(source, name))


def load_requests(source: str | IO, name: str | None = None) -> pd.DataFrame:
    """Read a requests file (long or wide) and return the canonical requests table."""
    return normalize_requests(read_table(source, name))


def read_table(source: str | IO, name: str | None = None) -> pd.DataFrame:
    """Read CSV, tab- or semicolon-separated text, or Excel, keeping every value as text.

    ``name`` is the file name, used to spot Excel files when ``source`` is a stream.
    """
    name = name or (str(source) if isinstance(source, (str, Path)) else "")
    if name.lower().endswith(EXCEL):
        return pd.read_excel(source, dtype=str)
    raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else source.read()
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    header = next((line for line in text.splitlines() if line.strip()), "")
    # Spreadsheets copied or saved as text use tabs; some locales use semicolons.
    if "\t" in header:
        sep = "\t"
    elif header.count(";") > header.count(","):
        sep = ";"
    else:
        sep = ","
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str)


def normalize_costs(raw: pd.DataFrame) -> pd.DataFrame:
    df = _rename_known_columns(raw)
    if "month" not in df.columns:
        df = _wide_to_long(df, value_name="cost")
    if "cost" not in df.columns:
        raise DataError("Could not find a cost column (e.g. 'cost', 'amount', 'BilledCost').")
    if "service_item" not in df.columns and "service" not in df.columns:
        raise DataError("Could not find a service or service item column.")

    if "service_item" not in df.columns:
        df["service_item"] = df["service"]
    df["service_item"] = df["service_item"].astype(str).str.strip()
    if "service" not in df.columns:
        # "EC2: Compute Instance" -> service "EC2"
        df["service"] = df["service_item"].str.split(":", n=1).str[0]
    df["service"] = df["service"].astype(str).str.strip()

    df["month"] = _parse_months(df["month"])
    df["cost"] = _parse_numbers(df["cost"], "cost")

    # Daily or line-item exports collapse to one row per item per month.
    out = df.groupby(["month", "service", "service_item"], as_index=False)["cost"].sum()
    return out[COST_COLUMNS].sort_values(["service", "service_item", "month"], ignore_index=True)


def normalize_requests(raw: pd.DataFrame) -> pd.DataFrame:
    df = _rename_known_columns(raw)
    if "month" not in df.columns:
        df = _wide_to_long(df, value_name="requests")
    if "requests" not in df.columns:
        raise DataError("Could not find a requests column (e.g. 'requests', 'api_calls').")

    df["month"] = _parse_months(df["month"])
    df["requests"] = _parse_numbers(df["requests"], "requests")
    out = df.groupby("month", as_index=False)["requests"].sum()
    return out[REQUEST_COLUMNS].sort_values("month", ignore_index=True)


def _key(name: str) -> str:
    return re.sub(r"[\s_\-]", "", str(name)).lower()


def _rename_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    renames = {}
    for col in df.columns:
        for canonical, aliases in ALIASES.items():
            if _key(col) in aliases and canonical not in renames.values():
                renames[col] = canonical
                break
    return df.rename(columns=renames)


def _wide_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Turn a spreadsheet with one column per month into one row per month."""
    month_cols = [c for c in df.columns if _looks_like_month(c)]
    if not month_cols:
        raise DataError(
            "Could not find a month column, or month headers like '2023-01-01', in the file."
        )
    id_cols = [c for c in df.columns if c not in month_cols and c != value_name]
    long = df.melt(id_vars=id_cols, value_vars=month_cols, var_name="month", value_name=value_name)
    # A wide file without labelled columns: treat the first label column as the item.
    if id_cols and value_name == "cost" and not {"service", "service_item"} & set(id_cols):
        long = long.rename(columns={id_cols[0]: "service_item"})
    return long.dropna(subset=[value_name])


def _looks_like_month(value: object) -> bool:
    text = str(value).strip()
    if not re.search(r"\d{4}", text):
        return False
    return not pd.isna(pd.to_datetime(text, errors="coerce", format="mixed"))


def _parse_months(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values.astype(str).str.strip(), errors="coerce", format="mixed")
    bad = values[parsed.isna()]
    if not bad.empty:
        examples = ", ".join(repr(v) for v in bad.head(3))
        raise DataError(f"{len(bad)} rows have a month we could not read, e.g. {examples}.")
    return parsed.dt.to_period("M").dt.to_timestamp()


def _parse_numbers(values: pd.Series, label: str) -> pd.Series:
    """Parse '$1,234.50', '1 234', '(12.00)' (negative) and plain numbers."""
    text = values.astype(str).str.strip()
    negative = text.str.match(r"^\(.*\)$")
    cleaned = text.str.replace(r"[^\d.\-eE]", "", regex=True)
    numbers = pd.to_numeric(cleaned, errors="coerce")
    bad = values[numbers.isna() & values.notna() & (text != "")]
    if not bad.empty:
        examples = ", ".join(repr(v) for v in bad.head(3))
        raise DataError(f"{len(bad)} {label} values are not numbers, e.g. {examples}.")
    numbers = numbers.fillna(0.0)
    return numbers.where(~negative, -numbers.abs())
