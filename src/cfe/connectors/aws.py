"""AWS Cost Explorer connector.

Credentials come from the standard AWS chain (named profile, environment, SSO);
the engine never asks for or stores access keys. The caller needs the
``ce:GetCostAndUsage`` permission. Each Cost Explorer API page costs $0.01.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from cfe.ingest import normalize_costs

# Cost Explorer can group by two dimensions: the service, plus one finer "item" dimension.
ITEM_DIMENSIONS = ["USAGE_TYPE", "OPERATION", "INSTANCE_TYPE"]
METRICS = ["UnblendedCost", "AmortizedCost", "NetAmortizedCost"]


def fetch_costs(
    start: date,
    end: date,
    profile: str | None = None,
    item_dimension: str = "USAGE_TYPE",
    metric: str = "UnblendedCost",
    include_marketplace: bool = False,
    client=None,
) -> pd.DataFrame:
    """Return monthly cost per service item between ``start`` and ``end`` (exclusive).

    AWS Marketplace purchases (spend with other vendors) are left out unless
    ``include_marketplace`` is set.
    """
    if item_dimension not in ITEM_DIMENSIONS:
        raise ValueError(f"item_dimension must be one of {ITEM_DIMENSIONS}")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    if client is None:
        import boto3  # optional dependency: pip install cloud-forecasting-engine[aws]

        client = boto3.Session(profile_name=profile).client("ce", region_name="us-east-1")

    request = {
        "TimePeriod": {"Start": start.isoformat(), "End": end.isoformat()},
        "Granularity": "MONTHLY",
        "Metrics": [metric],
        "GroupBy": [
            {"Type": "DIMENSION", "Key": "SERVICE"},
            {"Type": "DIMENSION", "Key": item_dimension},
        ],
    }
    if not include_marketplace:
        request["Filter"] = {
            "Not": {"Dimensions": {"Key": "BILLING_ENTITY", "Values": ["AWS Marketplace"]}}
        }
    rows = []
    while True:
        page = client.get_cost_and_usage(**request)
        for period in page["ResultsByTime"]:
            for group in period.get("Groups", []):
                service, item = group["Keys"]
                rows.append({
                    "month": period["TimePeriod"]["Start"],
                    "service": service,
                    "service_item": f"{service}: {item}",
                    "cost": group["Metrics"][metric]["Amount"],
                })
        token = page.get("NextPageToken")
        if not token:
            break
        request["NextPageToken"] = token

    if not rows:
        return pd.DataFrame(columns=["month", "service", "service_item", "cost"])
    return normalize_costs(pd.DataFrame(rows))
