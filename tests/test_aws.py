from datetime import date

import pytest

from cfe.connectors.aws import fetch_costs


class FakeCostExplorer:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get_cost_and_usage(self, **request):
        self.calls.append(dict(request))
        return self.pages[len(self.calls) - 1]


def group(service, item, amount):
    return {"Keys": [service, item], "Metrics": {"UnblendedCost": {"Amount": amount}}}


def test_fetch_costs_follows_pages_and_normalizes():
    client = FakeCostExplorer([
        {"ResultsByTime": [{"TimePeriod": {"Start": "2023-01-01"},
                            "Groups": [group("Amazon EC2", "BoxUsage", "10.5")]}],
         "NextPageToken": "next"},
        {"ResultsByTime": [{"TimePeriod": {"Start": "2023-02-01"},
                            "Groups": [group("Amazon EC2", "BoxUsage", "12")]}]},
    ])
    df = fetch_costs(date(2023, 1, 1), date(2023, 3, 1), client=client)

    assert client.calls[1]["NextPageToken"] == "next"
    assert client.calls[0]["GroupBy"][1] == {"Type": "DIMENSION", "Key": "USAGE_TYPE"}
    assert df["service_item"].unique().tolist() == ["Amazon EC2: BoxUsage"]
    assert df["cost"].tolist() == [10.5, 12.0]


def test_rejects_unknown_dimension():
    with pytest.raises(ValueError):
        fetch_costs(date(2023, 1, 1), date(2024, 1, 1), item_dimension="TAG", client=object())


def test_marketplace_left_out_unless_asked():
    page = {"ResultsByTime": []}
    client = FakeCostExplorer([page, page])
    fetch_costs(date(2023, 1, 1), date(2024, 1, 1), client=client)
    assert client.calls[0]["Filter"] == {
        "Not": {"Dimensions": {"Key": "BILLING_ENTITY", "Values": ["AWS Marketplace"]}}
    }
    fetch_costs(date(2023, 1, 1), date(2024, 1, 1), include_marketplace=True, client=client)
    assert "Filter" not in client.calls[1]
