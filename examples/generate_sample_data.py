"""Generate synthetic sample data (2023-2025) for trying the engine.

Run:  uv run python examples/generate_sample_data.py

Writes examples/sample_costs.csv (long format) and examples/sample_requests.csv
(wide format, one column per month, like a spreadsheet). The numbers are made
up, but follow patterns seen in real cloud bills:

- requests are seasonal and grow every year
- compute is request-driven, but its price per request drops each year
  (savings plans, rightsizing), so pooling all years weakens the correlation
- storage grows steadily regardless of traffic; a legacy database shrinks
- support and tax are (almost) flat
- marketplace software is billed in irregular lumps
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
months = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
year_index = months.year - 2023
season = np.array([1.0, 1.1, 1.15, 1.05, 0.95, 0.6, 0.5, 0.9, 1.5, 1.55, 1.25, 0.85])

requests = 150e9 * season[months.month - 1] * 1.3**year_index * rng.normal(1, 0.04, len(months))

unit_price = 4.5e-6 * 0.7**year_index  # $ per request, cheaper every year
items = {
    "EC2: Compute Instance": 250_000 + unit_price * requests * rng.normal(1, 0.06, len(months)),
    "EC2: Data Transfer": 2.1e-7 * requests * rng.normal(1, 0.03, len(months)),
    "S3: Storage": 40_000 + 1_800 * np.arange(len(months)) + rng.normal(0, 1_500, len(months)),
    "CloudWatch: Logs": 15_000 + 3e-8 * requests + rng.normal(0, 2_500, len(months)),
    "Lambda: Requests": 8_000 + 6e-8 * requests + rng.normal(0, 3_000, len(months)),
    "RDS: Legacy Database": np.maximum(90_000 - 2_400 * np.arange(len(months))
                                       + rng.normal(0, 2_000, len(months)), 0),
    "Marketplace: Software": rng.choice([5_000, 5_000, 5_000, 60_000], len(months))
                             + rng.normal(0, 500, len(months)),
    "Support: Business": np.full(len(months), 30_000.0),
    "Tax: Sales Tax": 52_000 * rng.normal(1, 0.01, len(months)),
}

out = Path(__file__).parent
costs = pd.DataFrame([
    {"month": m.date(), "service_item": item, "cost": round(float(c), 2)}
    for item, values in items.items()
    for m, c in zip(months, values)
])
costs.to_csv(out / "sample_costs.csv", index=False)

wide = pd.DataFrame([np.round(requests).astype(np.int64)], columns=[m.date() for m in months])
wide.insert(0, "metric", "requests")
wide.to_csv(out / "sample_requests.csv", index=False)
print(f"Wrote {len(costs)} cost rows and {len(months)} months of requests to {out}")
