"""Download the raw series from FRED and build the quarterly dataset.

Output: data/data.csv — one row per quarter (quarter-end dates), 1987 Q1 onward.

Usage:
    1. Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html
    2. Copy .env.example to .env and paste your key
    3. python src/build_dataset.py
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

# FRED series ID -> column name used in the analysis
SERIES = {
    "DRCLACBS": "delinquency",          # Delinquency rate on consumer loans, all commercial banks (quarterly)
    "T10Y2Y": "yield_spread",           # 10-year minus 2-year Treasury yield (daily)
    "UNRATE": "unemployment",           # Unemployment rate (monthly)
    "FEDFUNDS": "federal_fund_rate",    # Effective federal funds rate (monthly)
    "UMCSENT": "consumer_confidence",   # University of Michigan consumer sentiment (monthly)
}

START = "1987-01-01"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "data.csv"


def main() -> None:
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise SystemExit("FRED_API_KEY not found. Copy .env.example to .env and add your key.")

    fred = Fred(api_key=api_key)

    # Pull each series and convert to quarterly averages (quarter-end dates)
    quarterly = []
    for series_id, name in SERIES.items():
        s = fred.get_series(series_id, observation_start=START)
        s.index = pd.to_datetime(s.index)
        quarterly.append(s.resample("QE").mean().rename(name))

    data = pd.concat(quarterly, axis=1).dropna()
    data.index.name = "date"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_PATH)
    print(f"Saved {len(data)} quarters ({data.index.min():%Y-%m} to {data.index.max():%Y-%m}) to {OUT_PATH}")


if __name__ == "__main__":
    main()
