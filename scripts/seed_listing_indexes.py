"""
Seed the ticker universe grouped by listing indexes using local CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from src.data.db_manager import DatabaseManager


DEFAULT_INDEX_FILES: Dict[str, Path] = {
    "SP500": Path("sp500.csv"),
    "NASDAQ100": Path("nasdaq100.csv"),
}


def _extract_tickers(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Ticker source not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for candidate in ("Symbol", "Ticker", "티커"):
        if candidate in df.columns:
            tickers = (
                df[candidate]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .tolist()
            )
            return tickers

    raise ValueError(
        f"CSV '{csv_path}' does not contain a recognizable ticker column "
        "(expected one of Symbol/Ticker/티커)."
    )


def seed_listing_indexes(
    db_path: str,
    index_files: Dict[str, Path] = DEFAULT_INDEX_FILES,
) -> None:
    db_manager = DatabaseManager(db_path=db_path)
    try:
        for index_name, csv_path in index_files.items():
            tickers = _extract_tickers(csv_path)
            if not tickers:
                print(f"[WARN] No tickers found in {csv_path}")
                continue

            created = 0
            for ticker in tickers:
                ticker_id = db_manager.get_or_create_ticker(
                    ticker_code=ticker,
                    name=None,
                    market=index_name,
                )
                if ticker_id:
                    created += 1
            print(f"[INFO] Indexed {len(tickers)} tickers for {index_name} (created {created})")
    finally:
        db_manager.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed tickers grouped by listing index.")
    parser.add_argument(
        "--db-path",
        default="data/database/stocks.db",
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--index-file",
        action="append",
        metavar="INDEX=PATH",
        help="Override or add an index CSV mapping (e.g. SP500=data/tickers/sp500.csv).",
    )
    return parser.parse_args()


def _parse_index_overrides(entries: Iterable[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid index mapping '{entry}'. Expected format INDEX=PATH.")
        key, value = entry.split("=", 1)
        mapping[key.strip()] = Path(value.strip())
    return mapping


def main() -> None:
    args = parse_args()
    index_files = DEFAULT_INDEX_FILES.copy()
    if args.index_file:
        index_files.update(_parse_index_overrides(args.index_file))

    seed_listing_indexes(db_path=args.db_path, index_files=index_files)


if __name__ == "__main__":
    main()
