"""
One-click helper to seed listing indexes and run the ticker filtering workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.seed_listing_indexes import DEFAULT_INDEX_FILES, seed_listing_indexes
from src.data.ticker_filtering import TickerFilteringWorkflow


def _parse_index_overrides(entries: Iterable[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid index mapping '{entry}'. Expected format INDEX=PATH.")
        key, value = entry.split("=", 1)
        mapping[key.strip()] = Path(value.strip())
    return mapping


def run_pipeline(
    db_path: str,
    listing_indexes: Optional[Sequence[str]],
    lookback: int,
    return_quantile: float,
    volume_quantile: float,
    index_files: Dict[str, Path],
) -> None:
    # Step 1: seed the ticker universe.
    seed_listing_indexes(db_path=db_path, index_files=index_files)

    # Step 2: run filtering.
    workflow = TickerFilteringWorkflow(
        db_path=db_path,
        listing_indexes=listing_indexes,
        lookback=lookback,
        return_quantile=return_quantile,
        volume_quantile=volume_quantile,
    )
    result = workflow.run()

    print("\n=== Final flattened tickers ===")
    print(result.flattened)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed indexes and run ticker filtering.")
    parser.add_argument(
        "--db-path",
        default="data/database/stocks.db",
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--listing-index",
        action="append",
        help="Name of the listing index to include (default: all indexes from the CSV mappings). "
             "Use multiple times for multiple indexes.",
    )
    parser.add_argument(
        "--index-file",
        action="append",
        metavar="INDEX=PATH",
        help="Override/add CSV mapping for indexes (e.g. SP500=data/tickers/sp500.csv).",
    )
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--return-quantile", type=float, default=0.8)
    parser.add_argument("--volume-quantile", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_files = DEFAULT_INDEX_FILES.copy()
    if args.index_file:
        index_files.update(_parse_index_overrides(args.index_file))

    run_pipeline(
        db_path=args.db_path,
        listing_indexes=args.listing_index,
        lookback=args.lookback,
        return_quantile=args.return_quantile,
        volume_quantile=args.volume_quantile,
        index_files=index_files,
    )


if __name__ == "__main__":
    main()
