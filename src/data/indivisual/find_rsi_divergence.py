"""로컬 DB 데이터만 사용해 NYSE 티커 중 RSI 상승 다이버전스가 발생한 종목을 찾습니다."""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# 스크립트를 직접 실행할 때도 리포지토리 모듈을 불러오기 위해 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.indicator_calculator import IndicatorCalculator


def load_nyse_tickers(csv_path: Path) -> List[str]:
    """로컬 CSV에서 NYSE 심볼을 로드합니다."""
    df = pd.read_csv(csv_path)
    symbol_col = next((c for c in df.columns if "symbol" in c.lower()), None)
    if symbol_col is None:
        raise ValueError(f"Could not find a symbol column in {csv_path}")

    symbols = (
        df[symbol_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    # 중복을 제거하면서 원래 순서를 보존
    seen = set()
    ordered = []
    for sym in symbols:
        if sym and sym not in seen:
            ordered.append(sym)
            seen.add(sym)
    return ordered


def available_tickers(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT ticker_code FROM tickers").fetchall()
    return {row[0] for row in rows}


def latest_price_date(conn: sqlite3.Connection) -> pd.Timestamp:
    row = conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()
    if not row or row[0] is None:
        raise RuntimeError("daily_prices table is empty")
    return pd.to_datetime(row[0])


def load_price_and_rsi(
    conn: sqlite3.Connection,
    tickers: Iterable[str],
    start_date: str,
) -> pd.DataFrame:
    """SQLite 바인딩 제한을 피하기 위해 청크 단위로 가격/RSI를 불러옵니다."""
    tickers = list(tickers)
    if not tickers:
        return pd.DataFrame()

    chunk_size = 900  # SQLite 바인딩 한도(999)를 고려한 안전 청크 크기
    frames: list[pd.DataFrame] = []
    base_sql = """
        SELECT t.ticker_code,
               p.date,
               p.adj_close,
               p.close,
               ti.rsi
        FROM daily_prices AS p
        JOIN tickers AS t ON t.ticker_id = p.ticker_id
        LEFT JOIN technical_indicators AS ti
               ON ti.ticker_id = t.ticker_id AND ti.date = p.date
        WHERE t.ticker_code IN ({placeholders})
          AND p.date >= ?
        ORDER BY t.ticker_code, p.date
    """

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        sql = base_sql.format(placeholders=placeholders)
        params = chunk + [start_date]
        frames.append(pd.read_sql_query(sql, conn, params=params, parse_dates=["date"]))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for col in ("adj_close", "close", "rsi"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _pivot_lows(series: pd.Series, min_gap_days: int) -> pd.Series | None:
    """
    피벗 저점 리스트를 반환하되, min_gap_days 이내에 몰려 있는 피벗은
    '더 낮거나(같으면 더 최근인) 저점'으로 대표시켜 보수적으로 유지한다.
    """
    raw = series[(series.shift(1) > series) & (series.shift(-1) > series)]
    if len(raw) < 2:
        return None

    if not min_gap_days or min_gap_days <= 0:
        return raw

    # 클러스터 압축: 간격이 min_gap_days 이하이면 더 낮은(동일 시 더 최근) 피벗으로 대체
    clustered_dates = []
    clustered_values = []

    current_date = raw.index[0]
    current_val = raw.iloc[0]

    for dt, val in raw.iloc[1:].items():
        gap = (dt - current_date).days
        if gap <= min_gap_days:
            # 같은 클러스터: 더 낮거나(같으면 더 최근) 한 쪽만 유지
            if (val < current_val) or (val == current_val and dt > current_date):
                current_date, current_val = dt, val
        else:
            clustered_dates.append(current_date)
            clustered_values.append(current_val)
            current_date, current_val = dt, val

    clustered_dates.append(current_date)
    clustered_values.append(current_val)

    lows = pd.Series(clustered_values, index=pd.Index(clustered_dates, name=raw.index.name))
    if len(lows) < 2:
        return None
    return lows


def _calc_rsi_safe(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """RSI 계산: 라이브러리가 None을 주면 수식으로 직접 계산."""
    rsi = IndicatorCalculator._calc_rsi(df, length)
    if rsi is None:
        closes = df["adj_close"]
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=df.index)


def has_bullish_rsi_divergence(
    df: pd.DataFrame,
    lookback_days: int,
    min_gap_days: int,
    recent_days: int,
) -> bool:
    """가격은 더 낮거나 같은 저점, RSI는 더 높은 저점을 만드는 상승 다이버전스 여부를 판정."""
    if df.empty or "adj_close" not in df.columns:
        return False

    df = df.sort_index()
    if "rsi" not in df.columns:
        df["rsi"] = _calc_rsi_safe(df, 14)
    elif df["rsi"].isna().any():
        df["rsi"] = df["rsi"].fillna(_calc_rsi_safe(df, 14))

    window_start = df.index.max() - pd.Timedelta(days=lookback_days)
    window = df[df.index >= window_start].dropna(subset=["adj_close", "rsi"])
    if len(window) < 20:
        return False

    pivot_lows = _pivot_lows(window["adj_close"], min_gap_days)
    if pivot_lows is None:
        return False

    last_two = pivot_lows.tail(2)
    d1, d2 = last_two.index[0], last_two.index[1]
    # 최근성 필터: 마지막 피벗 저점이 최근 recent_days 이내여야 함 (0이면 비활성화)
    if recent_days and (window.index.max() - d2).days > recent_days:
        return False

    r1 = window.loc[d1, "rsi"] if d1 in window.index else None
    r2 = window.loc[d2, "rsi"] if d2 in window.index else None
    if pd.isna(r1) or pd.isna(r2):
        return False

    return last_two.iloc[1] <= last_two.iloc[0] and r2 > r1


def find_divergent_tickers(
    df: pd.DataFrame,
    lookback_days: int,
    min_gap_days: int,
    recent_days: int,
) -> list[str]:
    winners: list[str] = []
    for ticker, group in df.groupby("ticker_code"):
        ticker_df = group.sort_values("date").set_index("date")
        if has_bullish_rsi_divergence(ticker_df, lookback_days, min_gap_days, recent_days):
            winners.append(ticker)
    return winners


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="로컬 SQLite 데이터를 사용해 NYSE 종목의 RSI 상승 다이버전스를 찾습니다."
    )
    parser.add_argument(
        "--db-path",
        default="data/database/stocks.db",
        help="파이프라인이 채운 SQLite DB 경로.",
    )
    parser.add_argument(
        "--nyse-file",
        default="data/tickers/nyse.csv",
        help="NYSE 심볼이 들어 있는 CSV (Symbol 컬럼).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=120,
        help="피벗 저점을 찾을 룩백 기간(일).",
    )
    parser.add_argument(
        "--min-gap-days",
        type=int,
        default=5,
        help="노이즈를 줄이기 위한 피벗 저점 간 최소 일수.",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=20,
        help="가장 최근 봉 대비 마지막 피벗 저점이 이 일수 이내여야 함(0이면 비활성화).",
    )
    parser.add_argument(
        "--start-date",
        help="조회 시작일 수동 지정(YYYY-MM-DD).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nyse_path = PROJECT_ROOT / args.nyse_file

    tickers = load_nyse_tickers(nyse_path)
    if not tickers:
        raise SystemExit("No NYSE tickers loaded from CSV.")

    with sqlite3.connect(args.db_path) as conn:
        have_data = available_tickers(conn)
        filtered = [t for t in tickers if t in have_data]
        if not filtered:
            raise SystemExit("No NYSE tickers found in the database.")

        latest_date = latest_price_date(conn)
        start_date = args.start_date or (latest_date - timedelta(days=args.lookback_days + 30)).strftime("%Y-%m-%d")

        price_df = load_price_and_rsi(conn, filtered, start_date)

    if price_df.empty:
        raise SystemExit("No price data loaded for the requested window.")

    results = find_divergent_tickers(
        price_df,
        lookback_days=args.lookback_days,
        min_gap_days=args.min_gap_days,
        recent_days=args.recent_days,
    )

    print(f"Found {len(results)} NYSE tickers with RSI bullish divergence:")
    for ticker in results:
        print(ticker)


if __name__ == "__main__":
    main()
