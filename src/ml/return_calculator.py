"""
return_calculator.py
====================
일별 가격 데이터로부터 월별/분기별 수익률을 계산한다.

- 베타 추정용: 월별 수익률 (월말 종가 기준)
- 라벨링용: 분기별 수익률 (리밸런싱 날짜 기준)
- 시장(SPY) 수익률 및 무위험 수익률도 함께 계산

사용법:
    rc = ReturnCalculator(db_path="data/database/fmp_stocks.db")
    monthly, quarterly, rf = rc.compute_all(symbols, rebalance_dates, ...)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.fmp_db_manager import FmpDatabaseManager

logger = logging.getLogger(__name__)


class ReturnCalculator:
    """일별 가격 → 월별/분기별 수익률 계산기."""

    def __init__(self, db_path: str = "data/database/fmp_stocks.db"):
        self.db_path = db_path

    def compute_all(
        self,
        symbols: List[str],
        rebalance_dates: List[str],
        lookback_months: int = 36,
        market_proxy: str = "SPY",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        월별 수익률, 분기별 수익률, 무위험 수익률을 한 번의 DB 로드로 모두 계산한다.

        Returns
        -------
        (monthly_returns, quarterly_returns, monthly_risk_free)
        """
        rb_dates = sorted(pd.to_datetime(rebalance_dates))

        # 통합 가격 로드 범위
        earliest_rb = rb_dates[0]
        lookback_start = earliest_rb - pd.DateOffset(months=lookback_months + 3)
        price_start = lookback_start.strftime("%Y-%m-%d")
        price_end = (rb_dates[-1] + pd.Timedelta(days=120)).strftime("%Y-%m-%d")

        all_symbols = list(set(symbols + [market_proxy]))

        logger.info(f"가격 데이터 로드 중: {price_start} ~ {price_end}, {len(all_symbols)}종목...")
        with FmpDatabaseManager(self.db_path) as db:
            prices_dict = db.load_prices_bulk(all_symbols, price_start, price_end)
            treasury_df = db.load_treasury_rates(price_start, price_end)

        if not prices_dict:
            logger.warning("가격 데이터를 로드할 수 없습니다.")
            empty_m = pd.DataFrame(columns=["symbol", "year_month", "monthly_return"])
            empty_q = pd.DataFrame(columns=["rebalance_date", "symbol", "forward_return"])
            empty_rf = pd.DataFrame(columns=["year_month", "risk_free_monthly"])
            return empty_m, empty_q, empty_rf

        # 월별 수익률
        monthly = self._compute_monthly_returns(prices_dict)

        # 분기별 수익률
        quarterly = self._compute_quarterly_returns(prices_dict, rb_dates)

        # 무위험 수익률
        risk_free = self._compute_risk_free(treasury_df)

        return monthly, quarterly, risk_free

    # ── 개별 메서드 (compute_all 내부에서 호출, 외부에서도 사용 가능) ──

    def compute_monthly_returns(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        market_proxy: str = "SPY",
    ) -> pd.DataFrame:
        """월별 수익률을 계산한다."""
        all_symbols = list(set(symbols + [market_proxy]))
        with FmpDatabaseManager(self.db_path) as db:
            prices_dict = db.load_prices_bulk(all_symbols, start_date, end_date)
        if not prices_dict:
            return pd.DataFrame(columns=["symbol", "year_month", "monthly_return"])
        return self._compute_monthly_returns(prices_dict)

    def compute_quarterly_returns(
        self,
        symbols: List[str],
        rebalance_dates: List[str],
        market_proxy: str = "SPY",
    ) -> pd.DataFrame:
        """분기별 수익률을 계산한다."""
        rb_dates = sorted(pd.to_datetime(rebalance_dates))
        price_start = (rb_dates[0] - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        price_end = (rb_dates[-1] + pd.Timedelta(days=120)).strftime("%Y-%m-%d")
        all_symbols = list(set(symbols + [market_proxy]))
        with FmpDatabaseManager(self.db_path) as db:
            prices_dict = db.load_prices_bulk(all_symbols, price_start, price_end)
        if not prices_dict:
            return pd.DataFrame(columns=["rebalance_date", "symbol", "forward_return"])
        return self._compute_quarterly_returns(prices_dict, rb_dates)

    def compute_monthly_risk_free(
        self, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        """월별 무위험 수익률을 계산한다."""
        with FmpDatabaseManager(self.db_path) as db:
            treasury_df = db.load_treasury_rates(start_date, end_date)
        return self._compute_risk_free(treasury_df)

    # ── 내부 구현 ──────────────────────────────────────────────────────────

    def _compute_monthly_returns(self, prices_dict: dict) -> pd.DataFrame:
        """prices_dict에서 월별 수익률을 계산한다."""
        results = []
        for sym, pdf in prices_dict.items():
            mr = self._monthly_return_from_daily(pdf)
            if mr is not None and not mr.empty:
                mr["symbol"] = sym
                results.append(mr)

        if not results:
            return pd.DataFrame(columns=["symbol", "year_month", "monthly_return"])

        df = pd.concat(results, ignore_index=True)
        logger.info(
            f"월별 수익률 계산 완료: {df['symbol'].nunique()}종목, "
            f"{df['year_month'].nunique()}개월"
        )
        return df[["symbol", "year_month", "monthly_return"]]

    def _compute_quarterly_returns(
        self, prices_dict: dict, rb_dates: List[pd.Timestamp],
    ) -> pd.DataFrame:
        """prices_dict에서 분기별 수익률을 계산한다."""
        results = []
        for sym, pdf in prices_dict.items():
            qr = self._quarterly_return_for_symbol(pdf, rb_dates)
            if qr is not None and not qr.empty:
                qr["symbol"] = sym
                results.append(qr)

        if not results:
            return pd.DataFrame(columns=["rebalance_date", "symbol", "forward_return"])

        df = pd.concat(results, ignore_index=True)
        logger.info(
            f"분기별 수익률 계산 완료: {df['symbol'].nunique()}종목, "
            f"{df['rebalance_date'].nunique()}개 리밸런싱 날짜"
        )
        return df[["rebalance_date", "symbol", "forward_return"]]

    def _compute_risk_free(self, treasury_df: pd.DataFrame) -> pd.DataFrame:
        """Treasury 데이터에서 월별 무위험 수익률을 계산한다."""
        if treasury_df is None or treasury_df.empty:
            logger.warning("Treasury 데이터를 로드할 수 없습니다.")
            return pd.DataFrame(columns=["year_month", "risk_free_monthly"])

        treasury_df = treasury_df.copy()
        treasury_df["date"] = pd.to_datetime(treasury_df["date"])
        treasury_df["year_month"] = treasury_df["date"].dt.to_period("M").astype(str)

        monthly = treasury_df.groupby("year_month").last().reset_index()
        # month_3 (3개월물 연이율 %) → 월 수익률: (1 + r/100)^(1/12) - 1
        monthly["risk_free_monthly"] = (1 + monthly["month_3"] / 100) ** (1 / 12) - 1

        return monthly[["year_month", "risk_free_monthly"]]

    def _monthly_return_from_daily(self, price_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """일별 가격 → 월말 종가 → 월별 수익률."""
        if price_df is None or price_df.empty:
            return None

        df = price_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"])

        df["year_month"] = df["date"].dt.to_period("M").astype(str)
        month_end = df.groupby("year_month").last().reset_index()

        month_end["monthly_return"] = month_end["close"].pct_change()
        month_end = month_end.dropna(subset=["monthly_return"])

        return month_end[["year_month", "monthly_return"]]

    def _quarterly_return_for_symbol(
        self,
        price_df: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp],
    ) -> Optional[pd.DataFrame]:
        """한 종목의 각 리밸런싱 날짜별 forward return을 계산한다."""
        if price_df is None or price_df.empty:
            return None

        df = price_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")

        rows = []
        for i, rb_date in enumerate(rebalance_dates[:-1]):
            next_rb_date = rebalance_dates[i + 1]

            p_now = self._get_nearest_price(df, rb_date, max_offset_days=5)
            p_next = self._get_nearest_price(df, next_rb_date, max_offset_days=5)

            if p_now is not None and p_next is not None and p_now > 0:
                rows.append({
                    "rebalance_date": rb_date.strftime("%Y-%m-%d"),
                    "forward_return": p_next / p_now - 1,
                })
            else:
                rows.append({
                    "rebalance_date": rb_date.strftime("%Y-%m-%d"),
                    "forward_return": np.nan,
                })

        if not rows:
            return None
        return pd.DataFrame(rows)

    @staticmethod
    def _get_nearest_price(
        price_indexed: pd.DataFrame,
        target_date: pd.Timestamp,
        max_offset_days: int = 5,
    ) -> Optional[float]:
        """target_date 이하에서 가장 가까운 거래일의 close 가격을 반환한다."""
        start = target_date - pd.Timedelta(days=max_offset_days)
        subset = price_indexed.loc[start:target_date]
        if subset.empty:
            return None
        return float(subset["close"].iloc[-1])
