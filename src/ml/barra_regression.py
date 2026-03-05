"""
barra_regression.py
===================
종목별 시계열 회귀로 시장 베타(market beta)를 추정한다.

각 리밸런싱 날짜 T, 각 종목 i에 대해:
    과거 lookback_months개월의 월별 초과수익률로:
    R_i,t - R_f,t = alpha + beta_market × (R_m,t - R_f,t) + epsilon
    → beta_market_i 추출

사용법:
    from src.ml.barra_regression import BarraRegression
    from src.ml.return_calculator import ReturnCalculator

    rc = ReturnCalculator(db_path="data/database/fmp_stocks.db")
    monthly = rc.compute_monthly_returns(symbols, start, end)
    rf = rc.compute_monthly_risk_free(start, end)

    reg = BarraRegression(lookback_months=36)
    betas = reg.estimate_all_dates(monthly, rf, rebalance_dates, market_proxy="SPY")
"""

import logging
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


def _ols_beta_numpy(y: np.ndarray, x: np.ndarray):
    """numpy 기반 단순 OLS (상수항 포함). statsmodels보다 ~50x 빠름."""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    try:
        # (X'X)^-1 X'y
        params = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ params
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return params[1], params[0], r_squared  # beta, alpha, r2
    except Exception:
        return np.nan, np.nan, np.nan


class BarraRegression:
    """종목별 시계열 회귀를 통한 시장 베타 추정."""

    def __init__(
        self,
        lookback_months: int = 36,
        min_observations: int = 12,
        vif_threshold: float = 10.0,
        ridge_alpha: float = 1.0,
    ):
        """
        Parameters
        ----------
        lookback_months : 베타 추정에 사용할 과거 월 수 (a 파라미터)
        min_observations : 최소 관측치 수. 이보다 적으면 beta=1.0 기본값 사용
        vif_threshold : VIF 임계값. 다중 팩터 확장 시 사용
        ridge_alpha : Ridge 회귀 시 alpha 값
        """
        self.lookback_months = lookback_months
        self.min_observations = min_observations
        self.vif_threshold = vif_threshold
        self.ridge_alpha = ridge_alpha

    def estimate_all_dates(
        self,
        monthly_returns: pd.DataFrame,
        monthly_risk_free: pd.DataFrame,
        rebalance_dates: List[str],
        market_proxy: str = "SPY",
    ) -> pd.DataFrame:
        """
        모든 리밸런싱 날짜에 대해 종목별 시장 베타를 추정한다.

        Parameters
        ----------
        monthly_returns : [symbol, year_month, monthly_return] from ReturnCalculator
        monthly_risk_free : [year_month, risk_free_monthly] from ReturnCalculator
        rebalance_dates : 리밸런싱 날짜 리스트
        market_proxy : 시장 프록시 심볼 (기본: SPY)

        Returns
        -------
        DataFrame [rebalance_date, symbol, beta_market, alpha, r_squared, n_obs]
        """
        # 초과수익률 계산
        mr = monthly_returns.merge(monthly_risk_free, on="year_month", how="left")
        mr["risk_free_monthly"] = mr["risk_free_monthly"].fillna(0)
        mr["excess_return"] = mr["monthly_return"] - mr["risk_free_monthly"]

        # 시장 초과수익률 추출
        market_excess = mr.loc[
            mr["symbol"] == market_proxy,
            ["year_month", "excess_return"],
        ].rename(columns={"excess_return": "market_excess"})

        # 종목별 초과수익률에 시장 초과수익률 병합
        stock_returns = mr[mr["symbol"] != market_proxy].copy()
        stock_returns = stock_returns.merge(market_excess, on="year_month", how="inner")
        stock_returns = stock_returns.dropna(subset=["excess_return", "market_excess"])

        # ── 핵심 최적화: 종목별 그룹 + year_month 인덱스로 빠른 슬라이싱 ──
        # year_month를 정렬된 정수 인덱스로 변환
        all_months = sorted(stock_returns["year_month"].unique())
        month_to_idx = {m: i for i, m in enumerate(all_months)}
        stock_returns["_month_idx"] = stock_returns["year_month"].map(month_to_idx)

        # 종목별 그룹화 (dict of arrays)
        sym_groups = {}
        for sym, grp in stock_returns.groupby("symbol"):
            grp_sorted = grp.sort_values("_month_idx")
            sym_groups[sym] = {
                "month_idx": grp_sorted["_month_idx"].values,
                "excess": grp_sorted["excess_return"].values,
                "market": grp_sorted["market_excess"].values,
            }

        symbols = list(sym_groups.keys())

        # 리밸런싱 날짜별 lookback 범위 (month_idx 기준)
        rb_ranges = []
        for rb_date_str in rebalance_dates:
            rb_date = pd.Timestamp(rb_date_str)
            cutoff = rb_date - pd.DateOffset(months=self.lookback_months)
            cutoff_ym = cutoff.to_period("M").start_time.strftime("%Y-%m")
            rb_ym = rb_date.to_period("M").start_time.strftime("%Y-%m")
            lo = month_to_idx.get(cutoff_ym, 0)
            hi = month_to_idx.get(rb_ym, len(all_months) - 1)
            rb_ranges.append((rb_date_str, lo, hi))

        # ── 벡터화된 베타 추정 ──
        results = []
        total = len(symbols) * len(rb_ranges)
        done = 0

        for sym in symbols:
            data = sym_groups[sym]
            midx = data["month_idx"]
            exc = data["excess"]
            mkt = data["market"]

            for rb_date_str, lo, hi in rb_ranges:
                # numpy boolean mask로 빠른 필터링
                mask = (midx >= lo) & (midx <= hi)
                n = mask.sum()

                if n < self.min_observations:
                    results.append((rb_date_str, sym, 1.0, 0.0, np.nan, int(n)))
                else:
                    beta, alpha, r2 = _ols_beta_numpy(exc[mask], mkt[mask])
                    if np.isnan(beta):
                        results.append((rb_date_str, sym, 1.0, 0.0, np.nan, int(n)))
                    else:
                        results.append((rb_date_str, sym, beta, alpha, r2, int(n)))

            done += len(rb_ranges)
            if done % 50000 == 0 or done == total:
                logger.info(f"  베타 추정 진행: {done}/{total} ({done/total:.0%})")

        df = pd.DataFrame(
            results,
            columns=["rebalance_date", "symbol", "beta_market", "alpha", "r_squared", "n_obs"],
        )

        # 이상치 베타 클리핑: 합리적 범위(-2 ~ 5)로 제한
        raw_outliers = ((df["beta_market"] < -2) | (df["beta_market"] > 5)).sum()
        df["beta_market"] = df["beta_market"].clip(-2, 5)

        n_valid = (df["beta_market"] != 1.0).sum()
        n_default = (df["beta_market"] == 1.0).sum()
        logger.info(
            f"베타 추정 완료: {len(df)}행 "
            f"(유효 {n_valid}, 기본값 {n_default}, 클리핑 {raw_outliers})"
        )
        logger.info(
            f"  베타 분포: mean={df['beta_market'].mean():.3f}, "
            f"std={df['beta_market'].std():.3f}, "
            f"median={df['beta_market'].median():.3f}"
        )
        return df

    def check_multicollinearity(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        다중 팩터 모델 확장 시 VIF를 검사한다.

        Parameters
        ----------
        X : 독립변수 DataFrame (상수항 포함하지 않음)

        Returns
        -------
        DataFrame [feature, vif, warning]
        """
        X_clean = X.dropna()
        X_with_const = sm.add_constant(X_clean)

        vif_data = []
        for i, col in enumerate(X_with_const.columns):
            if col == "const":
                continue
            vif_val = variance_inflation_factor(X_with_const.values, i)
            warning = ""
            if vif_val > self.vif_threshold:
                warning = "위험 - 삭제 또는 직교화 필요"
            elif vif_val > 5:
                warning = "주의 - 직교화 고려"
            vif_data.append({
                "feature": col,
                "vif": round(vif_val, 2),
                "warning": warning,
            })

        df = pd.DataFrame(vif_data)
        high_vif = df[df["vif"] > self.vif_threshold]
        if not high_vif.empty:
            logger.warning(
                f"높은 VIF 감지: {high_vif[['feature', 'vif']].to_dict('records')}"
            )
        return df
