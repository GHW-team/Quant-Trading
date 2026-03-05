"""
barra_labeler.py
================
시장 베타를 활용한 알파 기반 라벨링.

각 리밸런싱 날짜 T, 각 종목 i에 대해:
    alpha_proxy_i = R_i(T→T+b) - beta_i × R_m(T→T+b)
    C = quantile(alpha_proxy, 1 - target_ratio)
    label_i = 1 if alpha_proxy_i > C else 0

사용법:
    from src.ml.barra_labeler import BarraLabeler

    labeler = BarraLabeler(target_ratio=0.3)
    labeled = labeler.label(forward_returns, betas)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BarraLabeler:
    """Barra 모델 기반 알파 라벨링."""

    def __init__(
        self,
        target_ratio: float = 0.3,
    ):
        """
        Parameters
        ----------
        target_ratio : 라벨 1의 비율 (c 파라미터). 상위 c%를 라벨 1로 지정.
        """
        if not 0 < target_ratio < 1:
            raise ValueError(f"target_ratio는 0~1 사이여야 합니다: {target_ratio}")
        self.target_ratio = target_ratio

    def label(
        self,
        forward_returns: pd.DataFrame,
        betas: pd.DataFrame,
        market_proxy: str = "SPY",
    ) -> pd.DataFrame:
        """
        알파 프록시를 계산하고 라벨링한다.

        Parameters
        ----------
        forward_returns : [rebalance_date, symbol, forward_return]
            종목별 + 시장(SPY)의 분기 수익률
        betas : [rebalance_date, symbol, beta_market, ...]
            BarraRegression.estimate_all_dates() 출력

        Returns
        -------
        DataFrame [rebalance_date, symbol, alpha_proxy, label, threshold_C]
        """
        # 시장 forward return 추출
        market_fwd = forward_returns[forward_returns["symbol"] == market_proxy][
            ["rebalance_date", "forward_return"]
        ].rename(columns={"forward_return": "market_forward_return"})

        # 종목 forward return
        stock_fwd = forward_returns[forward_returns["symbol"] != market_proxy].copy()

        # 시장 수익률 병합
        merged = stock_fwd.merge(market_fwd, on="rebalance_date", how="inner")

        # 베타 병합
        merged = merged.merge(
            betas[["rebalance_date", "symbol", "beta_market"]],
            on=["rebalance_date", "symbol"],
            how="inner",
        )

        # 알파 프록시 = 종목 수익률 - beta × 시장 수익률
        merged["alpha_proxy"] = (
            merged["forward_return"]
            - merged["beta_market"] * merged["market_forward_return"]
        )

        # 리밸런싱 날짜별 라벨링
        results = []
        for rb_date, group in merged.groupby("rebalance_date"):
            valid = group.dropna(subset=["alpha_proxy"])
            if valid.empty:
                continue

            # C = 상위 target_ratio의 경계값
            threshold_C = valid["alpha_proxy"].quantile(1 - self.target_ratio)
            labeled = valid.copy()
            labeled["label"] = (labeled["alpha_proxy"] > threshold_C).astype(int)
            labeled["threshold_C"] = threshold_C

            actual_ratio = labeled["label"].mean()
            logger.info(
                f"  {rb_date}: C={threshold_C:.4f}, "
                f"라벨1={actual_ratio:.1%} ({labeled['label'].sum()}/{len(labeled)})"
            )
            results.append(labeled)

        if not results:
            logger.warning("라벨링 결과가 비어있습니다.")
            return pd.DataFrame(
                columns=["rebalance_date", "symbol", "alpha_proxy", "label", "threshold_C"]
            )

        df = pd.concat(results, ignore_index=True)
        total_ratio = df["label"].mean()
        logger.info(
            f"라벨링 완료: {len(df)}행, "
            f"라벨1 비율={total_ratio:.1%}, "
            f"리밸런싱 날짜 {df['rebalance_date'].nunique()}개"
        )

        return df[["rebalance_date", "symbol", "alpha_proxy", "label", "threshold_C"]]
