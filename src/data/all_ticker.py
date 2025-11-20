from __future__ import annotations
#타입 힌트를 나중에 평가하도록 하는 옵션
"""Exchange-wide ticker loader with yfinance-ready symbols."""

import FinanceDataReader as fdr
from pykrx import stock as pykrx_stock  # 변경 사항: KRX는 pykrx 사용
import logging
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)


class TickerUniverse:
    """Build ticker lists grouped by exchange and ready for yfinance/data_fetcher.

    Notes:
        - KRX 심볼은 yfinance 접미사(.KS/.KQ)를 자동으로 붙임.
        - 미국 거래소(NASDAQ/NYSE/AMEX)는 심볼을 그대로 반환.
    """

    SUPPORTED_EXCHANGES = ("KRX", "KOSPI", "KOSDAQ", "S&P500", "NYSE", "AMEX")  # 변경 사항: KRX/KOSPI/KOSDAQ + S&P500/NYSE/AMEX

    def __init__(self, add_kr_suffix: bool = True) -> None:
        self.add_kr_suffix = add_kr_suffix

    def get(self, exchanges: Optional[Iterable[str]] = None) -> List[str]:
        """Return flat ticker list for the requested exchanges.

        Args:
            exchanges: iterable of exchange codes. None -> all supported.

        Returns:
            List[str]: deduplicated ticker list (order preserved) ready for StockDataFetcher.
        """
        exchanges = list(exchanges) if exchanges else list(self.SUPPORTED_EXCHANGES)
        tickers: List[str] = []

        for ex in exchanges:
            ex_upper = ex.upper()
            try:
                if ex_upper == "KRX":
                    tickers.extend(self._load_krx())
                elif ex_upper == "KOSPI":  # 변경 사항: KOSPI만 선택
                    tickers.extend(self._load_krx_market("KOSPI"))
                elif ex_upper == "KOSDAQ":  # 변경 사항: KOSDAQ만 선택
                    tickers.extend(self._load_krx_market("KOSDAQ"))
                elif ex_upper == "S&P500":  # 변경 사항: S&P500 구성 종목
                    tickers.extend(self._load_sp500())
                elif ex_upper in ("NYSE", "AMEX"):
                    tickers.extend(self._load_us_exchange(ex_upper))
                elif ex_upper == "NASDAQ":
                    logger.warning("NASDAQ 지원이 제거되었습니다. S&P500을 사용하세요.  # 변경 사항")
                else:
                    logger.warning("Unsupported exchange requested: %s", ex)
            except Exception as exc:  # network/errors from FDR
                logger.error("Failed to load tickers for %s: %s", ex_upper, exc)

        return self._dedup_preserve_order(tickers)

    def _load_krx(self) -> List[str]:
        # 변경 사항: pykrx로 KOSPI+KOSDAQ 모두 로드
        try:
            codes = []
            for market in ("KOSPI", "KOSDAQ"):
                codes.extend(self._load_krx_pykrx(market))
            return codes
        except Exception as exc:
            logger.error("pykrx KRX 전체 로드 실패: %s", exc)
            df = fdr.StockListing("KRX")
            return self._convert_krx_df(df)

    def _load_krx_market(self, market_prefix: str) -> List[str]:
        """특정 시장만 필터링 (변경 사항)."""
        try:
            return self._load_krx_pykrx(market_prefix)
        except Exception as exc:
            logger.error("pykrx %s 로드 실패: %s", market_prefix, exc)
            df = fdr.StockListing("KRX")
            if "Market" in df.columns:
                df = df[df["Market"].str.upper().str.startswith(market_prefix)]
            return self._convert_krx_df(df)

    def _load_krx_pykrx(self, market: str) -> List[str]:
        """pykrx로 시장별 코드 로드 (변경 사항)."""
        tickers = pykrx_stock.get_market_ticker_list(market=market)
        suffix = ".KS" if market.upper() == "KOSPI" else ".KQ"
        return [code + suffix for code in tickers]

    def _convert_krx_df(self, df) -> List[str]:
        if df.empty or "Symbol" not in df.columns:
            logger.warning("KRX listing is empty or missing Symbol column")
            return []

        symbols = []
        for _, row in df.iterrows():
            raw = str(row["Symbol"]).strip()
            if not raw or raw.lower() == "nan":
                continue
            market = str(row.get("Market", "")).upper()
            suffix = ""
            if self.add_kr_suffix:
                if market.startswith("KOSPI"):
                    suffix = ".KS"
                elif market.startswith("KOSDAQ"):
                    suffix = ".KQ"
            symbols.append(raw + suffix)
        return symbols

    def _load_sp500(self) -> List[str]:
        """S&P500 구성 종목 로드 (변경 사항)."""
        df = fdr.StockListing("S&P500")
        if df.empty or "Symbol" not in df.columns:
            logger.warning("S&P500 listing is empty or missing Symbol column")
            return []
        return [str(sym).strip() for sym in df["Symbol"].dropna() if str(sym).strip()]

    @staticmethod
    def _load_us_exchange(exchange: str) -> List[str]:
        df = fdr.StockListing(exchange)
        if df.empty or "Symbol" not in df.columns:
            logger.warning("%s listing is empty or missing Symbol column", exchange)
            return []
        return [str(sym).strip() for sym in df["Symbol"].dropna() if str(sym).strip()]

    @staticmethod
    def _dedup_preserve_order(tickers: Iterable[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for t in tickers:
            if t and t not in seen:
                ordered.append(t)
                seen.add(t)
        return ordered


def load_all_tickers() -> dict[str, list[str]]:
    """하위 호환용: 거래소별 티커 dict 반환."""
    return {
        "KRX": [code + ".KS" for code in pykrx_stock.get_market_ticker_list(market="KOSPI")]  # 변경 사항: pykrx
               + [code + ".KQ" for code in pykrx_stock.get_market_ticker_list(market="KOSDAQ")],
        "KOSPI": [code + ".KS" for code in pykrx_stock.get_market_ticker_list(market="KOSPI")],  # 변경 사항
        "KOSDAQ": [code + ".KQ" for code in pykrx_stock.get_market_ticker_list(market="KOSDAQ")],  # 변경 사항
        "S&P500": fdr.StockListing("S&P500")["Symbol"].tolist(),  # 변경 사항
        "NYSE": fdr.StockListing("NYSE")["Symbol"].tolist(),
        "AMEX": fdr.StockListing("AMEX")["Symbol"].tolist(),
        # 필요시 추가: "KRX-DELISTING", "KRX-DEL", "SP500", "SPX", "HSI", "SSE", "SZSE" 등
    }
