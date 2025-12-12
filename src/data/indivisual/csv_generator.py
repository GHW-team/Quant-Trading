import logging
from pathlib import Path
from typing import List

import FinanceDataReader as fdr
import pandas as pd

logger = logging.getLogger(__name__)


def load_sp500_local(csv_path: str = "data/tickers/sp500.csv") -> List[str]:
    """
    로컬 CSV에서 S&P500 심볼을 읽어 리스트로 반환한다.
    CSV는 Symbol 컬럼 하나만 있다고 가정.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning("S&P500 CSV가 없습니다: %s", csv_path)
        return []

    df = pd.read_csv(path)
    if "Symbol" not in df.columns:
        logger.warning("CSV에 Symbol 컬럼이 없습니다")
        return []

    return [str(sym).strip() for sym in df["Symbol"].dropna() if str(sym).strip()]


def load_sp500_with_fallback(csv_path: str = "data/tickers/sp500.csv") -> List[str]:
    """
    FDR로 S&P500 심볼을 시도하고 실패 시 로컬 CSV로 대체한다.
    """
    try:
        df = fdr.StockListing("S&P500")
        if df.empty or "Symbol" not in df.columns:
            raise ValueError("FDR 결과가 비었거나 Symbol 컬럼이 없음")
        return [str(sym).strip() for sym in df["Symbol"].dropna() if str(sym).strip()]
    except Exception as exc:
        logger.warning("FDR S&P500 로드 실패(%s) → 로컬 CSV로 대체", exc)
        return load_sp500_local(csv_path)
    
from src.data.csv_generator import load_sp500_with_fallback
tks = load_sp500_with_fallback("data/tickers/sp500.csv")
print(len(tks), tks[:5])

from src.data.csv_generator import load_sp500_local
from src.data.pipeline import DataPipeline

t = load_sp500_local('data/tickers/sp500.csv')
print(len(t), 'tickers loaded', t[:5])

with DataPipeline(db_path='data/database/stocks.db') as p:
    res = p.run_price_pipeline(
        ticker_list=t,
        period='1y',
        interval='1d',
        update_if_exists=True,
        batch_size=100,
    )
    print(res.get('summary'))
