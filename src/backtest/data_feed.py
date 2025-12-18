# src/backtest/data_feed.py
# DB에서 가격/지표 데이터를 로드하여 Backtrader 피드로 변환

import logging
from datetime import datetime
from typing import Dict, List, Optional

import backtrader as bt
import pandas as pd

from src.data.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class DatabaseFeed(bt.feeds.PandasData):
    """
    DB에서 로드한 DataFrame을 Backtrader 피드로 변환
    
    기본 OHLCV + 지표 컬럼을 라인으로 추가
    """
    # 기본 OHLCV 컬럼 매핑
    params = (
        ('datetime', 'date'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),  # 사용 안함
        
        # 추가 지표 라인들
        ('ma_5', -1),
        ('ma_10', -1),
        ('ma_20', -1),
        ('ma_50', -1),
        ('ma_60', -1),
        ('ma_100', -1),
        ('ma_120', -1),
        ('ma_200', -1),
        ('macd', -1),
        ('macd_signal', -1),
        ('macd_hist', -1),
        ('rsi', -1),
        ('bb_upper', -1),
        ('bb_mid', -1),
        ('bb_lower', -1),
        ('bb_pct', -1),
        ('atr', -1),
        ('hv', -1),
        ('stoch_k', -1),
        ('stoch_d', -1),
        ('obv', -1),
        
        # ML 예측 신호
        ('signal', -1),
    )
    
    # 추가 라인 정의 (기본 OHLCV 외)
    lines = (
        'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_60', 
        'ma_100', 'ma_120', 'ma_200',
        'macd', 'macd_signal', 'macd_hist',
        'rsi',
        'bb_upper', 'bb_mid', 'bb_lower', 'bb_pct',
        'atr', 'hv',
        'stoch_k', 'stoch_d',
        'obv',
        'signal',
    )

def create_feeds_from_dataframe(
    df_dict: Dict[str, pd.DataFrame],
) -> Dict[str, DatabaseFeed]:
    """
    DataFrame 딕셔너리에서 직접 Backtrader 피드 생성
    (DB 거치지 않고 파이프라인 결과물 직접 사용 시)
    
    Args:
        df_dict: {ticker: DataFrame} - OHLCV + 지표 포함
            -'date'는 컬럼
    
    Returns:
        {ticker: DatabaseFeed} 딕셔너리
    """
    feeds = {}
    
    for ticker, df in df_dict.items():
        try:
            df = df.copy()
            
            # 빈 df 예외처리
            if df.empty:
                logger.warning(f"{ticker}: Empty DataFrame, skipping")
                continue
            
            # date 처리 (인덱스화 + 정렬)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')
            
            # 피드 파라미터
            #line 기본 변수
            feed_params = {
                'dataname': df,
                'datetime': None,
                'open': 'open' if 'open' in df.columns else -1,
                'high': 'high' if 'high' in df.columns else -1,
                'low': 'low' if 'low' in df.columns else -1,
                'close': 'close' if 'close' in df.columns else -1,
                'volume': 'volume' if 'volume' in df.columns else -1,
                'openinterest': -1,
            }
            
            #line에 지표 추가
            indicator_cols = [
                'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_60',
                'ma_100', 'ma_120', 'ma_200',
                'macd', 'macd_signal', 'macd_hist',
                'rsi', 'bb_upper', 'bb_mid', 'bb_lower', 'bb_pct',
                'atr', 'hv', 'stoch_k', 'stoch_d', 'obv',
                'signal'
            ]
            
            for col in indicator_cols:
                feed_params[col] = col if col in df.columns else -1
            
            feed = DatabaseFeed(**feed_params)
            feeds[ticker] = feed
            
            logger.info(f"{ticker}: Created feed with {len(df)} records")
            
        except Exception as e:
            logger.error(f"{ticker}: Failed to create feed - {e}")
            continue
    
    return feeds
