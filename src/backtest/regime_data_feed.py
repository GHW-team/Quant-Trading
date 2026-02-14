"""
레짐 스위칭 전략용 커스텀 데이터 피드

기존 DatabaseFeed에 p_trend(추세 확률) 라인을 추가합니다.
"""
import backtrader as bt
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class RegimeDataFeed(bt.feeds.PandasData):
    """
    레짐 전략용 데이터 피드
    
    기본 OHLCV + 기술적 지표 + p_trend(ML 예측 확률)
    """
    params = (
        ('datetime', 'date'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
        
        # 전략에 필요한 지표
        ('ma_200', -1),
        ('ma_50', -1),
        ('rsi', -1),
        ('bb_upper', -1),
        ('bb_mid', -1),
        ('bb_lower', -1),
        
        # ML 레짐 예측 확률 (SPY 피드에만 존재)
        ('p_trend', -1),
    )
    
    lines = (
        'ma_200', 'ma_50',
        'rsi',
        'bb_upper', 'bb_mid', 'bb_lower',
        'p_trend',
    )


def create_regime_feeds(
    spy_df: pd.DataFrame,
    stock_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, RegimeDataFeed]:
    """
    SPY + 개별 종목 데이터프레임을 RegimeDataFeed로 변환
    
    Args:
        spy_df: SPY 데이터프레임 (p_trend 컬럼 포함)
        stock_dfs: {ticker: DataFrame} 개별 종목 데이터
        
    Returns:
        {'SPY': feed, 'AAPL': feed, ...} 딕셔너리
        SPY가 반드시 첫 번째여야 합니다.
    """
    feeds = {}
    
    # 1. SPY 피드 (p_trend 포함)
    spy_df = spy_df.copy()
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    spy_df = spy_df.sort_values('date').set_index('date')
    
    spy_params = {
        'dataname': spy_df,
        'datetime': None,
        'open': 'open', 'high': 'high', 'low': 'low',
        'close': 'close', 'volume': 'volume', 'openinterest': -1,
        'p_trend': 'p_trend' if 'p_trend' in spy_df.columns else -1,
    }
    # 지표 컬럼 자동 매핑
    for col in ['ma_200', 'ma_50', 'rsi', 'bb_upper', 'bb_mid', 'bb_lower']:
        spy_params[col] = col if col in spy_df.columns else -1
    
    feeds['SPY'] = RegimeDataFeed(**spy_params)
    logger.info(f"SPY 피드 생성 완료 ({len(spy_df)} rows)")
    
    # 2. 개별 종목 피드
    for ticker, df in stock_dfs.items():
        try:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            
            params = {
                'dataname': df,
                'datetime': None,
                'open': 'open', 'high': 'high', 'low': 'low',
                'close': 'close', 'volume': 'volume', 'openinterest': -1,
                'p_trend': -1,  # 개별 종목에는 p_trend 없음
            }
            for col in ['ma_200', 'ma_50', 'rsi', 'bb_upper', 'bb_mid', 'bb_lower']:
                params[col] = col if col in df.columns else -1
            
            feeds[ticker] = RegimeDataFeed(**params)
        except Exception as e:
            logger.error(f"{ticker}: 피드 생성 실패 - {e}")
    
    logger.info(f"총 {len(feeds)}개 피드 생성 완료")
    return feeds
