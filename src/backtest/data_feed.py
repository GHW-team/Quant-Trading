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
        
        # 추가 지표 라인들 (DB에서 로드)
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
        
        # ML 예측 신호 (외부에서 주입)
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


def create_feeds_from_db(
    db_path: str,
    ticker_codes: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    signals: Optional[Dict[str, pd.Series]] = None,
) -> Dict[str, DatabaseFeed]:
    """
    DB에서 데이터를 로드하여 Backtrader 피드 딕셔너리 생성
    
    Args:
        db_path: 데이터베이스 경로
        ticker_codes: 종목 코드 리스트
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        signals: ML 예측 신호 {ticker: pd.Series(index=date, values=0/1)}
    
    Returns:
        {ticker_code: DatabaseFeed} 딕셔너리
    """
    feeds = {}
    
    with DatabaseManager(db_path=db_path) as db:
        # 가격 데이터 로드
        price_dict = db.load_price_data(
            ticker_codes=ticker_codes,
            start_date=start_date,
            end_date=end_date,
        )
        
        # 지표 데이터 로드
        indicator_dict = db.load_indicators(
            ticker_codes=ticker_codes,
            start_date=start_date,
            end_date=end_date,
        )
    
    for ticker in ticker_codes:
        try:
            price_df = price_dict.get(ticker)
            indicator_df = indicator_dict.get(ticker)
            
            if price_df is None or price_df.empty:
                logger.warning(f"{ticker}: No price data available")
                continue
            
            # 가격 + 지표 병합
            df = price_df.copy()
            
            if indicator_df is not None and not indicator_df.empty:
                # date 컬럼 기준으로 병합
                df = pd.merge(
                    df, 
                    indicator_df, 
                    on='date', 
                    how='left',
                    suffixes=('', '_ind')
                )
            
            # ML 신호 추가 (있는 경우)
            if signals and ticker in signals:
                signal_series = signals[ticker]
                
                # Series를 DataFrame으로 변환하여 병합
                signal_df = signal_series.reset_index()
                signal_df.columns = ['date', 'signal']
                
                df = pd.merge(
                    df,
                    signal_df,
                    on='date',
                    how='left'
                )
                df['signal'] = df['signal'].fillna(0).astype(int)
            else:
                df['signal'] = 0
            
            # date를 datetime으로 변환 및 인덱스 설정
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df = df.set_index('date')
            
            # NaN 처리 (Backtrader는 NaN을 싫어함)
            df = df.fillna(method='ffill').fillna(0)
            
            # 피드 파라미터 동적 설정
            feed_params = {
                'dataname': df,
                'datetime': None,  # 인덱스 사용
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'openinterest': -1,
            }
            
            # 존재하는 지표 컬럼만 파라미터에 추가
            indicator_cols = [
                'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_60',
                'ma_100', 'ma_120', 'ma_200',
                'macd', 'macd_signal', 'macd_hist',
                'rsi', 'bb_upper', 'bb_mid', 'bb_lower', 'bb_pct',
                'atr', 'hv', 'stoch_k', 'stoch_d', 'obv',
                'signal'
            ]
            
            for col in indicator_cols:
                if col in df.columns:
                    feed_params[col] = col
                else:
                    feed_params[col] = -1
            
            # 피드 생성
            feed = DatabaseFeed(**feed_params)
            feeds[ticker] = feed
            
            logger.info(f"{ticker}: Created feed with {len(df)} records")
            
        except Exception as e:
            logger.error(f"{ticker}: Failed to create feed - {e}")
            continue
    
    return feeds


def create_feeds_from_dataframe(
    df_dict: Dict[str, pd.DataFrame],
    signals: Optional[Dict[str, pd.Series]] = None,
) -> Dict[str, DatabaseFeed]:
    """
    DataFrame 딕셔너리에서 직접 Backtrader 피드 생성
    (DB 거치지 않고 파이프라인 결과물 직접 사용 시)
    
    Args:
        df_dict: {ticker: DataFrame} - OHLCV + 지표 포함
        signals: {ticker: pd.Series} - ML 예측 신호
    
    Returns:
        {ticker: DatabaseFeed} 딕셔너리
    """
    feeds = {}
    
    for ticker, df in df_dict.items():
        try:
            df = df.copy()
            
            # 신호 추가
            if signals and ticker in signals:
                signal_series = signals[ticker]
                if isinstance(signal_series.index, pd.DatetimeIndex):
                    df = df.set_index('date') if 'date' in df.columns else df
                    df['signal'] = signal_series
                    df = df.reset_index()
                else:
                    signal_df = signal_series.reset_index()
                    signal_df.columns = ['date', 'signal']
                    df = pd.merge(df, signal_df, on='date', how='left')
                
                df['signal'] = df['signal'].fillna(0).astype(int)
            else:
                df['signal'] = 0
            
            # date 처리
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')
            
            df = df.fillna(method='ffill').fillna(0)
            
            # 피드 파라미터
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
