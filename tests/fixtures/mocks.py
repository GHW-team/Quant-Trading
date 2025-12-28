"""
Mock 객체 Fixture
yfinance Mock
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_yfinance():
    """yfinance Mock"""
    with patch('yfinance.Ticker') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        # history() 메서드 설정
        mock_instance.history.return_value = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Adj Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000],
        }, index=pd.date_range('2020-01-01', periods=3, name = 'Date'))

        yield mock

@pytest.fixture
def mock_exchange_calendars():
    """
    exchange_calendars Mock (정적)

    Unit 테스트용 - 고정된 4개 날짜 반환
    날짜에 상관없이 항상 동일한 결과를 반환하여 테스트 격리성 보장
    """
    with patch('exchange_calendars.get_calendar') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        mock_instance.sessions_in_range.return_value = pd.DatetimeIndex(
            ['2020-01-01', '2020-01-02', '2020-01-03','2020-01-04']
        ).tz_localize('UTC')

        yield mock


@pytest.fixture
def mock_exchange_calendars_dynamic():
    """
    exchange_calendars Mock (동적)

    Integration 테스트용 - sessions_in_range 호출 시 start/end 파라미터에 맞는 거래일 반환
    실제 exchange_calendars API와 유사하게 동작 (주말 제외)
    """
    with patch('exchange_calendars.get_calendar') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        def sessions_in_range_side_effect(start, end):
            """start ~ end 사이의 거래일 반환 (주말 제외)"""
            # pandas bdate_range로 거래일 생성
            trading_days = pd.bdate_range(start=start, end=end)
            # UTC 타임존 추가 (exchange_calendars 형식)
            return trading_days.tz_localize('UTC')

        mock_instance.sessions_in_range.side_effect = sessions_in_range_side_effect

        yield mock


# ==========================================
# 동적 Mock - 입력 파라미터에 따라 데이터 생성
# ==========================================

def _get_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    거래일만 반환 (주말 제외)

    Args:
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)

    Returns:
        DatetimeIndex with business days only (Monday-Friday)
    """
    return pd.bdate_range(start=start_date, end=end_date)


def _generate_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    base_price: float = 100.0,
    volatility: float = 0.02
) -> pd.DataFrame:
    """
    티커별 가격 데이터 생성 (재현 가능한 랜덤 데이터)

    각 날짜는 티커와 날짜의 조합으로 고유한 시드를 사용하므로,
    같은 날짜는 항상 같은 가격을 반환합니다 (증분 업데이트 시 일관성 보장)

    Args:
        ticker: 티커 코드 (시드값 생성용)
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        base_price: 초기 가격
        volatility: 일일 변동성 (표준편차)

    Returns:
        DataFrame with columns: [Open, High, Low, Close, Volume, Adj Close]
        Index: DatetimeIndex (거래일만)
    """
    # 거래일 생성
    dates = _get_trading_days(start_date, end_date)
    n = len(dates)

    if n == 0:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])

    # 티커 시드값 (기본)
    ticker_seed = hash(ticker) % 10000

    # 각 날짜마다 고유한 시드로 가격 생성 (일관성 보장)
    closes = []
    opens = []
    highs = []
    lows = []
    volumes = []

    for date in dates:
        # 날짜별 고유 시드 (티커 + 날짜)
        # 같은 날짜는 항상 같은 시드를 사용하므로 일관성 보장
        date_seed = (ticker_seed + hash(date.strftime('%Y-%m-%d'))) % 100000
        np.random.seed(date_seed)

        # Close 가격 (날짜별로 독립적으로 생성)
        # 날짜로부터 파생된 가격 (일관성 보장)
        days_from_epoch = (date - pd.Timestamp('2000-01-01')).days
        trend = 1 + (days_from_epoch * 0.0001)  # 장기 상승 추세
        daily_variation = np.random.normal(1, volatility)
        close_price = base_price * trend * daily_variation
        closes.append(close_price)

        # OHLC 생성
        open_price = close_price * (1 + np.random.uniform(-0.01, 0.01))
        opens.append(open_price)

        high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.02))
        highs.append(high_price)

        low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.02))
        lows.append(low_price)

        volume = np.random.randint(1000000, 5000000)
        volumes.append(volume)

    # DataFrame 생성
    data = {
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes,
        'Adj Close': closes,
    }

    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df


def _calculate_start_from_period(end_date: str, period: str) -> str:
    """
    period 파라미터를 start_date로 변환

    Args:
        end_date: 종료일 (YYYY-MM-DD)
        period: yfinance period 형식 (예: '1y', '6mo', '1d')

    Returns:
        start_date (YYYY-MM-DD)
    """
    end_dt = pd.to_datetime(end_date)

    # period 파싱
    if period.endswith('y'):
        years = int(period[:-1])
        start_dt = end_dt - timedelta(days=years * 365)
    elif period.endswith('mo'):
        months = int(period[:-2])
        start_dt = end_dt - timedelta(days=months * 30)
    elif period.endswith('d'):
        days = int(period[:-1])
        start_dt = end_dt - timedelta(days=days)
    else:
        # 기본값: 1년
        start_dt = end_dt - timedelta(days=365)

    return start_dt.strftime('%Y-%m-%d')


@pytest.fixture
def mock_yfinance_dynamic():
    """
    동적 yfinance Mock - 입력 파라미터에 따라 데이터 생성

    특징:
    - start/end/period 파라미터에 맞는 날짜 범위 생성
    - 주말 제외 (거래일만)
    - 티커별로 다른 시드값으로 가격 패턴 생성 (재현 가능)
    """
    with patch('yfinance.Ticker') as mock_ticker_class:
        # 티커 코드 저장용
        current_ticker = {'value': 'DEFAULT.KS'}

        def ticker_init_side_effect(*args, **kwargs):
            """Ticker 생성 시 티커 코드 저장"""
            # ticker는 위치 인자 또는 키워드 인자로 올 수 있음
            if args:
                ticker_code = args[0]
            else:
                ticker_code = kwargs.get('ticker', 'DEFAULT.KS')

            current_ticker['value'] = ticker_code
            mock_instance = MagicMock()
            mock_instance.history.side_effect = history_side_effect
            return mock_instance

        def history_side_effect(*args, **kwargs):
            """history() 호출 시 동적으로 데이터 생성"""
            start = kwargs.get('start')
            end = kwargs.get('end')
            period = kwargs.get('period')

            # period → start/end 변환
            if period and not (start and end):
                end = datetime.now().strftime('%Y-%m-%d')
                start = _calculate_start_from_period(end, period)

            # start/end가 없으면 기본값
            if not start or not end:
                end = datetime.now().strftime('%Y-%m-%d')
                start = _calculate_start_from_period(end, '1y')

            # 데이터 생성
            ticker_code = current_ticker['value']
            df = _generate_price_data(ticker_code, start, end)
            return df

        mock_ticker_class.side_effect = ticker_init_side_effect

        yield mock_ticker_class
