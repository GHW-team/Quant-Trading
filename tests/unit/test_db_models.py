"""
데이터베이스 ORM 모델 단위 테스트
Ticker, DailyPrice, TechnicalIndicator 모델 검증
"""

import pytest
import os
import tempfile
from datetime import datetime, timezone, date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from src.data.db_models import (
    Base, Ticker, DailyPrice, TechnicalIndicator,
    create_tables, get_utc_now
)


@pytest.fixture
def temp_db():
    """임시 테스트 DB"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        engine = create_engine(f'sqlite:///{db_path}', echo=False)

        # Foreign key 활성화
        from sqlalchemy import event
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        yield session, engine

        session.close()
        engine.dispose()

# get_ut_now
class TestUtilityFunctions:
    """유틸리티 함수 테스트"""

    def test_get_utc_now(self):
        """get_utc_now 함수"""
        before = datetime.now(timezone.utc)
        result = get_utc_now()
        after = datetime.now(timezone.utc)

        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc
        assert before <= result <= after

    def test_get_utc_now_timezone_aware(self):
        """UTC 시간대인지 확인"""
        result = get_utc_now()
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc

# Ticker
class TestTickerModel:
    """Ticker 모델 테스트"""
    #==========================================
    # 함수 정상 동작 테스트
    #==========================================
    def test_ticker_creation(self, temp_db):
        """Ticker 레코드 생성"""
        session, _ = temp_db

        ticker = Ticker(
            ticker_code='005930.KS',
            name='삼성전자',
            market='KOSPI',
            sector='전자'
        )
        session.add(ticker)
        session.commit()

        result = session.query(Ticker).filter_by(ticker_code='005930.KS').first()
        assert result is not None
        assert result.name == '삼성전자'
        assert result.market == 'KOSPI'

    def test_ticker_repr(self, temp_db):
        """Ticker 문자열 표현"""
        session, _ = temp_db

        ticker = Ticker(ticker_code='005930.KS', name='삼성전자')
        session.add(ticker)
        session.commit()

        result = session.query(Ticker).filter_by(ticker_code='005930.KS').first()
        repr_str = repr(result)
        assert '005930.KS' in repr_str
        assert '삼성전자' in repr_str

    def test_ticker_auto_increment_id(self, temp_db):
        """Ticker ID 자동 증가"""
        session, _ = temp_db

        ticker1 = Ticker(ticker_code='005930.KS', name='삼성전자')
        ticker2 = Ticker(ticker_code='000660.KS', name='LG전자')
        session.add_all([ticker1, ticker2])
        session.commit()

        t1 = session.query(Ticker).filter_by(ticker_code='005930.KS').first()
        t2 = session.query(Ticker).filter_by(ticker_code='000660.KS').first()

        assert t1.ticker_id is not None
        assert t2.ticker_id is not None
        assert t1.ticker_id + 1 == t2.ticker_id
    #==========================================
    # 함수 엣지 케이스 테스트
    #==========================================
    def test_ticker_nullable_fields(self, temp_db):
        """Ticker 선택적 필드"""
        session, _ = temp_db

        ticker = Ticker(ticker_code='005930.KS')  # name, market, sector 없음
        session.add(ticker)
        session.commit()

        result = session.query(Ticker).filter_by(ticker_code='005930.KS').first()
        assert result is not None
        assert result.name is None
        assert result.market is None
        assert result.sector is None
    #==========================================
    # 함수 에러 케이스 테스트
    #==========================================
    def test_ticker_unique_constraint(self, temp_db):
        """Ticker 코드 중복 제약"""
        session, _ = temp_db

        ticker1 = Ticker(ticker_code='005930.KS', name='삼성전자')
        ticker2 = Ticker(ticker_code='005930.KS', name='다른회사')

        session.add(ticker1)
        session.commit()

        session.add(ticker2)
        with pytest.raises(IntegrityError):
            session.commit()

# DailyPrice
class TestDailyPriceModel:
    """DailyPrice 모델 테스트"""

    @pytest.fixture
    def sample_ticker(self, temp_db):
        """테스트용 Ticker"""
        session, _ = temp_db
        ticker = Ticker(ticker_code='005930.KS', name='삼성전자')
        session.add(ticker)
        session.commit()
        return ticker
    #==========================================
    # 함수 정상 동작 테스트
    #==========================================
    def test_daily_price_creation(self, temp_db, sample_ticker):
        """DailyPrice 레코드 생성"""
        session, _ = temp_db

        price = DailyPrice(
            ticker_id=sample_ticker.ticker_id,
            date=date(2020, 1, 1),
            open=50000,
            high=51000,
            low=49000,
            close=50500,
            adj_close=50500,
            volume=1000000
        )
        session.add(price)
        session.commit()

        result = session.query(DailyPrice).first()
        assert result is not None
        assert result.open == 50000
        assert result.close == 50500
    
    def test_daily_price_repr(self, temp_db, sample_ticker):
        """DailyPrice 문자열 표현"""
        session, _ = temp_db

        price = DailyPrice(
            ticker_id=sample_ticker.ticker_id,
            date=date(2020, 1, 1),
            open=50000, high=51000, low=49000,
            close=50500, adj_close=50500, volume=1000000
        )
        session.add(price)
        session.commit()

        result = session.query(DailyPrice).first()
        repr_str = repr(result)
        assert 'DailyPrice' in repr_str
        assert '2020-01-01' in repr_str

    def test_daily_price_retrieved_at_default(self, temp_db, sample_ticker):
        """retrieved_at 자동 설정"""
        session, _ = temp_db

        before = datetime.now(timezone.utc).replace(tzinfo=None)
        price = DailyPrice(
            ticker_id=sample_ticker.ticker_id,
            date=date(2020, 1, 1),
            open=50000, high=51000, low=49000,
            close=50500, adj_close=50500, volume=1000000
        )
        session.add(price)
        session.commit()
        after = datetime.now(timezone.utc).replace(tzinfo=None)

        result = session.query(DailyPrice).first()
        assert result.retrieved_at is not None
        assert before <= result.retrieved_at <= after

    def test_daily_price_cascade_delete(self, temp_db, sample_ticker):
        """Ticker 삭제 시 DailyPrice 자동 삭제"""
        session, _ = temp_db

        price = DailyPrice(
            ticker_id=sample_ticker.ticker_id,
            date=date(2020, 1, 1),
            open=50000, high=51000, low=49000,
            close=50500, adj_close=50500, volume=1000000
        )
        session.add(price)
        session.commit()

        # Ticker 삭제
        session.delete(sample_ticker)
        session.commit()

        result = session.query(DailyPrice).count()
        assert result == 0
    #==========================================
    # 함수 엣지 케이스 테스트
    #==========================================
    def test_daily_price_same_date_different_tickers(self, temp_db):
        """같은 날짜, 다른 ticker는 가능"""
        session, _ = temp_db

        ticker1 = Ticker(ticker_code='005930.KS', name='삼성전자')
        ticker2 = Ticker(ticker_code='000660.KS', name='LG전자')
        session.add_all([ticker1, ticker2])
        session.commit()

        price1 = DailyPrice(
            ticker_id=ticker1.ticker_id,
            date=date(2020, 1, 1),
            open=50000, high=51000, low=49000,
            close=50500, adj_close=50500, volume=1000000
        )
        price2 = DailyPrice(
            ticker_id=ticker2.ticker_id,
            date=date(2020, 1, 1),
            open=60000, high=61000, low=59000,
            close=60500, adj_close=60500, volume=1100000
        )

        session.add_all([price1, price2])
        session.commit()

        result = session.query(DailyPrice).count()
        assert result == 2
    #==========================================
    # 함수 에러 처리 테스트
    #==========================================
    def test_daily_price_unique_constraint(self, temp_db, sample_ticker):
        """DailyPrice ticker_id + date 중복 제약"""
        session, _ = temp_db

        price1 = DailyPrice(
            ticker_id=sample_ticker.ticker_id,
            date=date(2020, 1, 1),
            open=50000, high=51000, low=49000,
            close=50500, adj_close=50500, volume=1000000
        )
        price2 = DailyPrice(
            ticker_id=sample_ticker.ticker_id,
            date=date(2020, 1, 1),
            open=50100, high=51100, low=49100,
            close=50600, adj_close=50600, volume=1100000
        )

        session.add(price1)
        session.commit()

        session.add(price2)
        with pytest.raises(IntegrityError):
            session.commit()

# TechnicalIndicator
class TestTechnicalIndicatorModel:
    """TechnicalIndicator 모델 테스트"""

    @pytest.fixture
    def sample_ticker(self, temp_db):
        """테스트용 Ticker"""
        session, _ = temp_db
        ticker = Ticker(ticker_code='005930.KS', name='삼성전자')
        session.add(ticker)
        session.commit()
        return ticker
    #==========================================
    # 함수 정상 동작 테스트
    #==========================================
    def test_technical_indicator_creation(self, temp_db, sample_ticker):
        """TechnicalIndicator 레코드 생성"""
        session, _ = temp_db

        indicator = TechnicalIndicator(
            ticker_id=sample_ticker.ticker_id,
            date=date(2020, 1, 1),
            ma_5=50000,
            ma_20=50100,
            ma_200=50200,
            macd=100,
            caculated_version='v2.0'
        )
        session.add(indicator)
        session.commit()

        result = session.query(TechnicalIndicator).first()
        assert result is not None
        assert result.ma_5 == 50000
        assert result.ma_20 == 50100
        assert result.ma_200 == 50200
        assert result.macd == 100
        assert result.caculated_version == 'v2.0'
    
    def test_technical_indicator_calculated_at_default(self, temp_db, sample_ticker):
        """calculated_at 자동 설정"""
        session, _ = temp_db

        before = datetime.now(timezone.utc).replace(tzinfo=None)
        indicator = TechnicalIndicator(
            ticker_id=sample_ticker.ticker_id,
            date=date(2020, 1, 1),
            ma_5=50000
        )
        session.add(indicator)
        session.commit()
        after = datetime.now(timezone.utc).replace(tzinfo=None)

        result = session.query(TechnicalIndicator).first()
        assert result.calculated_at is not None
        assert before <= result.calculated_at <= after

    def test_technical_indicator_cascade_delete(self, temp_db, sample_ticker):
        """Ticker 삭제 시 TechnicalIndicator 자동 삭제"""
        session, _ = temp_db

        indicator = TechnicalIndicator(
            ticker_id=sample_ticker.ticker_id,
            date=date(2020, 1, 1),
            ma_5=50000
        )
        session.add(indicator)
        session.commit()

        session.delete(sample_ticker)
        session.commit()

        result = session.query(TechnicalIndicator).count()
        assert result == 0
    #==========================================
    # 함수 에러 케이스 테스트
    #==========================================
    def test_technical_indicator_unique_constraint(self, temp_db, sample_ticker):
        """TechnicalIndicator ticker_id + date 중복 제약"""
        session, _ = temp_db

        ind1 = TechnicalIndicator(
            ticker_id=sample_ticker.ticker_id,
            date=date(2020, 1, 1),
            ma_5=50000, ma_20=50100, ma_200=50200, macd=100
        )
        ind2 = TechnicalIndicator(
            ticker_id=sample_ticker.ticker_id,
            date=date(2020, 1, 1),
            ma_5=50001, ma_20=50101, ma_200=50201, macd=101
        )

        session.add(ind1)
        session.commit()

        session.add(ind2)
        with pytest.raises(IntegrityError):
            session.commit()

# 통합 테스트트
class TestModelRelationships:
    """모델 관계 테스트"""

    def test_ticker_daily_prices_relationship(self, temp_db):
        """Ticker - DailyPrice 1:N 관계"""
        session, _ = temp_db

        ticker = Ticker(ticker_code='005930.KS', name='삼성전자')
        session.add(ticker)
        session.commit()

        price1 = DailyPrice(
            ticker_id=ticker.ticker_id,
            date=date(2020, 1, 1),
            open=50000, high=51000, low=49000,
            close=50500, adj_close=50500, volume=1000000
        )
        price2 = DailyPrice(
            ticker_id=ticker.ticker_id,
            date=date(2020, 1, 2),
            open=50500, high=51500, low=49500,
            close=51000, adj_close=51000, volume=1100000
        )
        session.add_all([price1, price2])
        session.commit()

        # Ticker에서 daily_prices 접근
        result = session.query(Ticker).filter_by(ticker_code='005930.KS').first()
        assert len(result.daily_prices) == 2

    def test_ticker_indicators_relationship(self, temp_db):
        """Ticker - TechnicalIndicator 1:N 관계"""
        session, _ = temp_db

        ticker = Ticker(ticker_code='005930.KS', name='삼성전자')
        session.add(ticker)
        session.commit()

        ind1 = TechnicalIndicator(
            ticker_id=ticker.ticker_id,
            date=date(2020, 1, 1),
            ma_5=50000
        )
        ind2 = TechnicalIndicator(
            ticker_id=ticker.ticker_id,
            date=date(2020, 1, 2),
            ma_5=50100
        )
        session.add_all([ind1, ind2])
        session.commit()

        result = session.query(Ticker).filter_by(ticker_code='005930.KS').first()
        assert len(result.indicators) == 2

    def test_price_ticker_relationship(self, temp_db):
        """DailyPrice - Ticker 관계"""
        session, _ = temp_db

        ticker = Ticker(ticker_code='005930.KS', name='삼성전자')
        session.add(ticker)
        session.commit()

        price = DailyPrice(
            ticker_id=ticker.ticker_id,
            date=date(2020, 1, 1),
            open=50000, high=51000, low=49000,
            close=50500, adj_close=50500, volume=1000000
        )
        session.add(price)
        session.commit()

        result = session.query(DailyPrice).first()
        assert result.ticker.ticker_code == '005930.KS'
        assert result.ticker.name == '삼성전자'