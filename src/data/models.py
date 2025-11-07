# src/data/models.py - 테이블 정의

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date, 
    DateTime, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Ticker(Base):
    """종목 메타데이터"""
    __tablename__ = 'tickers'
    
    ticker_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_code = Column(String(20), unique=True, nullable=False)
    name = Column(String(100))
    market = Column(String(20))
    sector = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<Ticker(code='{self.ticker_code}', name='{self.name}')>"


class DailyPrice(Base):
    """일별 가격 데이터"""
    __tablename__ = 'daily_prices'
    
    price_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey('tickers.ticker_id'), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    adjusted_close = Column(Float)
    retrieved_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        UniqueConstraint('ticker_id', 'date', name='uix_ticker_date'),
        Index('idx_ticker_date', 'ticker_id', 'date'),
    )
    
    def __repr__(self):
        return f"<DailyPrice(ticker_id={self.ticker_id}, date={self.date})>"


class TechnicalIndicator(Base):
    """기술적 지표"""
    __tablename__ = 'technical_indicators'
    
    indicator_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey('tickers.ticker_id'), nullable=False)
    date = Column(Date, nullable=False)
    # 이동평균선
    ma_5 = Column(Float)
    ma_20 = Column(Float)
    ma_60 = Column(Float)
    ma_200 = Column(Float)
    # MACD
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    # RSI
    rsi_14 = Column(Float)
    # Bollinger Bands
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    # 메타데이터
    calculated_at = Column(DateTime, default=datetime.now)
    calculation_version = Column(String(20), default='v1.0')
    
    __table_args__ = (
        UniqueConstraint('ticker_id', 'date', name='uix_ticker_date_ind'),
        Index('idx_ticker_date_ind', 'ticker_id', 'date'),
    )


def create_tables(db_path: str = "data/database/stocks.db"):
    """모든 테이블 생성"""
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    Base.metadata.create_all(engine)
    print(f"✓ All tables created in {db_path}")
    return engine