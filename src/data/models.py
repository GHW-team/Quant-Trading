#Ticker / DailyPrice / TechnicalIndicator

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (Column,Integer,String,
                        Date,Float,DateTime,ForeignKey,
                        UniqueConstraint, Index, create_engine)
from sqlalchemy.orm import relationship 
from datetime import datetime, timezone  
from pathlib import Path

Base = declarative_base()

def get_utc_now():
    return datetime.now(timezone.utc)

class Ticker(Base):
    __tablename__ = 'tickers'

    ticker_id = Column(Integer, primary_key=True, autoincrement= True)
    ticker_code = Column(String(20),unique=True, nullable=False)
    name = Column(String(100))
    market = Column(String(20))
    sector = Column(String(50))
    
    daily_prices = relationship("DailyPrice", back_populates="ticker", cascade="all, delete-orphan")
    indicators = relationship("TechnicalIndicator", back_populates="ticker", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Ticker(code='{self.ticker_code}', name= '{self.name}')>"

class DailyPrice(Base):
    __tablename__ = 'daily_prices'
    
    price_id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    ticker_id = Column(Integer, ForeignKey('tickers.ticker_id', ondelete='CASCADE'),nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    adj_close = Column(Float, nullable=False)
    retrieved_at = Column(DateTime(timezone=True), default=get_utc_now)

    ticker = relationship("Ticker", back_populates="daily_prices")

    __table_args__ = (
        UniqueConstraint('ticker_id', 'date', name='uix_ticker_date'),
        Index('idx_ticker_date','ticker_id', 'date'),
    )

    def __repr__(self):
        return f"<DailyPrice(ticker_id='{self.ticker_id}', date='{self.date}')>"

class TechnicalIndicator(Base):
    __tablename__ = 'technical_indicators'

    ticker_id = Column(Integer, ForeignKey('tickers.ticker_id', ondelete='CASCADE'), nullable=False)
    indicator_id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    ma_5 = Column(Float)
    ma_20 = Column(Float)
    ma_200 = Column(Float)
    macd = Column(Float)
    calculated_version = Column(String(30),default='v1.0')
    calculated_at = Column(DateTime(timezone=True), default=get_utc_now)

    # 관계 설정
    ticker = relationship("Ticker", back_populates="indicators")

    __table_args__ = (
        UniqueConstraint('ticker_id','date', name= 'uix_ticker_date_ind'),
        Index('idx_ticker_date_ind', 'ticker_id','date'),
    )


def create_tables(db_path: str = 'data/database/stocks.db'):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f'sqlite:///{db_path}', echo=False)

    from sqlalchemy import event
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    print(f'All tables created: {db_path}')
    return engine

if __name__ == "__main__":
    create_tables()