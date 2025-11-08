import sqlite3
from pathlib import Path

def init_database(db_path: str = "data/database/stocks.db"):
    """데이터베이스 초기화"""
    
    # 경로 생성
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    #날짜는 어느어느 테이블에 넣는게 적절한지 여부 알아보기
    #다음 정보 알아보기: long format
        #2번 방안의 아이디어 적용
        #year_month INTEGER GENERATED ALWAYS AS (strftime('%Y%m', date)) STORED,
    # 1. tickers 테이블
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tickers (
        ticker_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker_code VARCHAR(20) UNIQUE NOT NULL,
        name VARCHAR(100),
        market VARCHAR(20)
    )
    """)
        #추가 항목 후보
        # sector / 상장폐지 여부 / 생성,업데이트 시간
        # Trigger 활용.
    # 인덱스 생성
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_tickers_code 
    ON tickers(ticker_code)
    """)
    
    # 2. daily_prices 테이블
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_prices (
        price_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker_id INTEGER NOT NULL,
        date DATE NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume INTEGER NOT NULL,
        adj_close REAL,
        retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (ticker_id) REFERENCES tickers(ticker_id),
        UNIQUE(ticker_id, date)
    )
    """)
    # 인덱스 생성
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date 
    ON daily_prices(ticker_id, date DESC)
    """)

    
    # 3. technical_indicators 테이블
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS technical_indicators (
        indicator_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker_id INTEGER NOT NULL,
        date DATE NOT NULL,
        ma_5 REAL,
        ma_20 REAL,
        ma_200 REAL,
        macd REAL,
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        calculation_version VARCHAR(20) DEFAULT 'v1.0',
        FOREIGN KEY (ticker_id) REFERENCES tickers(ticker_id),
        UNIQUE(ticker_id, date)
    )
    """)
    #indicator_type VARCHAR(20) NOT NULL,    -- 예: 'MA', 'MACD', 'RSI' 이 형식은 어떤지
    #
    # 인덱스 생성
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_indicators_ticker_date 
    ON technical_indicators(ticker_id, date DESC)
    """)
    
    conn.commit()
    conn.close()
    
    print(f"✓ Database initialized: {db_path}")

if __name__ == "__main__":
    init_database()