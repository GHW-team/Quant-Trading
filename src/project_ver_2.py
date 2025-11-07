#코스피 시총 상위 n개 추출하기

import pandas as pd
import yfinance as yf
import sqlite3
from datetime import datetime

# 1. KOSPI 전체 종목 리스트 가져오기
# Excel 파일 읽기 (.xls)
file_path = r"C:\Users\najdorf\Downloads\korea_stock_list.xls"

kospi_all = pd.read_html(file_path)

# 상장구분이 KOSPI인 종목만 선택
kospi_all = kospi_all[0]
kospi_all.rename(columns=lambda x: x.strip(), inplace=True)
# 1. '유가' 항목만 추출 (즉, KOSPI 종목)
kospi = kospi_all[kospi_all['시장구분'] == '유가'].copy()

# 2. 종목코드를 문자열로 변환하고, 6자리 맞춘 뒤 ".KS" 붙이기
kospi['티커'] = kospi['종목코드'].astype(str).str.zfill(6) + '.KS'

# 3. 결과 일부 확인
#print(kospi[['회사명', '티커']])

#2. yfinance로 시가총액 조회
import time, random
market_cap = []
kospi_tickers = kospi['티커'].tolist()

for i, ticker in enumerate(kospi_tickers):
    try:
        stock = yf.Ticker(ticker)
        cap = stock.info.get('marketCap',0)
    except Exception as e:
        print(f"{ticker}조회 실패: {e}")
        cap = 0
    market_cap.append(cap)
    delay = random.uniform(1.0, 2.0)
    print(f"{delay:1f}초 대기 중...\n")
    time.sleep(delay)

kospi['MarketCap'] = market_cap
top_n = 10
top_stocks = kospi.sort_values('MarketCap', ascending=False).head(top_n)
print(top_stocks[['회사명', '티커', 'MarketCap']])

# ===================================
# SQLite 저장
# ===================================

conn = sqlite3.connect("quant_data.db")
cursor = conn.cursor()

# 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS price_data (
    Date TEXT,
    Ticker TEXT,
    Open REAL,
    High REAL,
    Low REAL,
    Close REAL,
    Volume INTEGER,
    PRIMARY KEY (Date, Ticker)
)
''')
conn.commit()

# 다운로드 및 저장 함수
def download_and_save(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=True)
        if df.empty:
            print(f"[경고] {ticker} 데이터 없음")
            return

        df.reset_index(inplace=True)

        # 컬럼명이 ('Date', '') 같은 형태로 나오는 걸 방지
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        df['Ticker'] = ticker

        # Date 컬럼을 문자열로 변환
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        for idx, row in df.iterrows():
            cursor.execute('''
            INSERT OR REPLACE INTO price_data (Date, Ticker, Open, High, Low, Close, Volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (row['Date'], row['Ticker'], row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
        conn.commit()
        print(f"{ticker} 저장 완료")

    except Exception as e:
        print(f"[에러] {ticker} 다운로드 실패: {e}")



# 상위 N개 종목 다운로드 후 DB 저장
top_tickers = top_stocks['티커'].tolist()
for ticker in top_tickers:
    download_and_save(ticker)

# sql 저장용 테스트
#test_tickers = ["005930.KS", "000660.KS", "373220.KS"]
#for ticker in test_tickers:
    #download_and_save(ticker)
conn.close()
print("모든 데이터 다운로드 및 저장 완료")

#매수시그널
#1. sqlite에서 데이터 불러오기

conn = sqlite3.connect("quant_data.db")
df = pd.read_sql_query("SELECT * FROM price_data ORDER BY Ticker, Date", conn)
conn.close()

#2. MACD계산
ema_short = 12
ema_long = 26
ema_signal = 9

def calculating_macd(group):
    group = group.sort_values('Date')
    group['EMA12'] = group['Close'].ewm(span=ema_short, adjust=False).mean()
    group['EMA26'] = group['Close'].ewm(span=ema_long, adjust=False).mean()
    group['MACD'] = group['EMA12'] - group['EMA26']
    group['Signal'] = group['MACD'].ewm(span=ema_signal, adjust=False).mean()

    #매수시그널 1: 전일대비 MACD 양수 전환
    group['MACD_prev'] = group['MACD'].shift(1)
    group['buy_signal_1'] = ((group['MACD_prev']<0)&(group['MACD']>0)).astype(int) #1=True, 0=False
    return group

df_macd = df.groupby('Ticker').apply(calculating_macd).reset_index(drop=True)

print(df_macd[['Ticker', 'Date', 'Close', 'MACD', 'buy_signal_1']].head(10))