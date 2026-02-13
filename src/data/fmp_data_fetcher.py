import os
import requests
import pandas as pd
from datetime import datetime
import io
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FMP_API_KEY")
if not API_KEY:
    raise ValueError("API_KEY를 찾을 수 없습니다. .env 파일을 확인해주세요.")
BASE_URL = "https://financialmodelingprep.com/stable"
DOWNLOAD_FOLDER = "data/fmp"

def fmp_down_save(url, save_path):
    """
    파일 확장자(.csv, .json)에 따라 알맞게 저장하고 읽어오는 만능 함수
    """
    # 1. 폴더 생성
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # 2. 캐시 확인 (파일이 이미 있으면 로딩)
    if os.path.exists(save_path):
        print(f"✅ [Cache Hit] : {save_path}")
        try:
            if save_path.endswith('.csv'):
                return pd.read_csv(save_path)
            elif save_path.endswith('.json'):
                return pd.read_json(save_path)
            else:
                return None # 텍스트 파일 등은 DataFrame 변환 안 함
        except Exception as e:
            print(f"⚠️ 파일 읽기 실패 (손상 가능성 있음): {e}")
            # 파일이 깨졌을 경우 삭제 후 다시 받는 로직을 추가할 수도 있음
            return None

    # 3. 다운로드
    print(f"⬇️ [Download] : {url}")
    try:
        # timeout을 설정하여 무한 대기 방지 (30초)
        response = requests.get(url, timeout=30)
        response.raise_for_status() # 404, 500 등 에러 발생 시 즉시 예외 처리
        
        # 4. 저장
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"💾 [Saved] : {save_path}")
        
        # 5. 읽어서 반환
        if save_path.endswith('.csv'):
            # API 데이터가 UTF-8이라고 가정
            return pd.read_csv(io.BytesIO(response.content), encoding='utf-8')
        elif save_path.endswith('.json'):
            return pd.read_json(io.BytesIO(response.content))
        
    except requests.exceptions.Timeout:
        print(f"❌ [Error] 시간 초과 (Timeout): {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ [Error] 네트워크/요청 오류: {e}")
        return None
    except Exception as e:
        print(f"❌ [Error] 알 수 없는 오류: {e}")
        return None

### 1. 재무제표 데이터 ###
#(사용불가 / Ultimate Plan 구독 필요)A. 특정 연도의 모든 기업 재무제표 다운로드
def finanacial_data_year_income(year, period="quarter"):
    """특정 연도의 모든 기업의 손익계산서 다운로드"""
    try: 
        url = f"{BASE_URL}/income-statement-bulk?year={year}&period={period}&apikey={API_KEY}"
        download_path = f"financial/year/{year}_income.csv"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        fmp_down_save(url, final_path)
    except Exception as e:
        print(f"Error in income: {e}")

def finanacial_data_year_balance_sheet(year, period="quarter"):
    """특정 연도의 모든 기업의 대차대조표 다운로드"""
    try: 
        url = f"{BASE_URL}/balance-sheet-statement-bulk?year={year}&period={period}&apikey={API_KEY}"
        download_path = f"financial/year/{year}_balance_sheet.csv"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        fmp_down_save(url, final_path)
    except Exception as e:
        print(f"Error in balance: {e}")

def finanacial_data_year_cash_flow(year, period="quarter"):
    """특정 연도의 모든 기업의 현금흐름표 다운로드"""
    try: 
        url = f"{BASE_URL}/cash-flow-statement-bulk?year={year}&period={period}&apikey={API_KEY}"
        download_path = f"financial/year/{year}_cash_flow.csv"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        fmp_down_save(url, final_path)
    except Exception as e:
        print(f"Error in cash_flow: {e}")

#B. 특정 종목의 과거 전체 데이터 다운로드
def finanacial_data_ticker_income(ticker, period="quarter", limit=1000):
    """특정 종목의 과거 전체 손익계산서 다운로드"""
    try: 
        url = f"{BASE_URL}/income-statement/{ticker}?period={period}&limit={limit}&apikey={API_KEY}"
        download_path = f"financial/ticker/{ticker}_income.json"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        fmp_down_save(url, final_path)
    except Exception as e:
        print(e)

def finanacial_data_ticker_balance_statement(ticker, period="quarter", limit=1000):
    """특정 종목의 과거 전체 대차대조표 다운로드"""
    try: 
        url = f"{BASE_URL}/balance-sheet-statement/{ticker}?period={period}&limit={limit}&apikey={API_KEY}"
        download_path = f"financial/ticker/{ticker}_balance_sheet.json"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        fmp_down_save(url, final_path)
    except Exception as e:
        print(e)

def finanacial_data_ticker_cash_flow(ticker, period="quarter", limit=1000):
    """특정 종목의 과거 전체 현금흐름표 다운로드"""
    try: 
        url = f"{BASE_URL}/cash-flow-statement/{ticker}?period={period}&limit={limit}&apikey={API_KEY}"
        download_path = f"financial/ticker/{ticker}_cash_flow.json"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        fmp_down_save(url, final_path)
    except Exception as e:
        print(e)

### 2. 가격 데이터 ###
#A.특정 종목의 전 기간 데이터
def price_data_ticker(ticker, start_date, end_date):
    """특정 종목의 전 기간 데이터"""
    try: 
        url = f"{BASE_URL}/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={API_KEY}"
        download_path = f"price/ticker/{ticker}_{start_date}_to_{end_date}.json"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        fmp_down_save(url, final_path)
    except Exception as e:
        print(e)

#B.특정 날짜의 전 종목 데이터
def price_data_date(date):
    """특정 날짜의 전 종목 데이터 (입력 date는 문자열 'YYYY-MM-DD' 또는 datetime 객체)"""
    try: 
        # date가 datetime 객체라면 문자열로 변환, 문자열이면 그대로 사용
        if isinstance(date, datetime):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = str(date)
            
        url = f"{BASE_URL}/batch-request-eod-prices?date={date_str}&apikey={API_KEY}"
        
        # 날짜 문자열에서 연도 추출 (YYYY-MM-DD 형식 가정)
        year = date_str.split('-')[0]
        
        download_path = f"price/date/{year}/{date_str}.json"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        fmp_down_save(url, final_path)
    except Exception as e:
        print(f"Error in price_data_date: {e}")

### 3. 시가총액 정보 ###
#A.(사용불가 / Ultimate Plan 구독 필요)특정 연도의 모든 기업 시가총액 & 투자지표(PER, PBR 등)
def metrics_data_year(year, period = "quarter"):
    """특정 연도의 모든 기업 시가총액 & 투자지표(PER, PBR 등)"""
    try: 
        url = f"{BASE_URL}/key-metrics-bulk?year={year}&period={period}&apikey={API_KEY}"
        download_path = f"metrics/year/{year}.csv"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        return fmp_down_save(url, final_path)
    except Exception as e:
        print(e)

#B. 특정 종목의 일별 시가총액 전 기간 데이터 다운
def market_cap_data_ticker_date(ticker, start_date, end_date, limit=100):
    """
    특정 종목의 일별 시가총액 역사적 데이터를 다운로드합니다.
    limit=5000 설정 시 주말 제외 약 20년치 데이터를 가져옵니다.
    """
    try:
        url = f"{BASE_URL}/historical-market-capitalization?symbol={ticker}&from={start_date}&to={end_date}&limit={limit}&apikey={API_KEY}"
        download_path = f"market_cap/{ticker}_{start_date}_to_{end_date}.json"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        return fmp_down_save(url, final_path)
    except Exception as e:
        print(f"❌ {ticker} Daily Market Cap 다운로드 실패: {e}")

### 4. 상장폐지된/거래중인 종목 리스트 ###
#A.현재 거래중인 종목 리스트
def stock_list_data():
    """현재 거래중인 종목 리스트"""
    try: 
        url = f"{BASE_URL}/stock-list?apikey={API_KEY}"
        download_path = f"stock-list/stock-list.json"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        return fmp_down_save(url, final_path)
    except Exception as e:
        print(e)

#B.상장폐지된 종목 리스트
def delisted_companies_data(page = 0, limit = 5000):
    """상장 폐지된 종목 리스트"""
    try: 
        url = f"{BASE_URL}/delisted-companies?page={page}&limit={limit}&apikey={API_KEY}"
        download_path = f"delisted_companies/{page}.json"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        return fmp_down_save(url, final_path)
    except Exception as e:
        print(e)

#C.Screener 활용하여 특정 거래소 거래중인 종목 리스트 불러오기
def stock_screener_exchanges_stock_list(exchange, limit = 18800):
    """Stock Screener활용하여 특정 거래소의 거래중인 종목 리스트"""
    try: 
        url = f"{BASE_URL}/company-screener?exchange={exchange}&limit={limit}&apikey={API_KEY}"
        download_path = f"stock-list/{exchange}.json"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        return fmp_down_save(url, final_path)
    except Exception as e:
        print(e)

### 5. 특정 회사의 상세 정보 불러오기 ###
def company_profile_data_ticker(ticker):
    """특정 회사의 상세 프로파일 정보 불러오기"""
    try: 
        url = f"{BASE_URL}/profile?symbol={ticker}&apikey={API_KEY}"
        download_path = f"company_profile/{ticker}.json"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        return fmp_down_save(url, final_path)
    except Exception as e:
        print(e)

#특정 날짜의 주가 데이터 불러오기
    #
#목표하는 연도의 모든 기업의 시가총액 불러오기
    #원하는 거래소만 필터링
#시가총액 기준 50% 필터링하여 티커 리스트 생성
#티커 리스트에 있는 기업들의 재무제표 전체 불러오기 (중복 호출 방지)
    #각 종목의 재무제표 NULL값 / date값 조정


# 실행!
if __name__ == "__main__":

    exchanges = ["NASDAQ", "NYSE", "AMEX"]
    all_data = []
    for exchange in exchanges:
        df = stock_screener_exchanges_stock_list(exchange)
        all_data.append(df)
    
    final_df = pd.concat(all_data,ignore_index=True)
    
    #필터링
    final_df = final_df[(final_df['isEtf'] == False) & (final_df['isFund'] == False)]

    alive_ticker = final_df['symbol'].copy()
    ipo_dates = {}

    for ticker in alive_ticker:
        profile_df = company_profile_data_ticker(ticker)
        if profile_df is not None and not profile_df.empty:
            ipo_date = profile_df['ipoDate'] if 'ipoDate' in profile_df.columns else None
            ipo_dates[ticker] = ipo_date
        else:
            ipo_dates[ticker] = None
    print(ipo_dates)   
    print(len(ipo_dates))

    ## 2. final_df의 'symbol' 컬럼을 기준으로 딕셔너리 데이터를 매핑(Mapping)
    #final_df['ipoDate'] = final_df['symbol'].map(ipo_dates)

    ## 결과 확인
    #print(final_df[['symbol', 'ipoDate']].head(50))