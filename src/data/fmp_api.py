import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import io
from dotenv import load_dotenv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import List, Dict, Optional, Union

load_dotenv()

API_KEY = os.getenv("FMP_API_KEY")
if not API_KEY:
    raise ValueError("API_KEY를 찾을 수 없습니다. .env 파일을 확인해주세요.")
BASE_URL = "https://financialmodelingprep.com/stable"
DOWNLOAD_FOLDER = "/app/data/fmp"

# 모듈 레벨 rate limiter
_rate_lock = threading.Lock()
_request_timestamps = []
MAX_REQUESTS_PER_MINUTE = 700

#================================#
###     0. 다운로드 관련 함수      ###
#================================#

def _rate_limited_request(url: str, timeout: int = 30) -> requests.Response:
    """
    Rate limiting이 적용된 HTTP GET 요청 함수.
    분당 MAX_REQUESTS_PER_MINUTE 횟수를 초과하지 않도록 제어.

    슬라이딩 윈도우 방식: Lock은 타임스탬프 읽기/쓰기만 담당,
    대기(sleep)는 Lock 해제 후 각 스레드가 독립적으로 수행.
    → 한 스레드가 대기 중에도 다른 스레드는 Lock 획득 가능 (병렬성 유지)
    """
    while True:
        with _rate_lock:
            now = time.time()
            # 1분 초과 타임스탬프 제거
            _request_timestamps[:] = [t for t in _request_timestamps if now - t < 60]

            if len(_request_timestamps) < MAX_REQUESTS_PER_MINUTE:
                # 슬롯 확보: 타임스탬프 기록 후 즉시 Lock 해제
                _request_timestamps.append(now)
                break

            # 슬롯 없음: 가장 오래된 요청이 만료되는 시간 계산
            wait_time = 60 - (now - _request_timestamps[0]) + 0.01  # +10ms 여유
            print(f"⏳ Rate limit 도달. {wait_time:.1f}초 대기...")

        # Lock 해제 후 대기 → 다른 스레드는 이 시간 동안 Lock 획득 가능
        time.sleep(wait_time)

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response

def fmp_down_save(url: str, save_path: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    파일 확장자(.csv, .json)에 따라 알맞게 저장하고 읽어오는 만능 함수
    오류 발생 시 최대 max_retries회 재시도
    """
    # 1. 폴더 생성
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # 2. 캐시 확인 (파일이 이미 있으면 로딩)
    if os.path.exists(save_path):
        print(f"✅ [Cache Hit] : {save_path}")
        try:
            if save_path.endswith('.csv'):
                return pd.read_csv(save_path)
            elif save_path.endswith('.json'):
                return pd.read_json(save_path)
            else:
                return None  # 텍스트 파일 등은 DataFrame 변환 안 함
        except Exception as e:
            print(f"⚠️ 파일 읽기 실패 (손상 가능성 있음): {e}")
            return None

    # 3. 다운로드 (재시도 로직 포함)
    for attempt in range(1, max_retries + 1):
        try:
            if attempt == 1:
                print(f"⬇️ [Download] : {url}")
            else:
                print(f"🔄 [Retry {attempt}/{max_retries}] : {url}")

            response = _rate_limited_request(url, timeout=30)

            # 4. 저장
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"💾 [Saved] : {save_path}")

            # 5. 읽어서 반환
            if save_path.endswith('.csv'):
                return pd.read_csv(io.BytesIO(response.content), encoding='utf-8')
            elif save_path.endswith('.json'):
                return pd.read_json(io.BytesIO(response.content))

        except requests.exceptions.Timeout:
            print(f"❌ [Error] 시간 초과 (Timeout) [{attempt}/{max_retries}]: {url}")
        except requests.exceptions.RequestException as e:
            print(f"❌ [Error] 네트워크/요청 오류 [{attempt}/{max_retries}]: {e}")
        except Exception as e:
            print(f"❌ [Error] 알 수 없는 오류 [{attempt}/{max_retries}]: {e}")

        # 재시도 전 대기 (마지막 시도가 아닐 때만)
        if attempt < max_retries:
            wait = 2 ** attempt  # 2초, 4초, ...
            print(f"⏳ {wait}초 후 재시도...")
            time.sleep(wait)

    print(f"❌ [Failed] 최대 재시도 횟수 초과: {url}")
    return None

def fmp_download_parallel(download_tasks: List[tuple], max_workers: int = 40) -> Dict[str, Optional[pd.DataFrame]]:
    """
    여러 다운로드 작업을 병렬로 실행.

    Args:
        download_tasks: list of (ticker, url, save_path) 튜플
        max_workers: 동시 실행 쓰레드 수

    Returns:
        dict: {ticker: DataFrame or None}
    """
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fmp_down_save, url, path): ticker
            for ticker, url, path in download_tasks
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                print(f"❌ 실패: {ticker} - {e}")
                results[ticker] = None

    return results


#================================#
###       1. 재무제표 데이터       ###
#================================#
#A. (사용불가 / Ultimate Plan 구독 필요) 특정 연도의 모든 기업 재무제표 다운로드
def finanacial_data_year_income(year: int, period: str = "quarter") -> None:
    """특정 연도의 모든 기업의 손익계산서 다운로드"""
    try:
        url = f"{BASE_URL}/income-statement-bulk?year={year}&period={period}&apikey={API_KEY}"
        download_path = f"financial/year/{year}_income.csv"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        fmp_down_save(url, final_path)
    except Exception as e:
        print(f"Error in income: {e}")

def finanacial_data_year_balance_sheet(year: int, period: str = "quarter") -> None:
    """특정 연도의 모든 기업의 대차대조표 다운로드"""
    try:
        url = f"{BASE_URL}/balance-sheet-statement-bulk?year={year}&period={period}&apikey={API_KEY}"
        download_path = f"financial/year/{year}_balance_sheet.csv"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        fmp_down_save(url, final_path)
    except Exception as e:
        print(f"Error in balance: {e}")

def finanacial_data_year_cash_flow(year: int, period: str = "quarter") -> None:
    """특정 연도의 모든 기업의 현금흐름표 다운로드"""
    try:
        url = f"{BASE_URL}/cash-flow-statement-bulk?year={year}&period={period}&apikey={API_KEY}"
        download_path = f"financial/year/{year}_cash_flow.csv"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        fmp_down_save(url, final_path)
    except Exception as e:
        print(f"Error in cash_flow: {e}")


#B. 특정 종목의 과거 전체 데이터 다운로드
def finanacial_data_ticker_income(ticker_list: List[str], period: str = "quarter", limit: int = 1000) -> Dict[str, Optional[pd.DataFrame]]:
    """특정 종목의 과거 전체 손익계산서 다운로드"""
    try:
        tasks = []
        for ticker in ticker_list:
            url = f"{BASE_URL}/income-statement?symbol={ticker}&period={period}&limit={limit}&apikey={API_KEY}"
            download_path = f"financial/ticker/{ticker}/income.json"
            final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
            tasks.append((ticker, url, final_path))
        return fmp_download_parallel(tasks)
    except Exception as e:
        print(e)

def finanacial_data_ticker_balance_statement(ticker_list: List[str], period: str = "quarter", limit: int = 1000) -> Dict[str, Optional[pd.DataFrame]]:
    """특정 종목의 과거 전체 대차대조표 다운로드"""
    try:
        tasks = []
        for ticker in ticker_list:
            url = f"{BASE_URL}/balance-sheet-statement?symbol={ticker}&period={period}&limit={limit}&apikey={API_KEY}"
            download_path = f"financial/ticker/{ticker}/balance_sheet.json"
            final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
            tasks.append((ticker, url, final_path))
        return fmp_download_parallel(tasks)
    except Exception as e:
        print(e)

def finanacial_data_ticker_cash_flow(ticker_list: List[str], period: str = "quarter", limit: int = 1000) -> Dict[str, Optional[pd.DataFrame]]:
    """특정 종목의 과거 전체 현금흐름표 다운로드"""
    try:
        tasks = []
        for ticker in ticker_list:
            url = f"{BASE_URL}/cash-flow-statement?symbol={ticker}&period={period}&limit={limit}&apikey={API_KEY}"
            download_path = f"financial/ticker/{ticker}/cash_flow.json"
            final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
            tasks.append((ticker, url, final_path))
        return fmp_download_parallel(tasks)
    except Exception as e:
        print(e)



#================================#
###        2. 가격 데이터         ###
#================================#
#A. (사용불가 / Ultimate Plan 구독 필요) 특정 날짜의 전 종목 데이터
def price_data_date(date: Union[str, datetime]) -> None:
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

#B.특정 종목의 전 기간 데이터
PRICE_CHUNK_DAYS = 4000  # API 1회 요청당 최대 행 수(5000) 안전마진 포함
def price_data_ticker(ticker_list: List[str], start_date: str, end_date: str) -> Dict[str, Optional[pd.DataFrame]]:
    """
    특정 종목의 기간별 OHLCV 데이터 다운로드.
    API 1회 반환 한도(5000행)를 초과하는 구간은 PRICE_CHUNK_DAYS 단위로 분할 요청 후 통합.
    종목 단위 병렬 처리.
    """
    def _download_one(ticker: str) -> tuple:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt   = datetime.strptime(end_date,   '%Y-%m-%d')
        all_dfs = []
        chunk_start = start_dt

        while chunk_start <= end_dt:
            chunk_end = min(chunk_start + timedelta(days=PRICE_CHUNK_DAYS), end_dt)
            cs = chunk_start.strftime('%Y-%m-%d')
            ce = chunk_end.strftime('%Y-%m-%d')

            url = f"{BASE_URL}/historical-price-eod/full?symbol={ticker}&from={cs}&to={ce}&apikey={API_KEY}"
            tmp_path = os.path.join(DOWNLOAD_FOLDER, f"price/ticker/_tmp_{ticker}_{cs}_to_{ce}.json")

            df = fmp_down_save(url, tmp_path)
            if df is not None and not df.empty:
                all_dfs.append(df)

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            chunk_start = chunk_end + timedelta(days=1)

        if all_dfs:
            merged = pd.concat(all_dfs, ignore_index=True)
            merged = merged.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        else:
            merged = pd.DataFrame()

        final_path = os.path.join(DOWNLOAD_FOLDER, f"price/ticker/{ticker}_{start_date}_to_{end_date}.json")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        merged.to_json(final_path, orient='records', force_ascii=False, indent=4)
        print(f"✅ [Merged] {ticker} 총 {len(merged)}행 → {final_path}")
        # DataFrame을 반환하지 않아 메모리 절약 (파일로 저장됨)
        return ticker

    try:
        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = {executor.submit(_download_one, ticker): ticker for ticker in ticker_list}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ 실패: {ticker} - {e}")
    except Exception as e:
        print(e)



#================================#
###       3. 시가총액 정보         ###
#================================#
#A. (사용불가 / Ultimate Plan 구독 필요) 특정 연도의 모든 기업 시가총액 & 투자지표(PER, PBR 등)
def metrics_data_year(year: int, period: str = "quarter") -> Optional[pd.DataFrame]:
    """특정 연도의 모든 기업 시가총액 & 투자지표(PER, PBR 등)"""
    try:
        url = f"{BASE_URL}/key-metrics-bulk?year={year}&period={period}&apikey={API_KEY}"
        download_path = f"metrics/year/{year}.csv"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        return fmp_down_save(url, final_path)
    except Exception as e:
        print(e)

#B. 특정 종목의 일별 시가총액 전 기간 데이터 다운
MARKET_CAP_CHUNK_DAYS = 4000  # API 1회 반환 한도(5000행) 안전마진 포함
def market_cap_data_ticker_date(ticker_list: List[str], start_date: str, end_date: str) -> Dict[str, Optional[pd.DataFrame]]:
    """
    특정 종목의 일별 시가총액 역사적 데이터를 다운로드합니다.
    API 1회 반환 한도(5000행)를 초과하는 구간은 MARKET_CAP_CHUNK_DAYS 단위로 분할 요청 후 통합.
    종목 단위 병렬 처리.
    """
    def _download_one(ticker: str) -> tuple:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt   = datetime.strptime(end_date,   '%Y-%m-%d')
        all_dfs = []
        chunk_start = start_dt

        while chunk_start <= end_dt:
            chunk_end = min(chunk_start + timedelta(days=MARKET_CAP_CHUNK_DAYS), end_dt)
            cs = chunk_start.strftime('%Y-%m-%d')
            ce = chunk_end.strftime('%Y-%m-%d')

            url = f"{BASE_URL}/historical-market-capitalization?symbol={ticker}&from={cs}&to={ce}&apikey={API_KEY}"
            tmp_path = os.path.join(DOWNLOAD_FOLDER, f"market_cap/_tmp_{ticker}_{cs}_to_{ce}.json")

            df = fmp_down_save(url, tmp_path)
            if df is not None and not df.empty:
                all_dfs.append(df)

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            chunk_start = chunk_end + timedelta(days=1)

        if all_dfs:
            merged = pd.concat(all_dfs, ignore_index=True)
            merged = merged.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        else:
            merged = pd.DataFrame()

        final_path = os.path.join(DOWNLOAD_FOLDER, f"market_cap/{ticker}_{start_date}_to_{end_date}.json")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        merged.to_json(final_path, orient='records', force_ascii=False, indent=4)
        print(f"✅ [Merged] {ticker} 총 {len(merged)}행 → {final_path}")
        # DataFrame을 반환하지 않아 메모리 절약 (파일로 저장됨)
        return ticker

    try:
        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = {executor.submit(_download_one, ticker): ticker for ticker in ticker_list}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ 실패: {ticker} - {e}")
    except Exception as e:
        print(f"❌ Daily Market Cap 다운로드 실패: {e}")



#========================================#
###     4. 상장폐지된/거래중인 종목 리스트     ###
#========================================#
#A.현재 거래중인 종목 리스트
def stock_list_data() -> Optional[pd.DataFrame]:
    """현재 거래중인 전체 종목 리스트"""
    try:
        url = f"{BASE_URL}/stock-list?apikey={API_KEY}"
        download_path = f"stock-list/stock-list.json"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        return fmp_down_save(url, final_path)
    except Exception as e:
        print(e)

#B.상장폐지된 종목 리스트
def delisted_companies_data(limit: int = 5000) -> Dict[int, Optional[pd.DataFrame]]:
    """상장 폐지된 종목 리스트 (100페이지 병렬 다운로드)"""
    try:
        tasks = []
        for page in range(100):
            url = f"{BASE_URL}/delisted-companies?page={page}&limit={limit}&apikey={API_KEY}"
            download_path = f"delisted_companies/{page}.json"
            final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
            tasks.append((page, url, final_path))
        return fmp_download_parallel(tasks)
    except Exception as e:
        print(e)

#C.Screener 활용하여 특정 거래소 거래중인 종목 리스트 불러오기
def stock_screener_exchanges_stock_list(exchange: str, limit: int = 18800) -> Optional[pd.DataFrame]:
    """Stock Screener를 활용하여 특정 거래소의 거래중인 종목 리스트"""
    try:
        url = f"{BASE_URL}/company-screener?exchange={exchange}&limit={limit}&apikey={API_KEY}"
        download_path = f"stock-list/{exchange}.json"
        final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
        return fmp_down_save(url, final_path)
    except Exception as e:
        print(e)



#========================================#
###     5. 특정 회사의 상세 정보 불러오기     ###
#========================================#
def company_profile_data_ticker(ticker_list: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
    """특정 회사의 상세 프로파일 정보 다운로드"""
    try:
        tasks = []
        for ticker in ticker_list:
            url = f"{BASE_URL}/profile?symbol={ticker}&apikey={API_KEY}"
            download_path = f"company_profile/{ticker}.json"
            final_path = os.path.join(DOWNLOAD_FOLDER, download_path)
            tasks.append((ticker, url, final_path))
        return fmp_download_parallel(tasks)
    except Exception as e:
        print(e)
