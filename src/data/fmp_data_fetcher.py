import os
import glob
import json
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from fmp_api import (
    DOWNLOAD_FOLDER,
    BASE_URL,
    API_KEY,
    fmp_download_parallel,
    stock_screener_exchanges_stock_list,
    company_profile_data_ticker,
    delisted_companies_data,
    market_cap_data_ticker_date,
    price_data_ticker,
    finanacial_data_ticker_income,
    finanacial_data_ticker_balance_statement,
    finanacial_data_ticker_cash_flow,
)
#====================================#
###       PIT 티커 캘린더 함수         ###
#====================================#

# 저장된 캘린더 로드하는 함수
def load_calendar(exchanges: List[str], years: List[int]) -> Dict[str, List[str]]:
    """
    저장된 연도별 캘린더 파일을 로드해 통합 dict로 반환.

    Args:
        exchanges: 거래소 리스트 (예: ["NASDAQ", "NYSE", "AMEX"])
        years:     로드할 연도 리스트 (예: [2020, 2021, 2022])
                   빈 리스트([]) 전달 시 저장된 모든 연도를 자동으로 로드

    Returns:
        dict: {"YYYY-MM-DD": [ticker, ...]}
              여러 거래소 종목이 날짜별로 합산됨 (중복 제거)
    """
    merged: Dict[str, set] = {}

    for exchange in exchanges:
        cal_dir = _calendar_dir(exchange)
        if not os.path.isdir(cal_dir):
            print(f"⚠️ 캘린더 폴더 없음: {cal_dir}")
            continue

        # years=[] 이면 폴더 내 모든 연도 파일 자동 수집
        if years:
            target_years = years
        else:
            target_years = []
            for fname in os.listdir(cal_dir):
                if fname.startswith(f"{exchange}_") and fname.endswith(".json"):
                    try:
                        yr = int(fname.replace(f"{exchange}_", "").replace(".json", ""))
                        target_years.append(yr)
                    except ValueError:
                        pass
            target_years.sort()

        for year in target_years:
            fpath = _calendar_path(exchange, year)
            if not os.path.exists(fpath):
                print(f"⚠️ 파일 없음: {fpath}")
                continue
            with open(fpath, 'r', encoding='utf-8') as f:
                year_cal: Dict[str, list] = json.load(f)
            for date_str, tickers in year_cal.items():
                if date_str not in merged:
                    merged[date_str] = set()
                merged[date_str].update(tickers)

    return {k: list(v) for k, v in sorted(merged.items())}

# 캘린터 fmp에서 다운받아 생성 후 파일로 저장하는 함수
def create_universe_calendar(exchanges: List[str]) -> Dict[str, List[str]]:
    """
    과거~현재까지 거래중인 주식 PIT(Point-In-Time) 캘린더 생성.
    연도별 파일로 분할 저장: universe_calendar/{exchange}/{exchange}_{year}.json

    증분 업데이트 로직:
    - 과거 연도 파일이 이미 존재하면 재생성 스킵 (불변)
    - 현재 연도 파일은 항상 재생성 (당해 연도는 매일 변동 가능)

    Args:
        exchanges: 거래소 리스트 (예: ["NASDAQ", "NYSE", "AMEX"])

    Returns:
        dict: {"YYYY-MM-DD": [ticker, ...]} — 전체 기간 통합 캘린더
    """
    # ── 1. 종목 수집 ──────────────────────────────────────────────
    all_stock_list = []
    for exchange in exchanges:
        stock_list = stock_screener_exchanges_stock_list(exchange)
        all_stock_list.append(stock_list)

    final_df = pd.concat(all_stock_list, ignore_index=True)
    final_df = final_df[(final_df['isEtf'] == False) & (final_df['isFund'] == False)]
    final_df = final_df[final_df['country'] == 'US']
    final_df = final_df[~final_df['companyName'].str.contains('%', na=False)]
    final_df = final_df.drop_duplicates(subset=['symbol'], keep='first').reset_index(drop=True)

    alive_ticker = final_df['symbol'].tolist()

    # ── 2. alive 종목 IPO 날짜 확보 ──────────────────────────────
    profile_results = company_profile_data_ticker(alive_ticker)
    all_df_list = [df for df in profile_results.values() if df is not None and not df.empty]
    all_profile_df = pd.concat(all_df_list, ignore_index=True) if all_df_list else pd.DataFrame()

    alive_ipo_df = all_profile_df[['symbol', 'ipoDate', 'exchange']].copy()
    alive_ipo_df['delisted_date'] = None
    alive_ipo_df = alive_ipo_df.rename(columns={'ipoDate': 'ipo_date'})
    alive_ipo_df = alive_ipo_df[['symbol', 'ipo_date', 'delisted_date', 'exchange']]

    # ── 3. 상장폐지 종목 수집 및 필터 ────────────────────────────
    delisted_results = delisted_companies_data()
    all_delisted_data = [df for df in delisted_results.values() if df is not None and not df.empty]
    delisted_ipo_df = pd.concat(all_delisted_data, ignore_index=True)

    delisted_ipo_df = delisted_ipo_df[['symbol', 'ipoDate', 'delistedDate', 'exchange']].copy()
    delisted_ipo_df = delisted_ipo_df.rename(columns={'ipoDate': 'ipo_date', 'delistedDate': 'delisted_date'})
    delisted_ipo_df = delisted_ipo_df[delisted_ipo_df['exchange'].isin(exchanges)]

    delisted_tickers = delisted_ipo_df['symbol'].tolist()
    delisted_profile_results = company_profile_data_ticker(delisted_tickers)

    us_delisted = []
    for ticker in delisted_tickers:
        profile_df = delisted_profile_results.get(ticker)
        if profile_df is not None and not profile_df.empty:
            row = profile_df.iloc[0]
            if row.get('country') != 'US':
                continue
            if row.get('isEtf', False) or row.get('isFund', False):
                continue
            if '%' in str(row.get('companyName', '')):
                continue
        us_delisted.append(ticker)

    delisted_ipo_df = delisted_ipo_df[delisted_ipo_df['symbol'].isin(us_delisted)]

    # ── 4. 통합 및 전처리 ─────────────────────────────────────────
    all_df = pd.concat([alive_ipo_df, delisted_ipo_df], ignore_index=True)
    all_df['ipo_date']      = pd.to_datetime(all_df['ipo_date'])
    all_df['delisted_date'] = pd.to_datetime(all_df['delisted_date'])

    today = pd.Timestamp.now().normalize()
    all_df['delisted_date'] = all_df['delisted_date'].fillna(today)
    all_df = all_df.dropna(subset=['ipo_date'])

    # ── 5. 연도별 분할 저장 (증분: 과거 연도는 파일 있으면 스킵) ─
    min_year   = all_df['ipo_date'].min().year
    cur_year   = today.year
    all_years  = list(range(min_year, cur_year + 1))

    for exchange in exchanges:
        os.makedirs(_calendar_dir(exchange), exist_ok=True)

    saved, skipped = 0, 0
    for year in all_years:
        is_current_year = (year == cur_year)

        # 각 거래소별로 저장
        for exchange in exchanges:
            fpath = _calendar_path(exchange, year)
            # 과거 연도 파일이 이미 있으면 스킵 (불변)
            if not is_current_year and os.path.exists(fpath):
                skipped += 1
                continue

            # 해당 거래소 종목만 필터링해서 연도 캘린더 생성
            exchange_df = all_df[all_df['exchange'] == exchange].copy()
            # exchange 필드가 없는 alive 종목 보완: screener 결과의 exchange 컬럼 활용
            # (profile에 exchange 없는 경우 screener에서 가져온 exchange 사용)
            year_cal = _build_pit_calendar_from_df(exchange_df, year)

            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(year_cal, f, ensure_ascii=False, separators=(',', ':'))
            saved += 1

        if is_current_year:
            print(f"✅ [{year}] 현재 연도 재생성 완료")

    print(f"✅ 캘린더 저장 완료 — 신규/갱신: {saved}개 파일, 스킵: {skipped}개 파일")

    # ── 6. 전체 캘린더 반환 (모든 거래소 통합) ───────────────────
    return load_calendar(exchanges, [])

# 특정날짜의 시가총액 df 불러오기
def get_market_cap_snapshot(
    exchanges: List[str],
    date: str,
) -> pd.DataFrame:
    """
    특정 시점에 거래중인 종목들의 시가총액을 내림차순으로 반환.

    이미 저장된 market_cap 파일을 읽어서 처리 (다운로드 없음).
    FMP 주식수 오류(전날 대비 99% 이상 급감) 구간 제거.
    캘린더는 date의 연도 파일만 내부에서 자동 로드.

    Args:
        exchanges: 거래소 리스트 (예: ["NASDAQ", "NYSE", "AMEX"])
        date:      기준 날짜 문자열 'YYYY-MM-DD'

    Returns:
        DataFrame: index=symbol, columns=['date', 'marketCap']
                   marketCap 내림차순 정렬
    """
    # 1. 해당 날짜 연도의 캘린더만 로드 → 종목 추출
    year = int(date[:4])
    calendar = load_calendar(exchanges, [year])
    ticker_list = calendar.get(date, [])
    if not ticker_list:
        print(f"⚠️ {date} 캘린더에 종목 없음")
        return pd.DataFrame()

    # 2. 각 종목의 market_cap 파일에서 기준 날짜 데이터 추출
    cutoff = (pd.to_datetime(date) + timedelta(days=2)).strftime('%Y-%m-%d')
    rows = []

    # market_cap 폴더 파일 목록을 1회만 스캔해서 딕셔너리로 캐싱
    # (ticker별로 glob 반복 시 7000+개 파일 폴더를 매번 스캔하여 극심한 성능 저하)
    mcap_dir = os.path.join(DOWNLOAD_FOLDER, "market_cap")
    mcap_file_map: Dict[str, str] = {}
    for fname in os.listdir(mcap_dir):
        if not fname.endswith('.json'):
            continue
        ticker_part = fname.split('_')[0]
        mcap_file_map[ticker_part] = os.path.join(mcap_dir, fname)

    for ticker in ticker_list:
        file_path = mcap_file_map.get(ticker)
        if not file_path:
            continue
        try:
            df = pd.read_json(file_path)
        except Exception:
            continue
        if df.empty:
            continue

        df = df.sort_values('date').reset_index(drop=True)

        # FMP 주식수 오류 탐지: 전날 대비 99% 이상 급감 시점 이전 데이터 제거
        error_idx = df.index[df['marketCap'].pct_change() <= -0.99]
        if len(error_idx) > 0:
            df = df.iloc[error_idx[-1]:]

        # 기준 날짜 ± 2일 이내 데이터 추출 (가장 가까운 1행)
        filtered = df[(df['date'] >= date) & (df['date'] <= cutoff)].tail(1)
        if not filtered.empty:
            rows.append(filtered)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True).set_index('symbol')

    # 3. 이상치 제거: marketCap <= 0
    result = result[result['marketCap'] > 0]

    # 4. 내림차순 정렬
    return result.sort_values('marketCap', ascending=False)

#----------헬퍼 함수---------#
def _build_pit_calendar_from_df(all_ipo_delisted_df: pd.DataFrame, year: int) -> Dict[str, list]:
    """
    종목별 IPO/상장폐지 정보 DataFrame으로부터 특정 연도의 PIT 캘린더를 생성.

    Args:
        all_ipo_delisted_df: columns=[symbol, ipo_date, delisted_date] (datetime 타입)
        year: 생성할 연도 (int)

    Returns:
        dict: {"YYYY-MM-DD": [ticker, ...]} — 해당 연도 날짜만 포함
    """
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end   = pd.Timestamp(f"{year}-12-31")

    days_str = pd.date_range(start=year_start, end=year_end).strftime('%Y-%m-%d')
    pit = {d: set() for d in days_str}

    for row in all_ipo_delisted_df.itertuples():
        # 해당 연도와 겹치는 기간만 처리
        active_start = max(row.ipo_date, year_start)
        active_end   = min(row.delisted_date, year_end)
        if active_start > active_end:
            continue
        for d in pd.date_range(start=active_start, end=active_end).strftime('%Y-%m-%d'):
            if d in pit:
                pit[d].add(row.symbol)

    return {k: list(v) for k, v in pit.items()}

def _calendar_dir(exchange: str) -> str:
    """거래소별 캘린더 저장 폴더 경로."""
    return os.path.join(DOWNLOAD_FOLDER, "universe_calendar", exchange)

def _calendar_path(exchange: str, year: int) -> str:
    """연도별 캘린더 파일 경로. 형식: universe_calendar/{exchange}/{exchange}_{year}.json"""
    return os.path.join(_calendar_dir(exchange), f"{exchange}_{year}.json")
#--------------------------#



#===================================================================#
###       티커 리스트의 다운 가능한 역사적 모든 데이터를 다운로드 하는 함수       ###
#===================================================================#

# FMP 프리미엄 플랜 기준 최대 다운로드 가능 과거 날짜
FMP_EARLIEST_DATE = "1985-01-01"

#1. 시가총액 데이터
def download_all_market_cap(ticker_list: List[str]) -> None:
    """
    티커리스트의 모든 역사적 시가총액 데이터 다운로드.
    기존 파일이 있으면 해당 end_date 다음날부터 오늘까지만 증분 다운로드 후 기존 파일에 합쳐서 저장.

    Args:
        ticker_list: 종목 리스트
    """
    today = datetime.today().strftime('%Y-%m-%d')

    # 증분이 필요한 종목: (ticker, new_start, new_end, existing_filepath or None)
    new_tasks = []      # 기존 파일 없음 → 전체 다운로드
    update_tasks = []   # 기존 파일 있음 → 증분 다운로드

    # market_cap/ 폴더를 1회 스캔해 {ticker: filepath} dict 캐싱 (glob 반복 호출 방지)
    mc_dir = os.path.join(DOWNLOAD_FOLDER, "market_cap")
    mc_file_cache: Dict[str, str] = {}   # ticker → filepath
    mc_end_cache: Dict[str, str] = {}    # ticker → end_date (비교용)
    if os.path.isdir(mc_dir):
        for fname in os.listdir(mc_dir):
            if not fname.endswith('.json'):
                continue
            if fname.startswith('_tmp_'):
                continue
            if '_to_' in fname:
                before_to = fname.split('_to_')[0]   # "{ticker}_{start}"
                end_date_str = fname.split('_to_')[-1].replace('.json', '')
                ticker_part = before_to[:-11]         # 마지막 "_YYYY-MM-DD" 제거
                if not ticker_part:
                    continue
                if ticker_part not in mc_end_cache or end_date_str > mc_end_cache[ticker_part]:
                    mc_file_cache[ticker_part] = os.path.join(mc_dir, fname)
                    mc_end_cache[ticker_part] = end_date_str

    for ticker in ticker_list:
        existing = mc_file_cache.get(ticker)
        if existing is None:
            new_tasks.append(ticker)
        else:
            _, existing_end = _parse_date_range_from_filename(existing)
            # 이미 최신이면 스킵
            if existing_end >= today:
                print(f"✅ [Skip] {ticker} 이미 최신 ({existing_end})")
                continue
            # 기존 end_date 다음날부터 오늘까지 증분 다운로드
            new_start = (datetime.strptime(existing_end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            update_tasks.append((ticker, new_start, today, existing))

    # 전체 다운로드 (기존 파일 없는 종목)
    if new_tasks:
        market_cap_data_ticker_date(new_tasks, FMP_EARLIEST_DATE, today)

    # 증분 다운로드 (기존 파일 있는 종목, 병렬)
    if update_tasks:
        def _incremental_mc_one(task):
            ticker, new_start, new_end, existing_path = task
            inc_path = os.path.join(DOWNLOAD_FOLDER, f"market_cap/{ticker}_{new_start}_to_{new_end}.json")

            market_cap_data_ticker_date([ticker], new_start, new_end)

            existing_df = pd.read_json(existing_path)

            if os.path.exists(inc_path):
                new_df = pd.read_json(inc_path)
                os.remove(inc_path)
            else:
                new_df = pd.DataFrame()

            if not new_df.empty:
                merged_df = pd.concat([existing_df, new_df], ignore_index=True)
                merged_df = merged_df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
            else:
                merged_df = existing_df

            old_start, _ = _parse_date_range_from_filename(existing_path)
            new_path = os.path.join(DOWNLOAD_FOLDER, f"market_cap/{ticker}_{old_start}_to_{new_end}.json")
            os.remove(existing_path)
            merged_df.to_json(new_path, orient='records', force_ascii=False, indent=4)
            print(f"✅ [Updated] {ticker} → {new_path}")

        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = {executor.submit(_incremental_mc_one, task): task[0] for task in update_tasks}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ 증분 실패: {ticker} - {e}")

#2. 가격 데이터
def download_all_price(ticker_list: List[str]) -> None:
    """
    티커리스트의 모든 역사적 가격(OHLCV) 데이터 다운로드.
    기존 파일이 있으면 해당 end_date 다음날부터 오늘까지만 증분 다운로드 후 기존 파일에 합쳐서 저장.

    Args:
        ticker_list: 종목 리스트
    """
    today = datetime.today().strftime('%Y-%m-%d')

    new_tasks = []
    update_tasks = []

    # price/ticker/ 폴더를 1회 스캔해 {ticker: filepath} dict 캐싱 (glob 반복 호출 방지)
    # 같은 ticker에 파일이 여러 개 있으면 end_date가 가장 큰 파일을 선택
    price_dir = os.path.join(DOWNLOAD_FOLDER, "price", "ticker")
    price_file_cache: Dict[str, str] = {}   # ticker → filepath
    price_end_cache: Dict[str, str] = {}    # ticker → end_date (비교용)
    if os.path.isdir(price_dir):
        for fname in os.listdir(price_dir):
            if not fname.endswith('.json'):
                continue
            # _tmp_ 접두사 파일은 임시 파일이므로 스킵
            if fname.startswith('_tmp_'):
                continue
            # 파일명 형식: {ticker}_{start}_to_{end}.json → "_to_" 기준으로 ticker 추출
            if '_to_' in fname:
                before_to = fname.split('_to_')[0]   # "{ticker}_{start}"
                end_date_str = fname.split('_to_')[-1].replace('.json', '')  # "YYYY-MM-DD"
                ticker_part = before_to[:-11]         # 마지막 "_YYYY-MM-DD" 제거
                if not ticker_part:
                    continue
                # 같은 ticker의 기존 캐시보다 end_date가 최신인 경우만 덮어씀
                if ticker_part not in price_end_cache or end_date_str > price_end_cache[ticker_part]:
                    price_file_cache[ticker_part] = os.path.join(price_dir, fname)
                    price_end_cache[ticker_part] = end_date_str

    for ticker in ticker_list:
        existing = price_file_cache.get(ticker)
        if existing is None:
            new_tasks.append(ticker)
        else:
            _, existing_end = _parse_date_range_from_filename(existing)
            if existing_end >= today:
                print(f"✅ [Skip] {ticker} 이미 최신 ({existing_end})")
                continue
            new_start = (datetime.strptime(existing_end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            update_tasks.append((ticker, new_start, today, existing))

    # 전체 다운로드
    if new_tasks:
        price_data_ticker(new_tasks, FMP_EARLIEST_DATE, today)

    # 증분 다운로드 (병렬)
    if update_tasks:
        def _incremental_one(task):
            ticker, new_start, new_end, existing_path = task
            inc_path = os.path.join(DOWNLOAD_FOLDER, f"price/ticker/{ticker}_{new_start}_to_{new_end}.json")

            price_data_ticker([ticker], new_start, new_end)

            existing_df = pd.read_json(existing_path)

            if os.path.exists(inc_path):
                new_df = pd.read_json(inc_path)
                os.remove(inc_path)
            else:
                new_df = pd.DataFrame()

            if not new_df.empty:
                merged_df = pd.concat([existing_df, new_df], ignore_index=True)
                merged_df = merged_df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
            else:
                merged_df = existing_df

            old_start, _ = _parse_date_range_from_filename(existing_path)
            new_path = os.path.join(DOWNLOAD_FOLDER, f"price/ticker/{ticker}_{old_start}_to_{new_end}.json")
            os.remove(existing_path)
            merged_df.to_json(new_path, orient='records', force_ascii=False, indent=4)
            print(f"✅ [Updated] {ticker} → {new_path}")

        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = {executor.submit(_incremental_one, task): task[0] for task in update_tasks}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ 증분 실패: {ticker} - {e}")

#3. 재무제표 데이터
def download_all_financial(ticker_list: List[str]) -> None:
    """
    티커리스트의 모든 역사적 재무제표(손익계산서 + 대차대조표 + 현금흐름표) 다운로드.
    3종류를 통합하여 /app/data/fmp/financial/ticker/{ticker}/all_financial_{last_date}.json 으로 저장.

    증분 업데이트 기준:
    - all_financial_*.json 없음 → 전체 다운로드
    - all_financial_*.json 있음 → 파일명의 날짜로부터 _get_financial_quarter_days() 이상 경과한 경우만 증분 다운로드
      (실적 발표 시즌 중: 250일, 비시즌: 100일)
    - 상장폐지 종목(delistedDate <= 오늘) → 새 분기 데이터 없으므로 skip

    Args:
        ticker_list: 종목 리스트
    """
    today = datetime.today().date()
    quarter_days = _get_financial_quarter_days(today)

    # 상장폐지 종목 dict 로드 (함수 시작 시 1회)
    delisted_map = _load_delisted_set()

    new_tickers = []      # 파일 없는 종목 → 전체 다운로드
    update_tickers = []   # 있지만 새 분기 가능성 있는 종목 → 증분 다운로드

    for ticker in ticker_list:
        # 파일명에서 last_date 파싱 (파일 오픈 없이 glob으로만 판단)
        existing = _find_existing_file(DOWNLOAD_FOLDER, ticker, f"financial/ticker/{ticker}/all_financial_*.json")
        if existing is None:
            # 이미 빈 파일(데이터 없음)로 확인된 종목은 재다운로드 스킵
            ticker_dir = os.path.join(DOWNLOAD_FOLDER, f"financial/ticker/{ticker}")
            if os.path.isdir(ticker_dir):
                income_path = os.path.join(ticker_dir, "income.json")
                if os.path.exists(income_path) and os.path.getsize(income_path) <= 2:
                    print(f"✅ [Skip-NoData] {ticker} FMP 데이터 없음")
                    continue
            new_tickers.append(ticker)
        else:
            basename = os.path.basename(existing)  # all_financial_2025-12-31.json
            last_date_str = basename.replace("all_financial_", "").replace(".json", "")
            try:
                last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
            except ValueError:
                # 날짜 파싱 실패 시 전체 재다운로드
                new_tickers.append(ticker)
                continue

            # 상장폐지 종목 체크: 상장폐지일 이후로는 새 분기 데이터 없음
            if ticker in delisted_map:
                try:
                    delist_date = datetime.strptime(delisted_map[ticker], '%Y-%m-%d').date()
                    if delist_date <= today:
                        print(f"✅ [Skip-Delisted] {ticker} 상장폐지일 {delist_date}, 마지막 보고일 {last_date}")
                        continue
                except ValueError:
                    pass

            days_since = (today - last_date).days
            if days_since >= quarter_days:
                update_tickers.append((ticker, last_date, existing))
            else:
                print(f"✅ [Skip] {ticker} 마지막 보고일 {last_date} ({days_since}일 경과, {quarter_days}일 미만)")

    # ── 전체 다운로드 ──
    if new_tickers:
        income_dict = finanacial_data_ticker_income(new_tickers)
        balance_dict = finanacial_data_ticker_balance_statement(new_tickers)
        cashflow_dict = finanacial_data_ticker_cash_flow(new_tickers)

        for ticker in new_tickers:
            merged = _merge_financial_dfs(
                income_dict.get(ticker),
                balance_dict.get(ticker),
                cashflow_dict.get(ticker),
            )
            if merged is None:
                print(f"⚠️ {ticker} 재무제표 데이터 없음")
                continue

            last_date_str = pd.to_datetime(merged['date']).max().strftime('%Y-%m-%d')
            save_dir = os.path.join(DOWNLOAD_FOLDER, f"financial/ticker/{ticker}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"all_financial_{last_date_str}.json")
            merged.to_json(save_path, orient='records', force_ascii=False, indent=4)
            print(f"✅ [Saved] {ticker} → {save_path}")

    # ── 증분 업데이트 ──
    if update_tickers:
        update_ticker_names = [t for t, _, _ in update_tickers]

        # 경과 일수 기반으로 필요한 분기 수 동적 산출 (안전마진 +4분기)
        income_tasks, balance_tasks, cashflow_tasks = [], [], []
        for ticker, last_date, _ in update_tickers:
            days_since = (today - last_date).days
            limit = (days_since // 90) + 4

            income_tasks.append((
                ticker,
                f"{BASE_URL}/income-statement?symbol={ticker}&period=quarter&limit={limit}&apikey={API_KEY}",
                os.path.join(DOWNLOAD_FOLDER, f"financial/ticker/{ticker}/income_tmp.json"),
            ))
            balance_tasks.append((
                ticker,
                f"{BASE_URL}/balance-sheet-statement?symbol={ticker}&period=quarter&limit={limit}&apikey={API_KEY}",
                os.path.join(DOWNLOAD_FOLDER, f"financial/ticker/{ticker}/balance_sheet_tmp.json"),
            ))
            cashflow_tasks.append((
                ticker,
                f"{BASE_URL}/cash-flow-statement?symbol={ticker}&period=quarter&limit={limit}&apikey={API_KEY}",
                os.path.join(DOWNLOAD_FOLDER, f"financial/ticker/{ticker}/cash_flow_tmp.json"),
            ))

        income_dict = fmp_download_parallel(income_tasks)
        balance_dict = fmp_download_parallel(balance_tasks)
        cashflow_dict = fmp_download_parallel(cashflow_tasks)

        # 임시 파일 정리
        for ticker in update_ticker_names:
            for name in ['income_tmp', 'balance_sheet_tmp', 'cash_flow_tmp']:
                tmp = os.path.join(DOWNLOAD_FOLDER, f"financial/ticker/{ticker}/{name}.json")
                if os.path.exists(tmp):
                    os.remove(tmp)

        for ticker, last_date, existing_path in update_tickers:
            new_merged = _merge_financial_dfs(
                income_dict.get(ticker),
                balance_dict.get(ticker),
                cashflow_dict.get(ticker),
            )
            if new_merged is None:
                print(f"⚠️ {ticker} 재무제표 재다운로드 실패")
                continue

            existing_df = pd.read_json(existing_path)
            new_rows = new_merged[pd.to_datetime(new_merged['date']).dt.date > last_date]

            if new_rows.empty:
                print(f"ℹ️ {ticker} 새 분기 데이터 없음 (마지막 보고일 {last_date} 유지)")
                continue

            merged = pd.concat([existing_df, new_rows], ignore_index=True)
            merged = merged.drop_duplicates(subset=['date', 'period'] if 'period' in merged.columns else ['date'])
            merged = merged.sort_values('date', ascending=False).reset_index(drop=True)

            # 기존 파일 삭제 후 새 날짜로 저장
            new_last_date_str = pd.to_datetime(merged['date']).max().strftime('%Y-%m-%d')
            save_dir = os.path.dirname(existing_path)
            new_path = os.path.join(save_dir, f"all_financial_{new_last_date_str}.json")
            os.remove(existing_path)
            merged.to_json(new_path, orient='records', force_ascii=False, indent=4)
            print(f"✅ [Updated] {ticker} +{len(new_rows)}개 분기 추가 → {new_path}")

#----------헬퍼 함수---------#
def _find_existing_file(folder: str, ticker: str, pattern: str) -> Optional[str]:
    """
    폴더에서 ticker와 관련된 기존 파일을 찾아 경로 반환.
    없으면 None 반환.
    pattern 예: "market_cap/{ticker}_*_to_*.json"
    """
    search_path = os.path.join(DOWNLOAD_FOLDER, pattern.format(ticker=ticker))
    matches = glob.glob(search_path)
    return matches[0] if matches else None

def _parse_date_range_from_filename(filepath: str) -> tuple[str, str]:
    """
    파일명에서 start_date, end_date 파싱.
    파일명 형식: {ticker}_{start_date}_to_{end_date}.json
    반환: (start_date, end_date) 문자열 'YYYY-MM-DD'
    """
    basename = os.path.basename(filepath)           # AAPL_1985-01-01_to_2024-01-01.json
    name = basename.replace(".json", "")            # AAPL_1985-01-01_to_2024-01-01
    parts = name.split("_to_")                      # ['AAPL_1985-01-01', '2024-01-01']
    end_date = parts[-1]                            # '2024-01-01'
    start_date = parts[0].split("_", 1)[-1]        # '1985-01-01'
    return start_date, end_date

def _get_financial_quarter_days(today: date) -> int:
    """
    현재 날짜가 실적 발표 시즌(earnings season)인지에 따라 업데이트 기준일 수를 반환.

    미국 실적 발표 시즌 패턴 (분기 종료 후 약 14~60일):
      Q4(12/31 종료) → 1/14 ~ 2/29 발표
      Q1(3/31 종료)  → 4/14 ~ 5/31 발표
      Q2(6/30 종료)  → 7/14 ~ 8/31 발표
      Q3(9/30 종료)  → 10/14 ~ 11/30 발표

    발표 시즌 중: 아직 발표 안 된 종목이 있으므로 기준을 250일로 늘려
                  직전 분기(last_date)가 두 분기 전인 경우만 업데이트.
    비시즌:       기준 100일 (분기 90일 + 발표 지연 약 10일 후 여유)
    """
    m, d = today.month, today.day

    in_earnings_season = (
        (m == 1 and d >= 14) or (m == 2) or              # Q4 발표 시즌: 1/14~2/28
        (m == 4 and d >= 14) or (m == 5) or              # Q1 발표 시즌: 4/14~5/31
        (m == 7 and d >= 14) or (m == 8) or              # Q2 발표 시즌: 7/14~8/31
        (m == 10 and d >= 14) or (m == 11)               # Q3 발표 시즌: 10/14~11/30
    )

    if in_earnings_season:
        # 검증 결과: 147일(09/30 종목)은 실제 Q4 미발표 → 150일 이상만 업데이트
        # 150일 미만 종목은 시즌 중 미발표 상태이므로 skip이 올바름
        return 150
    else:
        # 검증 결과: 비시즌 시작 시 2025-12-31 종목이 60일 경과 즉시 업데이트 가능
        # SEC 10-Q 제출 기한(40~45일) 이후 여유분 포함
        return 60

def _merge_financial_dfs(income_df, balance_df, cashflow_df) -> Optional[pd.DataFrame]:
    """income / balance / cashflow 3종 DataFrame을 date+period 기준 outer join으로 통합."""
    dfs = [df for df in [income_df, balance_df, cashflow_df] if df is not None and not df.empty]
    if not dfs:
        return None

    merged = dfs[0]
    for df in dfs[1:]:
        merge_keys = [c for c in ['date', 'period'] if c in merged.columns and c in df.columns]
        merged = pd.merge(merged, df, on=merge_keys, how='outer', suffixes=('', '_dup'))
        # 중복 컬럼(_dup) 제거
        merged = merged[[c for c in merged.columns if not c.endswith('_dup')]]

    return merged.sort_values('date', ascending=False).reset_index(drop=True)

def _load_delisted_set() -> dict:
    """
    delisted_companies 폴더의 JSON 파일들을 읽어 {symbol: delistedDate} dict 반환.
    상장폐지 종목 판별에 사용.
    """
    delisted = {}
    delist_dir = os.path.join(DOWNLOAD_FOLDER, "delisted_companies")
    if not os.path.isdir(delist_dir):
        return delisted
    for fname in os.listdir(delist_dir):
        if not fname.endswith('.json'):
            continue
        try:
            with open(os.path.join(delist_dir, fname)) as f:
                rows = json.load(f)
            for r in rows:
                sym = r.get('symbol', '')
                dd = r.get('delistedDate', '')
                if sym and dd:
                    delisted[sym] = dd
        except Exception:
            pass
    return delisted
#--------------------------#



#===================================================================#
###      캘린더 기반 특정 기간의 모든 기업의 모든 역사적 데이터 다운로드 함수     ###
#===================================================================#
def download_data_for_period(
    exchanges: List[str],
    start_date: str,
    end_date: str,
) -> None:
    """
    PIT 캘린더 기반으로 특정 기간에 거래된 모든 종목의
    가격 / 시가총액 / 재무제표 데이터를 다운로드.

    캘린더는 start_date ~ end_date에 해당하는 연도 파일만 내부에서 자동 로드.

    중복 다운로드 방지:
    - start_date ~ end_date 전체 날짜에 걸쳐 등장한 유니크 티커 합집합을 먼저 구성
    - 이후 download_all_* 함수를 한 번씩만 호출
    - download_all_* 내부에 캐시/증분 로직이 있어 이미 다운된 종목은 자동 스킵

    Args:
        exchanges:  거래소 리스트 (예: ["NASDAQ", "NYSE", "AMEX"])
        start_date: 시작 날짜 문자열 'YYYY-MM-DD'
        end_date:   종료 날짜 문자열 'YYYY-MM-DD'
    """
    # 필요한 연도만 로드 (메모리 절약)
    start_year = int(start_date[:4])
    end_year   = int(end_date[:4])
    years = list(range(start_year, end_year + 1))
    calendar = load_calendar(exchanges, years)

    # start_date ~ end_date 범위의 유니크 티커 합집합 수집
    all_dates = pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d')

    unique_tickers: set = set()
    for date_str in all_dates:
        unique_tickers.update(calendar.get(date_str, []))

    ticker_list = sorted(unique_tickers)
    print(f"📋 기간 {start_date} ~ {end_date} 유니크 티커 수: {len(ticker_list)}")

    if not ticker_list:
        print("⚠️ 해당 기간에 티커가 없습니다.")
        return

    # 가격 데이터 다운로드 (캐시/증분 자동 처리)
    print("\n[ 1/3 ] 가격 데이터 다운로드 시작")
    download_all_price(ticker_list)

    # 시가총액 데이터 다운로드 (캐시/증분 자동 처리)
    print("\n[ 2/3 ] 시가총액 데이터 다운로드 시작")
    download_all_market_cap(ticker_list)

    # 재무제표 데이터 다운로드 (캐시/증분 자동 처리)
    print("\n[ 3/3 ] 재무제표 데이터 다운로드 시작")
    download_all_financial(ticker_list)

    print(f"\n✅ 전체 다운로드 완료: {len(ticker_list)}개 종목")

