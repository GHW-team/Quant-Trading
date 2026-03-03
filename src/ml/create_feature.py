"""
create_feature.py
=================
리밸런싱 날짜(rebalance_date) 기준으로 시장 / 산업 / 스타일 팩터를 계산한다.

## 미래편향(Look-ahead bias) 방지 원칙
- 재무제표: trading_date >= filingDate 를 만족하는 가장 최근 분기만 사용.
            filingDate 가 없는 경우 period_end_date + 90일 (3개월 lag) 을 기준으로 사용.
- 시가총액: 리밸런싱 전일(T-1) 기준값 사용.
- 주가:     FMP /historical-price-eod/full 엔드포인트의 close 는 수정주가(adjusted).
            adjClose 별도 컬럼 없이 close 자체가 split/dividend 반영된 값임.

## 팩터 목록
 시장   : market_excess_return  (SPY 1개월 수익률 - Treasury 3개월물 환산 무위험 수익률)
 산업   : industry_*            (GICS Level-2 기반 원핫 인코딩 더미)
 스타일 :
   value_bp       B/P   (장부가치 / 시가총액)
   value_ep       E/P   (최근 4분기 순이익 합 / 시가총액)
   size           Log(시가총액)
   profitability  (revenue - costOfRevenue - SGA) / totalAssets  (영업이익/자산, 4분기 분자 합 / 최신 분모)
   investment     YoY 총자산 성장률
   momentum       P_{t-1} / P_{t-12} - 1  (1개월 전 / 12개월 전 가격)
   low_vol        과거 252거래일 일간 수익률 표준편차
   stability      총부채 / 총자산  (leverage)

 위 스타일 팩터 각각에 대해 섹터별 Z-score 컬럼 (suffix: _z) 도 함께 반환.

## 사용법
    from create_feature import FeatureCreator

    fc = FeatureCreator(data_dir="/app/data/fmp")
    df = fc.compute(
        ticker_list    = ["AAPL", "MSFT", ...],
        rebalance_date = "2023-06-30",
    )
    # df: index=symbol, columns=팩터들 + _z Z-score 컬럼들
"""

import os
import warnings
import multiprocessing as mp
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from src.data.fmp_db_manager import FmpDatabaseManager
from src.data.fmp_db_models import FINANCIAL_COLUMN_MAP_REVERSE

warnings.filterwarnings("ignore", category=FutureWarning)

# ── DB snake_case → JSON camelCase 컬럼 매핑 (팩터 계산 코드 호환) ──────────
_PRICE_DB_RENAME = {"close": "adj_close"}
_MCAP_DB_RENAME = {"market_cap": "marketCap"}
_TREASURY_DB_RENAME = {"month_3": "month3"}
_FINANCIAL_META_RENAME = {"filing_date": "filingDate"}


# ──────────────────────────────────────────────────────────────────────────────
# GICS Level-2 산업 매핑
# company_profile / stock-list 의 industry 문자열 → 전략의 GICS Level-2 카테고리
# ──────────────────────────────────────────────────────────────────────────────
GICS_MAP: Dict[str, str] = {
    # 정보기술
    "Software": "IT_Software_Services",
    "Software-Application": "IT_Software_Services",
    "Software-Infrastructure": "IT_Software_Services",
    "Software - Application": "IT_Software_Services",
    "Software - Infrastructure": "IT_Software_Services",
    "Information Technology Services": "IT_Software_Services",
    "Tech Distribution": "IT_Software_Services",
    "IT Services": "IT_Software_Services",
    "Internet Services & Infrastructure": "IT_Software_Services",
    "Computer Hardware": "IT_Hardware_Equipment",
    "Consumer Electronics": "IT_Hardware_Equipment",
    "Electronic Components": "IT_Hardware_Equipment",
    "Electronics & Computer Distribution": "IT_Hardware_Equipment",
    "Scientific & Technical Instruments": "IT_Hardware_Equipment",
    "Hardware, Equipment & Parts": "IT_Hardware_Equipment",
    "Communication Equipment": "IT_Hardware_Equipment",
    "Semiconductors": "IT_Semiconductors",
    "Semiconductor Equipment & Materials": "IT_Semiconductors",
    "Semiconductors & Semiconductor Equipment": "IT_Semiconductors",
    # 커뮤니케이션
    "Internet Content & Information": "Comm_Media_Entertainment",
    "Entertainment": "Comm_Media_Entertainment",
    "Broadcasting": "Comm_Media_Entertainment",
    "Publishing": "Comm_Media_Entertainment",
    "Advertising Agencies": "Comm_Media_Entertainment",
    "Music & Entertainment": "Comm_Media_Entertainment",
    "Electronic Gaming & Multimedia": "Comm_Media_Entertainment",
    "Media & Entertainment": "Comm_Media_Entertainment",
    "Interactive Media & Services": "Comm_Media_Entertainment",
    "Movies & Entertainment": "Comm_Media_Entertainment",
    "Cable & Satellite": "Comm_Media_Entertainment",
    "Telecom Services": "Comm_Telecom",
    "Communication Services": "Comm_Telecom",
    "Wireless Telecom Services": "Comm_Telecom",
    "Telecommunications Services": "Comm_Telecom",
    "Diversified Telecommunication Services": "Comm_Telecom",
    "Wireless Telecommunications Services": "Comm_Telecom",
    # 임의소비재
    "Auto Manufacturers": "Cons_Disc_Auto",
    "Auto Parts": "Cons_Disc_Auto",
    "Auto & Truck Dealerships": "Cons_Disc_Auto",
    "Recreational Vehicles": "Cons_Disc_Auto",
    "Apparel Manufacturing": "Cons_Disc_Durables",
    "Apparel Retail": "Cons_Disc_Durables",
    "Footwear & Accessories": "Cons_Disc_Durables",
    "Furnishings Fixtures & Appliances": "Cons_Disc_Durables",
    "Luxury Goods": "Cons_Disc_Durables",
    "Leisure": "Cons_Disc_Services",
    "Restaurants": "Cons_Disc_Services",
    "Hotels & Motel Chains": "Cons_Disc_Services",
    "Personal Services": "Cons_Disc_Services",
    "Gambling": "Cons_Disc_Services",
    "Travel Services": "Cons_Disc_Services",
    "Resorts & Casinos": "Cons_Disc_Services",
    "Rental & Leasing Services": "Cons_Disc_Services",
    "Specialty Retail": "Cons_Disc_Distribution",
    "Department Stores": "Cons_Disc_Distribution",
    "Home Improvement Retail": "Cons_Disc_Distribution",
    "Online Retail": "Cons_Disc_Distribution",
    "Auto Dealerships": "Cons_Disc_Distribution",
    # 필수소비재
    "Food Distribution": "Cons_Staples_Food",
    "Packaged Foods": "Cons_Staples_Food",
    "Beverages-Non-Alcoholic": "Cons_Staples_Food",
    "Beverages-Alcoholic": "Cons_Staples_Food",
    "Tobacco": "Cons_Staples_Food",
    "Farm Products": "Cons_Staples_Food",
    "Household & Personal Products": "Cons_Staples_Personal",
    "Grocery Stores": "Cons_Staples_Retail",
    "Food & Drug Retail": "Cons_Staples_Retail",
    # 에너지
    "Oil & Gas E&P": "Energy",
    "Oil & Gas Integrated": "Energy",
    "Oil & Gas Midstream": "Energy",
    "Oil & Gas Refining & Marketing": "Energy",
    "Oil & Gas Equipment & Services": "Energy",
    "Oil & Gas Drilling": "Energy",
    "Oil & Gas Exploration & Production": "Energy",
    "Thermal Coal": "Energy",
    "Uranium": "Energy",
    "Renewable Energy": "Energy",
    "Solar": "Energy",
    "Energy Equipment & Services": "Energy",
    # 금융
    "Banks - Regional": "Fin_Banks",
    "Banks - Diversified": "Fin_Banks",
    "Banks - Global": "Fin_Banks",
    "Financial Services": "Fin_Financial_Services",
    "Capital Markets": "Fin_Financial_Services",
    "Asset Management": "Fin_Financial_Services",
    "Asset Management - Global": "Fin_Financial_Services",
    "Credit Services": "Fin_Financial_Services",
    "Financial - Mortgages": "Fin_Financial_Services",
    "Financial Conglomerates": "Fin_Financial_Services",
    "Financial - Conglomerates": "Fin_Financial_Services",
    "Financial - Capital Markets": "Fin_Financial_Services",
    "Financial - Credit Services": "Fin_Financial_Services",
    "Financial - Data & Stock Exchanges": "Fin_Financial_Services",
    "Diversified Financial Services": "Fin_Financial_Services",
    "Consumer Finance": "Fin_Financial_Services",
    "Mortgage Finance": "Fin_Financial_Services",
    "Insurance - Property & Casualty": "Fin_Insurance",
    "Insurance - Life": "Fin_Insurance",
    "Insurance - Diversified": "Fin_Insurance",
    "Insurance - Specialty": "Fin_Insurance",
    "Insurance - Reinsurance": "Fin_Insurance",
    # 헬스케어
    "Medical Devices": "HC_Equipment_Services",
    "Medical Instruments & Supplies": "HC_Equipment_Services",
    "Health Information Services": "HC_Equipment_Services",
    "Hospitals & Clinics": "HC_Equipment_Services",
    "Healthcare Plans": "HC_Equipment_Services",
    "Medical - Care Facilities": "HC_Equipment_Services",
    "Medical - Diagnostics & Research": "HC_Pharma_Bio",
    "Medical - Devices": "HC_Equipment_Services",
    "Medical - Healthcare Plans": "HC_Equipment_Services",
    "Medical - Healthcare Information Services": "HC_Equipment_Services",
    "Medical - Instruments & Supplies": "HC_Equipment_Services",
    "Medical - Distribution": "HC_Equipment_Services",
    "Health Care Equipment & Supplies": "HC_Equipment_Services",
    "Health Care Providers & Services": "HC_Equipment_Services",
    "Health Care Technology": "HC_Equipment_Services",
    "Health Care Facilities": "HC_Equipment_Services",
    "Drug Manufacturers - General": "HC_Pharma_Bio",
    "Drug Manufacturers - Specialty & Generic": "HC_Pharma_Bio",
    "Drug Manufacturers - Specialty": "HC_Pharma_Bio",
    "Biotechnology": "HC_Pharma_Bio",
    "Diagnostics & Research": "HC_Pharma_Bio",
    "Pharmaceutical Retailers": "HC_Pharma_Bio",
    "Life Sciences Tools & Services": "HC_Pharma_Bio",
    "Pharmaceuticals": "HC_Pharma_Bio",
    # 산업재
    "Aerospace & Defense": "Ind_Capital_Goods",
    "Industrial Machinery": "Ind_Capital_Goods",
    "Industrial - Machinery": "Ind_Capital_Goods",
    "Tools & Accessories": "Ind_Capital_Goods",
    "Electrical Equipment & Parts": "Ind_Capital_Goods",
    "Farm & Heavy Construction Machinery": "Ind_Capital_Goods",
    "Construction": "Ind_Capital_Goods",
    "Engineering & Construction": "Ind_Capital_Goods",
    "Infrastructure Operations": "Ind_Capital_Goods",
    "Building Products & Equipment": "Ind_Capital_Goods",
    "Specialty Industrial Machinery": "Ind_Capital_Goods",
    "Industrial - Distribution": "Ind_Commercial_Services",
    "Industrial - Pollution & Treatment Controls": "Ind_Commercial_Services",
    "Business Services": "Ind_Commercial_Services",
    "Staffing & Employment Services": "Ind_Commercial_Services",
    "Waste Management": "Ind_Commercial_Services",
    "Security & Protection Services": "Ind_Commercial_Services",
    "Consulting Services": "Ind_Commercial_Services",
    "Research & Consulting Services": "Ind_Commercial_Services",
    "Diversified Industrials": "Ind_Commercial_Services",
    "Education & Training Services": "Ind_Commercial_Services",
    "Airlines": "Ind_Transportation",
    "Airlines, Airports & Air Services": "Ind_Transportation",
    "Trucking": "Ind_Transportation",
    "Railroads": "Ind_Transportation",
    "Marine Shipping": "Ind_Transportation",
    "Integrated Freight & Logistics": "Ind_Transportation",
    "Airport Operations": "Ind_Transportation",
    "Shipping & Ports": "Ind_Transportation",
    "Ground Transportation": "Ind_Transportation",
    "Transportation Infrastructure": "Ind_Transportation",
    # 소재
    "Steel": "Materials",
    "Aluminum": "Materials",
    "Chemicals": "Materials",
    "Specialty Chemicals": "Materials",
    "Chemicals - Specialty": "Materials",
    "Gold": "Materials",
    "Silver": "Materials",
    "Building Materials": "Materials",
    "Paper & Paper Products": "Materials",
    "Copper": "Materials",
    "Other Precious Metals & Mining": "Materials",
    "Coking Coal": "Materials",
    "Lumber & Wood Production": "Materials",
    "Industrial Materials": "Materials",
    "Mining": "Materials",
    "Nonmetallic Mineral Mining": "Materials",
    "Metals & Mining": "Materials",
    "Diversified Metals & Mining": "Materials",
    "Construction Materials": "Materials",
    "Containers & Packaging": "Materials",
    "Paper & Forest Products": "Materials",
    # 부동산
    "Real Estate - General": "RE_Development",
    "Real Estate Services": "RE_Development",
    "Real Estate - Diversified": "RE_Development",
    "REIT - Retail": "RE_REIT",
    "REIT - Office": "RE_REIT",
    "REIT - Industrial": "RE_REIT",
    "REIT - Residential": "RE_REIT",
    "REIT - Healthcare Facilities": "RE_REIT",
    "REIT - Diversified": "RE_REIT",
    "REIT - Hotel & Motel": "RE_REIT",
    "REIT - Mortgage": "RE_REIT",
    "REIT - Specialty": "RE_REIT",
    # 유틸리티
    "Utilities - Regulated Electric": "Utilities",
    "Utilities - Regulated Gas": "Utilities",
    "Utilities - Diversified": "Utilities",
    "Utilities - Renewable": "Utilities",
    "Utilities - Independent Power Producers": "Utilities",
    "Utilities - Regulated Water": "Utilities",
    "Regulated Electric": "Utilities",
    "Regulated Gas": "Utilities",
    "Electric Utilities": "Utilities",
    "Gas Utilities": "Utilities",
    "Multi-Utilities": "Utilities",
    "Water Utilities": "Utilities",
    "Independent Power and Renewable Electricity Producers": "Utilities",
    # 필수소비재 추가
    "Beverages - Non-Alcoholic": "Cons_Staples_Food",
    "Beverages - Alcoholic": "Cons_Staples_Food",
    "Beverages - Wineries & Distilleries": "Cons_Staples_Food",
    "Confectioners": "Cons_Staples_Food",
    "Discount Stores": "Cons_Staples_Retail",
    "Drug Stores": "Cons_Staples_Retail",
    # 임의소비재 추가
    "Home Furnishings & Fixtures": "Cons_Disc_Durables",
    "Textiles": "Cons_Disc_Durables",
    "Textile Manufacturing": "Cons_Disc_Durables",
    "Residential Construction": "Cons_Disc_Durables",
    "Lodging": "Cons_Disc_Services",
    "Amusement Parks": "Cons_Disc_Services",
    "Entertainment - Diversified": "Cons_Disc_Services",
    "Internet Retail": "Cons_Disc_Distribution",
    "Catalog & Mail Order Houses": "Cons_Disc_Distribution",
    # ── FMP 독자 네이밍 추가 (Auto - , Apparel - 등 하이픈 패턴) ──
    "Auto - Parts": "Cons_Disc_Auto",
    "Auto - Dealerships": "Cons_Disc_Auto",
    "Auto - Manufacturers": "Cons_Disc_Auto",
    "Auto - Recreational Vehicles": "Cons_Disc_Auto",
    "Apparel - Retail": "Cons_Disc_Durables",
    "Apparel - Manufacturers": "Cons_Disc_Durables",
    "Apparel - Footwear & Accessories": "Cons_Disc_Durables",
    "Furnishings, Fixtures & Appliances": "Cons_Disc_Durables",
    "Home Improvement": "Cons_Disc_Distribution",
    "Gambling, Resorts & Casinos": "Cons_Disc_Services",
    "Travel Lodging": "Cons_Disc_Services",
    # 필수소비재 FMP 패턴
    "Agricultural Inputs": "Cons_Staples_Food",
    "Agricultural Farm Products": "Cons_Staples_Food",
    "Food Confectioners": "Cons_Staples_Food",
    "Personal Products & Services": "Cons_Staples_Personal",
    # 에너지 FMP 패턴
    "Oil & Gas Energy": "Energy",
    "Coal": "Energy",
    "Renewable Utilities": "Utilities",
    "Regulated Water": "Utilities",
    "Diversified Utilities": "Utilities",
    "Independent Power Producers": "Utilities",
    "General Utilities": "Utilities",
    # 금융 FMP 패턴
    "Insurance - Brokers": "Fin_Insurance",
    "Investment - Banking & Investment Services": "Fin_Financial_Services",
    "Asset Management - Income": "Fin_Financial_Services",
    "Asset Management - Cryptocurrency": "Fin_Financial_Services",
    "Financial - Diversified": "Fin_Financial_Services",
    "Banks": "Fin_Banks",
    # 헬스케어 FMP 패턴
    "Medical - Pharmaceuticals": "HC_Pharma_Bio",
    "Medical - Equipment & Services": "HC_Equipment_Services",
    "Medical - Specialties": "HC_Equipment_Services",
    # 산업재 FMP 패턴
    "Specialty Business Services": "Ind_Commercial_Services",
    "Business Equipment & Supplies": "Ind_Commercial_Services",
    "Industrial - Infrastructure Operations": "Ind_Capital_Goods",
    "Agricultural - Machinery": "Ind_Capital_Goods",
    "Manufacturing - Metal Fabrication": "Ind_Capital_Goods",
    "Manufacturing - Tools & Accessories": "Ind_Capital_Goods",
    "Technology Distributors": "IT_Software_Services",
    "Software - Services": "IT_Software_Services",
    "General Transportation": "Ind_Transportation",
    # 소재 FMP 패턴
    "Packaging & Containers": "Materials",
    "Paper, Lumber & Forest Products": "Materials",
    "Other Precious Metals": "Materials",
    "Manufacturing - Textiles": "Materials",
    # 부동산 FMP 패턴
    "Real Estate - Services": "RE_Development",
    "Real Estate - Development": "RE_Development",
    # 복합/기타 FMP 패턴
    "Conglomerates": "Ind_Commercial_Services",
}

ALL_INDUSTRIES = sorted(set(GICS_MAP.values()))


# ──────────────────────────────────────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────────────────────────────────────

def _to_date(val) -> Optional[pd.Timestamp]:
    """Unix ms timestamp 또는 문자열을 pd.Timestamp로 변환."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        return pd.Timestamp(int(val), unit="ms")
    s = str(val).strip()
    if not s or s == "nan":
        return None
    return pd.Timestamp(s[:10])  # "YYYY-MM-DD HH:MM:SS" → 날짜만


def _sector_zscore(series: pd.Series, sector_map: pd.Series) -> pd.Series:
    """
    섹터별 Z-score 계산.
    - 섹터 정보가 있는 종목: 섹터 내 Z-score
    - 섹터 정보가 없는 종목(unknown): NaN 유지
    """
    result = pd.Series(np.nan, index=series.index)

    aligned_sector = sector_map.reindex(series.index).fillna("__unknown__")
    for sector, grp_idx in aligned_sector.groupby(aligned_sector).groups.items():
        if sector == "__unknown__":
            continue  # 섹터 불명 종목은 비교 대상 없음 → NaN 유지
        grp = series.loc[grp_idx].dropna()
        if grp.empty:
            continue
        mu = grp.mean()
        sd = grp.std()
        if pd.notna(sd) and sd > 0:
            result.loc[grp.index] = (series.loc[grp.index] - mu) / sd
        # sd = 0 또는 NaN: 섹터 내 값이 모두 동일 → 비교 불가 → NaN 유지

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로더
# ──────────────────────────────────────────────────────────────────────────────

class _DataLoader:
    """사전 로드된 데이터를 보관하는 data holder (IO 없음).

    DB에서 벌크 로드한 DataFrame들을 받아 저장하고,
    팩터 계산 함수들이 기존 인터페이스로 접근할 수 있도록 한다.
    """

    def __init__(
        self,
        *,
        price_df: Optional[pd.DataFrame] = None,
        mcap_df: Optional[pd.DataFrame] = None,
        fin_df: Optional[pd.DataFrame] = None,
        profile: Optional[dict] = None,
        treasury_df: Optional[pd.DataFrame] = None,
    ):
        self._price_df = price_df
        self._mcap_df = mcap_df
        self._fin_df = fin_df
        self._profile = profile or {}
        self._treasury_df = treasury_df

    def load_profile(self, ticker: str) -> dict:
        return self._profile

    def load_treasury(self) -> Optional[pd.DataFrame]:
        return self._treasury_df

    def load_price(self, ticker: str) -> Optional[pd.DataFrame]:
        return self._price_df

    def load_mcap(self, ticker: str) -> Optional[pd.DataFrame]:
        return self._mcap_df

    def load_financial(self, ticker: str) -> Optional[pd.DataFrame]:
        return self._fin_df


def _prepare_financial_df(fin_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """DB에서 로드한 financial DataFrame을 팩터 계산 호환 형식으로 변환.

    변환 내용:
    1. snake_case → camelCase 컬럼 rename (FINANCIAL_COLUMN_MAP_REVERSE)
    2. filing_date → filingDate rename
    3. period_end 생성 (date 컬럼)
    4. filing_date_pit 생성 (filingDate, 없으면 period_end + 90일)
    5. period_end 내림차순 정렬
    """
    if fin_df is None or fin_df.empty:
        return None

    df = fin_df.rename(columns=FINANCIAL_COLUMN_MAP_REVERSE)
    df = df.rename(columns=_FINANCIAL_META_RENAME)

    df["period_end"] = pd.to_datetime(df["date"])

    df["filing_date_pit"] = df["filingDate"].apply(_to_date)
    no_filing = df["filing_date_pit"].isna()
    df.loc[no_filing, "filing_date_pit"] = (
        df.loc[no_filing, "period_end"] + pd.Timedelta(days=90)
    )

    df = df.sort_values("period_end", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PIT 재무제표 조회 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _get_pit_financials(
    fin_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    n_quarters: int = 1,
    _cache: Optional[Dict] = None,
) -> Optional[pd.DataFrame]:
    """
    PIT 기준으로 as_of_date 이전에 공시된 분기 재무제표를 반환.

    as_of_date >= filing_date_pit 인 분기만 유효하며,
    최신 n_quarters 개 분기를 내림차순으로 반환한다.

    엄격한 검증:
    1. as_of_date 기준 가장 최신이어야 할 분기(expected_latest_quarter)가
       실제 데이터의 최신 분기와 일치하지 않으면 None 반환.
       - expected_latest_quarter: as_of_date 이전에 끝났을 가장 최근 분기 말일
         (1Q=3/31, 2Q=6/30, 3Q=9/30, 4Q=12/31)
    2. n_quarters > 1일 때 연속 분기 사이에 빠진 분기가 있으면 None 반환.
       - 연속 분기 간 period_end 간격이 60~120일 범위를 벗어나면 gap으로 판단.

    Parameters
    ----------
    _cache : {(as_of_date, n_quarters): result} 딕셔너리.
             전달하면 동일 (as_of_date, n_quarters) 조합의 결과를 재사용.
             None 이면 캐싱 없이 즉시 계산.
    """
    if _cache is not None:
        key = (as_of_date, n_quarters)
        if key in _cache:
            return _cache[key]

    # fin_df는 이미 period_end 내림차순 정렬된 상태로 들어온다고 가정
    # (load_financial에서 정렬 후 저장, _calc_raw_for_ticker에서 pre-filter 후 전달)
    valid = fin_df[fin_df["filing_date_pit"] <= as_of_date]
    if valid.empty:
        if _cache is not None:
            _cache[(as_of_date, n_quarters)] = None
        return None

    # ── 검증 1: 최신 분기 존재 여부 ──────────────────────────────────────────
    # 재무제표는 분기 종료 후 45~90일 뒤에 공시됨.
    # 따라서 as_of_date 기준으로 "공시 가능했을 최신 분기"는
    # 직전 분기(~1분기 전)이다.
    # 예: as_of_date=6/30 → Q2 보고서는 아직 미공시 → Q1(3/31)이 최신 기대값
    year = as_of_date.year
    quarter_ends = [
        pd.Timestamp(year - 1, 3, 31),
        pd.Timestamp(year - 1, 6, 30),
        pd.Timestamp(year - 1, 9, 30),
        pd.Timestamp(year - 1, 12, 31),
        pd.Timestamp(year, 3, 31),
        pd.Timestamp(year, 6, 30),
        pd.Timestamp(year, 9, 30),
        pd.Timestamp(year, 12, 31),
    ]
    # 공시 lag 고려: as_of_date 기준 최소 45일 전에 종료된 분기가 기대 최신
    candidate_ends = [d for d in quarter_ends if d <= as_of_date - pd.Timedelta(days=45)]
    if not candidate_ends:
        candidate_ends = [pd.Timestamp(year - 1, 12, 31)]
    expected_latest = max(candidate_ends)

    actual_latest = valid["period_end"].iloc[0]
    if abs((actual_latest - expected_latest).days) > 20:
        if _cache is not None:
            _cache[(as_of_date, n_quarters)] = None
        return None

    # ── 검증 2: 연속성 체크 (n_quarters > 1) ──────────────────────────────────
    result = valid.head(n_quarters)
    if n_quarters > 1 and len(result) == n_quarters:
        ends = result["period_end"].values
        for i in range(len(ends) - 1):
            gap = (pd.Timestamp(ends[i]) - pd.Timestamp(ends[i + 1])).days
            if not (60 <= gap <= 120):
                if _cache is not None:
                    _cache[(as_of_date, n_quarters)] = None
                return None

    if len(result) < n_quarters:
        if _cache is not None:
            _cache[(as_of_date, n_quarters)] = None
        return None

    if _cache is not None:
        _cache[(as_of_date, n_quarters)] = result
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 팩터별 계산 함수
# ──────────────────────────────────────────────────────────────────────────────

def _get_mcap_on_date(
    ticker: str,
    loader: _DataLoader,
    as_of_date: pd.Timestamp,
) -> Optional[float]:
    """
    as_of_date 이하 가장 가까운 날짜의 시가총액 반환.
    최대 7일 이내 데이터가 없으면 None.
    """
    mcap_df = loader.load_mcap(ticker)
    if mcap_df is None:
        return None

    subset = mcap_df[mcap_df["date"] <= as_of_date]
    if subset.empty:
        return None

    latest_date = subset["date"].max()
    if (as_of_date - latest_date).days > 7:
        return None

    val = subset.loc[subset["date"] == latest_date, "marketCap"].iloc[0]
    return float(val) if pd.notna(val) and val > 0 else None


def _calc_value(
    ticker: str,
    loader: _DataLoader,
    as_of_date: pd.Timestamp,
    mcap: Optional[float],
    _cache: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    가치 팩터
    - B/P: 최신 1분기 totalStockholdersEquity / mcap(T-1)
    - E/P: 최근 4분기 netIncome 합 / mcap(T-1)
    """
    if mcap is None or mcap <= 0:
        return {"value_bp": np.nan, "value_ep": np.nan}

    fin = loader.load_financial(ticker)
    if fin is None:
        return {"value_bp": np.nan, "value_ep": np.nan}

    latest = _get_pit_financials(fin, as_of_date, n_quarters=1, _cache=_cache)
    if latest is None:
        return {"value_bp": np.nan, "value_ep": np.nan}

    equity = latest["totalStockholdersEquity"].iloc[0]
    bp = float(equity) / mcap if pd.notna(equity) else np.nan

    ttm = _get_pit_financials(fin, as_of_date, n_quarters=4, _cache=_cache)
    if ttm is None or "netIncome" not in ttm.columns:
        ep = np.nan
    else:
        ep = float(ttm["netIncome"].sum()) / mcap

    return {"value_bp": bp, "value_ep": ep}


def _calc_size(mcap: Optional[float]) -> float:
    """사이즈 팩터: log(시가총액)."""
    if mcap is None or mcap <= 0:
        return np.nan
    return np.log(mcap)


def _calc_profitability(
    ticker: str,
    loader: _DataLoader,
    as_of_date: pd.Timestamp,
    _cache: Optional[Dict] = None,
) -> float:
    """
    수익성 팩터 (Operating Profitability):
    - 분자: 최근 4분기 (revenue - costOfRevenue - SGA) 합 = 영업이익 TTM
    - 분모: 최신 1분기 totalAssets
    """
    fin = loader.load_financial(ticker)
    if fin is None:
        return np.nan

    latest = _get_pit_financials(fin, as_of_date, n_quarters=1, _cache=_cache)
    if latest is None:
        return np.nan

    total_assets = latest["totalAssets"].iloc[0]
    if pd.isna(total_assets) or total_assets <= 0:
        return np.nan

    ttm = _get_pit_financials(fin, as_of_date, n_quarters=4, _cache=_cache)
    if ttm is None:
        return np.nan

    sga = ttm["sellingGeneralAndAdministrativeExpenses"].fillna(0)
    op_profit = (ttm["revenue"] - ttm["costOfRevenue"] - sga).sum()
    return float(op_profit) / float(total_assets)


def _calc_investment(
    ticker: str,
    loader: _DataLoader,
    as_of_date: pd.Timestamp,
    _cache: Optional[Dict] = None,
) -> float:
    """
    투자 팩터 (YoY Asset Growth):
    - 최신 분기 totalAssets vs 4분기 전(1년 전 동일 분기) totalAssets 비교
    - (assets_now - assets_1y_ago) / assets_1y_ago
    """
    fin = loader.load_financial(ticker)
    if fin is None:
        return np.nan

    # 최신 5개 분기 필요 (현재 + 4분기 전)
    recent5 = _get_pit_financials(fin, as_of_date, n_quarters=5, _cache=_cache)
    if recent5 is None or len(recent5) < 5:
        return np.nan

    assets_now  = recent5["totalAssets"].iloc[0]
    assets_prev = recent5["totalAssets"].iloc[4]

    if pd.isna(assets_now) or pd.isna(assets_prev) or assets_prev <= 0:
        return np.nan

    return (float(assets_now) - float(assets_prev)) / float(assets_prev)


def _calc_momentum(
    ticker: str,
    loader: _DataLoader,
    as_of_date: pd.Timestamp,
) -> float:
    """
    모멘텀 팩터: P_{t-1} / P_{t-12} - 1
    - t-1:  as_of_date 기준 정확히 21 거래일 전 종가 (1개월 전)
    - t-12: as_of_date 기준 정확히 252 거래일 전 종가 (12개월 전)

    미래편향 방지:
    - as_of_date 이하 날짜만 사용
    - 253행(= 252 거래일 + T일) 미만이면 NaN
    """
    price = loader.load_price(ticker)
    if price is None:
        return np.nan

    hist = price[price["date"] <= as_of_date].sort_values("date").tail(253)
    if len(hist) < 253:  # 252 거래일 전 가격을 구하려면 최소 253행 필요
        return np.nan

    # tail(253)이므로:
    # iloc[-1]  = T일 (as_of_date)
    # iloc[-22] = 정확히 21 거래일 전 (t-1)
    # iloc[0]   = 정확히 252 거래일 전 (t-12)
    p_t1  = hist["adj_close"].iloc[-22]
    p_t12 = hist["adj_close"].iloc[0]

    if pd.isna(p_t1) or pd.isna(p_t12) or p_t12 <= 0:
        return np.nan

    return float(p_t1) / float(p_t12) - 1


def _calc_low_vol(
    ticker: str,
    loader: _DataLoader,
    as_of_date: pd.Timestamp,
    window: int = 252,
) -> float:
    """
    저변동성 팩터: 과거 window(252) 거래일 일간 수익률의 표준편차.
    252 거래일 데이터가 없으면 NaN.
    """
    price = loader.load_price(ticker)
    if price is None:
        return np.nan

    # tail(window + 1): 가격 253행 → pct_change → 수익률 정확히 252개
    hist = price[price["date"] <= as_of_date].sort_values("date").tail(window + 1)
    ret = hist["adj_close"].pct_change().dropna()

    if len(ret) < window:  # 252 거래일 미만이면 NaN
        return np.nan

    return float(ret.std())


def _calc_stability(
    ticker: str,
    loader: _DataLoader,
    as_of_date: pd.Timestamp,
    _cache: Optional[Dict] = None,
) -> float:
    """
    안정성 팩터 (Leverage): 총부채 / 총자산.
    최신 1분기 PIT 재무제표 기준.
    """
    fin = loader.load_financial(ticker)
    if fin is None:
        return np.nan

    latest = _get_pit_financials(fin, as_of_date, n_quarters=1, _cache=_cache)
    if latest is None:
        return np.nan

    total_liabilities = latest["totalLiabilities"].iloc[0]
    total_assets      = latest["totalAssets"].iloc[0]

    if pd.isna(total_liabilities) or pd.isna(total_assets) or total_assets <= 0:
        return np.nan

    return float(total_liabilities) / float(total_assets)


def _get_risk_free_rate(
    loader: _DataLoader,
    as_of_date: pd.Timestamp,
    lookback_days: int = 21,
) -> float:
    """
    무위험 수익률: Treasury month3(3개월물) 연이율을 기간 수익률로 변환.
    - as_of_date 이하 가장 가까운 날짜의 금리 사용 (최대 7일 이내)
    - 연이율(%) → 기간 수익률: (1 + r/100)^(days/365) - 1
    - 데이터 없거나 오류 시 NaN 반환.
    """
    treasury = loader.load_treasury()
    if treasury is None or treasury.empty:
        return np.nan

    subset = treasury[treasury["date"] <= as_of_date]
    if subset.empty:
        return np.nan

    latest_date = subset["date"].max()
    if (as_of_date - latest_date).days > 7:
        return np.nan

    rate_col = "month3" if "month3" in subset.columns else None
    if rate_col is None:
        return np.nan

    annual_rate = subset.loc[subset["date"] == latest_date, rate_col].iloc[0]
    if pd.isna(annual_rate):
        return np.nan

    # 연이율(%) → lookback_days 기간 수익률
    return (1 + float(annual_rate) / 100) ** (lookback_days / 365) - 1


def _calc_market_factor(
    loader: _DataLoader,
    as_of_date: pd.Timestamp,
    market_proxy: str = "SPY",
    lookback_days: int = 21,
) -> float:
    """
    시장 팩터: SPY 과거 ~1개월(21 거래일) 수익률 - 무위험 수익률.
    - 무위험 수익률: Treasury month3 연이율 → 기간 환산 (없으면 NaN).
    - SPY 또는 Treasury 데이터가 없으면 NaN.
    """
    price = loader.load_price(market_proxy)
    if price is None:
        return np.nan

    hist = price[price["date"] <= as_of_date].sort_values("date")
    if len(hist) < lookback_days + 1:  # T일 포함 최소 22행 필요
        return np.nan

    # iloc[-1]              = T일 종가
    # iloc[-lookback_days-1] = iloc[-22] = 정확히 21 거래일 전 종가
    p_end   = hist["adj_close"].iloc[-1]
    p_start = hist["adj_close"].iloc[-lookback_days - 1]

    if pd.isna(p_end) or pd.isna(p_start) or p_start <= 0:
        return np.nan

    market_ret = float(p_end) / float(p_start) - 1
    risk_free  = _get_risk_free_rate(loader, as_of_date, lookback_days)
    if pd.isna(risk_free):
        return np.nan

    return market_ret - risk_free


# ──────────────────────────────────────────────────────────────────────────────
# 병렬 처리용 Worker 함수 (module-level: multiprocessing pickle 가능)
# ──────────────────────────────────────────────────────────────────────────────

def _calc_raw_for_ticker(
    args: Tuple[str, List[pd.Timestamp], dict, Dict[pd.Timestamp, float]],
) -> List[Dict]:
    """
    단일 ticker에 대해 여러 날짜의 raw 팩터를 계산한다.

    multiprocessing.Pool.map에서 호출되므로 module-level 함수여야 함.
    메인 프로세스에서 DB 벌크 로드한 데이터를 dict로 받아 _DataLoader를 구성.

    Parameters
    ----------
    args : (ticker, dates, loader_data, market_ret_map)
        ticker         : 종목 심볼
        dates          : 계산할 날짜 목록 (pd.Timestamp)
        loader_data    : _DataLoader 생성용 dict (price_df, mcap_df, fin_df, profile, treasury_df)
        market_ret_map : {date: market_excess_return} (메인 프로세스에서 미리 계산)

    Returns
    -------
    List[Dict] : 날짜별 raw 팩터 딕셔너리 목록
    """
    ticker, dates, loader_data, market_ret_map = args
    loader = _DataLoader(**loader_data)

    profile  = loader.load_profile(ticker)
    industry = profile.get("industry", "")
    sector   = profile.get("sector",   "")
    gics_l2  = GICS_MAP.get(industry, "Other")

    rows = []
    for t in dates:
        # ── Option A: 날짜별 pit_cache 생성 ──────────────────────────────
        # 같은 날짜에서 _calc_value(n=1, n=4) + _calc_profitability(n=1, n=4) +
        # _calc_investment(n=5) + _calc_stability(n=1) 총 6회 호출 →
        # 실제 고유 (n_quarters) 조합은 1, 4, 5 뿐이므로 최대 3번 계산 후 재사용.
        pit_cache: Dict = {}

        t_prev = t - pd.Timedelta(days=1)
        mcap   = _get_mcap_on_date(ticker, loader, t_prev)

        val = _calc_value(ticker, loader, t, mcap, _cache=pit_cache)
        row: Dict = {
            "symbol":               ticker,
            "rebalance_date":       t,
            "raw_industry":         industry,
            "raw_sector":           sector,
            "gics_l2":              gics_l2,
            "mcap":                 mcap,
            "market_excess_return": market_ret_map.get(t, np.nan),
            "value_bp":             val["value_bp"],
            "value_ep":             val["value_ep"],
            "size":                 _calc_size(mcap),
            "profitability":        _calc_profitability(ticker, loader, t, _cache=pit_cache),
            "investment":           _calc_investment(ticker, loader, t, _cache=pit_cache),
            "momentum":             _calc_momentum(ticker, loader, t),
            "low_vol":              _calc_low_vol(ticker, loader, t),
            "stability":            _calc_stability(ticker, loader, t, _cache=pit_cache),
        }
        rows.append(row)

    return rows


def _apply_cross_sectional(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    날짜별 cross-sectional 정규화(Z-score)와 산업 더미 변수를 적용한다.

    Phase 2: _calc_raw_for_ticker로 수집된 raw 팩터 DataFrame에
    섹터별 Z-score, 모멘텀 전체 Z-score, 산업 더미를 추가한다.

    Parameters
    ----------
    df : raw 팩터 DataFrame (index=symbol, rebalance_date 컬럼 포함)

    Returns
    -------
    pd.DataFrame : Z-score와 산업 더미가 추가된 DataFrame
    """
    style_sector_z = [
        "value_bp", "value_ep", "size",
        "profitability", "investment", "low_vol", "stability",
    ]

    result_chunks = []
    for rdate, grp in df.groupby("rebalance_date"):
        grp = grp.copy()
        sector_map = grp["raw_sector"]

        # 섹터별 Z-score
        for col in style_sector_z:
            grp[f"{col}_z"] = _sector_zscore(grp[col], sector_map)

        # 모멘텀 전체 Z-score
        mu_mom = grp["momentum"].mean()
        sd_mom = grp["momentum"].std()
        if pd.notna(sd_mom) and sd_mom > 0:
            grp["momentum_z"] = (grp["momentum"] - mu_mom) / sd_mom
        # sd = 0 또는 NaN → momentum_z 컬럼 없음 → 아래에서 NaN으로 채움

        # 산업 더미 변수
        for ind in ALL_INDUSTRIES:
            grp[f"industry_{ind}"] = (grp["gics_l2"] == ind).astype(int)

        result_chunks.append(grp)

    out = pd.concat(result_chunks, ignore_index=True)

    # momentum_z가 일부 날짜에서 생성 안 됐을 경우 NaN 컬럼으로 통일
    if "momentum_z" not in out.columns:
        out["momentum_z"] = np.nan

    return out


# ──────────────────────────────────────────────────────────────────────────────
# 메인 팩터 생성 클래스
# ──────────────────────────────────────────────────────────────────────────────

class FeatureCreator:
    """
    리밸런싱 날짜(T) 기준으로 시장/산업/스타일 팩터를 계산한다.

    Parameters
    ----------
    data_dir : str
        FMP 데이터 루트 폴더 (기본: "/app/data/fmp")
    market_proxy : str
        시장 수익률 proxy 종목 심볼 (기본: "SPY")
    n_workers : int
        병렬 처리 Worker 수. 0 또는 1이면 단일 프로세스로 실행.
        기본값 -1은 os.cpu_count()를 자동 사용.
    """

    def __init__(
        self,
        data_dir:     str = "/app/data/fmp",
        db_path:      str = "/app/data/database/fmp_stocks.db",
        market_proxy: str = "SPY",
        n_workers:    int = -1,
    ):
        self.data_dir     = data_dir
        self.db_path      = db_path
        self.market_proxy = market_proxy
        self.n_workers    = os.cpu_count() if n_workers < 1 else n_workers

    # ── 내부: DB에서 전체 데이터를 벌크 로드 ────────────────────────────────
    def _bulk_load_all_data(
        self,
        ticker_list: List[str],
        dates: List[pd.Timestamp],
    ) -> Dict[str, dict]:
        """DB에서 전체 ticker 데이터를 일괄 로드하여 ticker별 dict로 반환.

        Returns
        -------
        Dict[str, dict]
            {ticker: {"price_df": ..., "mcap_df": ..., "fin_df": ...,
                       "profile": ..., "treasury_df": ...}}
        """
        # 날짜 범위: 모멘텀/변동성 lookback (253 거래일 ≈ 400 캘린더일) + 여유
        start_date = (min(dates) - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
        end_date = max(dates).strftime("%Y-%m-%d")

        all_symbols = sorted(set(ticker_list) | {self.market_proxy})

        print(f"  DB 벌크 로드 시작: {len(all_symbols)}개 심볼, "
              f"기간 {start_date} ~ {end_date}", flush=True)

        with FmpDatabaseManager(db_path=self.db_path) as db:
            prices_dict = db.load_prices_bulk(all_symbols, start_date, end_date)
            mcaps_dict = db.load_market_caps_bulk(all_symbols, start_date, end_date)
            fins_dict = db.load_financials_bulk(all_symbols)
            treasury_df = db.load_treasury_rates(start_date, end_date)
            metadata_df = db.load_ticker_metadata(all_symbols)

        # Treasury 컬럼 rename
        treasury_df = treasury_df.rename(columns=_TREASURY_DB_RENAME)

        # ticker별 데이터 dict 구성
        ticker_data_map: Dict[str, dict] = {}
        for sym in all_symbols:
            # Price: close → adj_close
            price_df = prices_dict.get(sym)
            if price_df is not None and not price_df.empty:
                price_df = price_df.rename(columns=_PRICE_DB_RENAME)
            else:
                price_df = None

            # Market cap: market_cap → marketCap
            mcap_df = mcaps_dict.get(sym)
            if mcap_df is not None and not mcap_df.empty:
                mcap_df = mcap_df.rename(columns=_MCAP_DB_RENAME)
            else:
                mcap_df = None

            # Financials: snake_case → camelCase + period_end/filing_date_pit 생성
            fin_df = _prepare_financial_df(fins_dict.get(sym))

            # Profile
            profile = {}
            if sym in metadata_df.index:
                row = metadata_df.loc[sym]
                profile = {
                    "industry": row.get("industry") or "",
                    "sector":   row.get("sector")   or "",
                }

            ticker_data_map[sym] = {
                "price_df":    price_df,
                "mcap_df":     mcap_df,
                "fin_df":      fin_df,
                "profile":     profile,
                "treasury_df": treasury_df,
            }

        print(f"  DB 벌크 로드 완료: price={len(prices_dict)}, "
              f"mcap={len(mcaps_dict)}, fin={len(fins_dict)}, "
              f"treasury={len(treasury_df)}행", flush=True)

        return ticker_data_map

    def _build_market_ret_map(
        self,
        dates: List[pd.Timestamp],
        ticker_data_map: Dict[str, dict],
    ) -> Dict[pd.Timestamp, float]:
        """SPY market_excess_return을 날짜별로 미리 계산해 dict로 반환."""
        spy_loader = _DataLoader(**ticker_data_map[self.market_proxy])
        return {t: _calc_market_factor(spy_loader, t, self.market_proxy) for t in dates}

    # ── 단일 날짜 계산 (외부 인터페이스 유지) ────────────────────────────────
    def compute(
        self,
        ticker_list:    List[str],
        rebalance_date: str,
    ) -> pd.DataFrame:
        """
        rebalance_date(T) 기준으로 팩터를 계산한다.

        미래편향 방지:
        - 시가총액: T-1 (전일) 기준
        - 재무제표: T 이전에 공시(filingDate)된 가장 최신 분기 기준 (PIT)
        - 주가/변동성: T 이하 날짜만 사용

        Parameters
        ----------
        ticker_list    : 종목 심볼 리스트
        rebalance_date : 리밸런싱 날짜 "YYYY-MM-DD"

        Returns
        -------
        pd.DataFrame
            index = symbol
            columns =
              raw 팩터: value_bp, value_ep, size, profitability,
                        investment, momentum, low_vol, stability
              Z-score:  value_bp_z, value_ep_z, size_z, profitability_z,
                        investment_z, momentum_z, low_vol_z, stability_z
              시장:     market_excess_return
              산업더미:  industry_{카테고리명} ...
              메타:     mcap, raw_industry, raw_sector, gics_l2
        """
        result = self.compute_multi_dates(ticker_list, [rebalance_date])
        if result.empty:
            return pd.DataFrame()
        return result.drop(columns=["rebalance_date"]).set_index("symbol")

    # ── 다중 날짜 계산 (2-Phase 병렬) ────────────────────────────────────────
    def compute_multi_dates(
        self,
        ticker_list:     List[str],
        rebalance_dates: List[str],
    ) -> pd.DataFrame:
        """
        여러 리밸런싱 날짜에 대해 팩터를 계산하고 long-format으로 반환.

        Phase 1 (병렬): ticker별 Worker가 모든 날짜의 raw 팩터를 계산.
                        파일 IO는 ticker당 1번으로 유지.
        Phase 2 (순차): 날짜별 cross-sectional Z-score와 산업 더미 적용.

        Parameters
        ----------
        ticker_list     : 종목 심볼 리스트
        rebalance_dates : 리밸런싱 날짜 리스트 ["YYYY-MM-DD", ...]

        Returns
        -------
        pd.DataFrame
            columns: rebalance_date, symbol, 팩터들 ...
        """
        dates = [pd.Timestamp(d) for d in rebalance_dates]

        # ── Phase 0: DB에서 전체 데이터 벌크 로드 ─────────────────────────────
        ticker_data_map = self._bulk_load_all_data(ticker_list, dates)

        # ── Phase 1-0: 시장 팩터 (SPY) 날짜별 사전 계산 ─────────────────────
        # SPY는 모든 종목 공통값 → 메인 프로세스에서 1번만 계산해 Worker에 전달
        market_ret_map = self._build_market_ret_map(dates, ticker_data_map)

        # ── Phase 1: ticker별 병렬 raw 팩터 계산 ─────────────────────────────
        worker_args = [
            (ticker, dates, ticker_data_map[ticker], market_ret_map)
            for ticker in ticker_list
            if ticker in ticker_data_map
        ]

        n_workers = min(self.n_workers, len(ticker_list))

        n_total = len(worker_args)
        report_every = max(1, n_total // 10)   # 10% 단위로 출력

        if n_workers <= 1:
            # 단일 프로세스 (디버깅 / 종목 수 적을 때)
            all_rows: List[Dict] = []
            for i, args in enumerate(worker_args, 1):
                all_rows.extend(_calc_raw_for_ticker(args))
                if i % report_every == 0 or i == n_total:
                    print(f"  Phase1 {i}/{n_total} tickers done", flush=True)
        else:
            # 멀티프로세스: imap_unordered로 완료된 ticker부터 즉시 수집
            all_rows = []
            with mp.Pool(processes=n_workers) as pool:
                for i, rows in enumerate(
                    pool.imap_unordered(_calc_raw_for_ticker, worker_args), 1
                ):
                    all_rows.extend(rows)
                    if i % report_every == 0 or i == n_total:
                        print(f"  Phase1 {i}/{n_total} tickers done", flush=True)

        if not all_rows:
            return pd.DataFrame()

        raw_df = pd.DataFrame(all_rows)

        # ── Phase 2: cross-sectional 정규화 (날짜별 순차) ────────────────────
        final_df = _apply_cross_sectional(raw_df)

        # 컬럼 순서 정리: rebalance_date, symbol 앞으로
        front = ["rebalance_date", "symbol"]
        other = [c for c in final_df.columns if c not in front]
        final_df = final_df[front + other].reset_index(drop=True)

        n_dates   = len(dates)
        n_tickers = len(ticker_list)
        print(f"✅ 완료: {n_tickers}개 종목 × {n_dates}개 날짜 = {len(final_df)}행 "
              f"(workers={n_workers})")

        return final_df
