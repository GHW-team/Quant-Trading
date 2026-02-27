"""
fmp_db_models.py
================
FMP(Financial Modeling Prep) 데이터 전용 SQLAlchemy ORM 모델.

기존 yfinance용 db_models.py(Base, stocks.db)와 완전 분리.
별도의 FmpBase → fmp_stocks.db 전용.

테이블 6개:
  fmp_tickers            종목 메타데이터 (company_profile)
  fmp_daily_prices       일봉 시세 (price)
  fmp_market_caps        시가총액 (market_cap)
  fmp_financials         분기별 재무제표 (all_financial)
  fmp_treasury_rates     미국 국채 금리 (treasury)
  fmp_universe_calendar  거래일별 상장 종목 (universe_calendar)
"""

import re
from datetime import datetime, timezone, date as date_type
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, Float, Date, DateTime, Boolean,
    ForeignKey, UniqueConstraint, Index, create_engine,
)
from sqlalchemy.orm import declarative_base, relationship

FmpBase = declarative_base()


def _utc_now():
    return datetime.now(timezone.utc)


def convert_fmp_date(val) -> Optional[date_type]:
    """FMP JSON의 date 필드를 Python date 객체로 변환."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return datetime.utcfromtimestamp(val / 1000).date()
    if isinstance(val, str) and len(val) >= 10:
        return datetime.strptime(val[:10], "%Y-%m-%d").date()
    return None


def parse_symbol_from_filename(filename: str) -> str:
    """파일명에서 심볼 추출. 예: 'BRK-A_1985-01-01_to_2026-02-25.json' → 'BRK-A'"""
    match = re.match(r"^(.+?)_\d{4}-\d{2}-\d{2}_to_", filename)
    if match:
        return match.group(1)
    return filename.replace(".json", "")


# ──────────────────────────────────────────────────────────────────────────────
# camelCase → snake_case 매핑 (financial JSON 수치 컬럼 119개)
# ──────────────────────────────────────────────────────────────────────────────
FINANCIAL_COLUMN_MAP = {
    "accountPayables": "account_payables",
    "accountsPayables": "accounts_payables",
    "accountsReceivables": "accounts_receivables",
    "accruedExpenses": "accrued_expenses",
    "accumulatedOtherComprehensiveIncomeLoss": "accumulated_other_comprehensive_income_loss",
    "acquisitionsNet": "acquisitions_net",
    "additionalPaidInCapital": "additional_paid_in_capital",
    "bottomLineNetIncome": "bottom_line_net_income",
    "capitalExpenditure": "capital_expenditure",
    "capitalLeaseObligations": "capital_lease_obligations",
    "capitalLeaseObligationsCurrent": "capital_lease_obligations_current",
    "capitalLeaseObligationsNonCurrent": "capital_lease_obligations_non_current",
    "cashAndCashEquivalents": "cash_and_cash_equivalents",
    "cashAndShortTermInvestments": "cash_and_short_term_investments",
    "cashAtBeginningOfPeriod": "cash_at_beginning_of_period",
    "cashAtEndOfPeriod": "cash_at_end_of_period",
    "changeInWorkingCapital": "change_in_working_capital",
    "commonDividendsPaid": "common_dividends_paid",
    "commonStock": "common_stock",
    "commonStockIssuance": "common_stock_issuance",
    "commonStockRepurchased": "common_stock_repurchased",
    "costAndExpenses": "cost_and_expenses",
    "costOfRevenue": "cost_of_revenue",
    "deferredIncomeTax": "deferred_income_tax",
    "deferredRevenue": "deferred_revenue",
    "deferredRevenueNonCurrent": "deferred_revenue_non_current",
    "deferredTaxLiabilitiesNonCurrent": "deferred_tax_liabilities_non_current",
    "depreciationAndAmortization": "depreciation_and_amortization",
    "ebit": "ebit",
    "ebitda": "ebitda",
    "effectOfForexChangesOnCash": "effect_of_forex_changes_on_cash",
    "eps": "eps",
    "epsDiluted": "eps_diluted",
    "freeCashFlow": "free_cash_flow",
    "generalAndAdministrativeExpenses": "general_and_administrative_expenses",
    "goodwill": "goodwill",
    "goodwillAndIntangibleAssets": "goodwill_and_intangible_assets",
    "grossProfit": "gross_profit",
    "incomeBeforeTax": "income_before_tax",
    "incomeTaxExpense": "income_tax_expense",
    "incomeTaxesPaid": "income_taxes_paid",
    "intangibleAssets": "intangible_assets",
    "interestExpense": "interest_expense",
    "interestIncome": "interest_income",
    "interestPaid": "interest_paid",
    "inventory": "inventory",
    "investmentsInPropertyPlantAndEquipment": "investments_in_property_plant_and_equipment",
    "longTermDebt": "long_term_debt",
    "longTermInvestments": "long_term_investments",
    "longTermNetDebtIssuance": "long_term_net_debt_issuance",
    "minorityInterest": "minority_interest",
    "netCashProvidedByFinancingActivities": "net_cash_provided_by_financing_activities",
    "netCashProvidedByInvestingActivities": "net_cash_provided_by_investing_activities",
    "netCashProvidedByOperatingActivities": "net_cash_provided_by_operating_activities",
    "netChangeInCash": "net_change_in_cash",
    "netCommonStockIssuance": "net_common_stock_issuance",
    "netDebt": "net_debt",
    "netDebtIssuance": "net_debt_issuance",
    "netDividendsPaid": "net_dividends_paid",
    "netIncome": "net_income",
    "netIncomeDeductions": "net_income_deductions",
    "netIncomeFromContinuingOperations": "net_income_from_continuing_operations",
    "netIncomeFromDiscontinuedOperations": "net_income_from_discontinued_operations",
    "netInterestIncome": "net_interest_income",
    "netPreferredStockIssuance": "net_preferred_stock_issuance",
    "netReceivables": "net_receivables",
    "netStockIssuance": "net_stock_issuance",
    "nonOperatingIncomeExcludingInterest": "non_operating_income_excluding_interest",
    "operatingCashFlow": "operating_cash_flow",
    "operatingExpenses": "operating_expenses",
    "operatingIncome": "operating_income",
    "otherAdjustmentsToNetIncome": "other_adjustments_to_net_income",
    "otherAssets": "other_assets",
    "otherCurrentAssets": "other_current_assets",
    "otherCurrentLiabilities": "other_current_liabilities",
    "otherExpenses": "other_expenses",
    "otherFinancingActivities": "other_financing_activities",
    "otherInvestingActivities": "other_investing_activities",
    "otherLiabilities": "other_liabilities",
    "otherNonCashItems": "other_non_cash_items",
    "otherNonCurrentAssets": "other_non_current_assets",
    "otherNonCurrentLiabilities": "other_non_current_liabilities",
    "otherPayables": "other_payables",
    "otherReceivables": "other_receivables",
    "otherTotalStockholdersEquity": "other_total_stockholders_equity",
    "otherWorkingCapital": "other_working_capital",
    "preferredDividendsPaid": "preferred_dividends_paid",
    "preferredStock": "preferred_stock",
    "prepaids": "prepaids",
    "propertyPlantEquipmentNet": "property_plant_equipment_net",
    "purchasesOfInvestments": "purchases_of_investments",
    "researchAndDevelopmentExpenses": "research_and_development_expenses",
    "retainedEarnings": "retained_earnings",
    "revenue": "revenue",
    "salesMaturitiesOfInvestments": "sales_maturities_of_investments",
    "sellingAndMarketingExpenses": "selling_and_marketing_expenses",
    "sellingGeneralAndAdministrativeExpenses": "selling_general_and_administrative_expenses",
    "shortTermDebt": "short_term_debt",
    "shortTermInvestments": "short_term_investments",
    "shortTermNetDebtIssuance": "short_term_net_debt_issuance",
    "stockBasedCompensation": "stock_based_compensation",
    "taxAssets": "tax_assets",
    "taxPayables": "tax_payables",
    "totalAssets": "total_assets",
    "totalCurrentAssets": "total_current_assets",
    "totalCurrentLiabilities": "total_current_liabilities",
    "totalDebt": "total_debt",
    "totalEquity": "total_equity",
    "totalInvestments": "total_investments",
    "totalLiabilities": "total_liabilities",
    "totalLiabilitiesAndTotalEquity": "total_liabilities_and_total_equity",
    "totalNonCurrentAssets": "total_non_current_assets",
    "totalNonCurrentLiabilities": "total_non_current_liabilities",
    "totalOtherIncomeExpensesNet": "total_other_income_expenses_net",
    "totalPayables": "total_payables",
    "totalStockholdersEquity": "total_stockholders_equity",
    "treasuryStock": "treasury_stock",
    "weightedAverageShsOut": "weighted_average_shs_out",
    "weightedAverageShsOutDil": "weighted_average_shs_out_dil",
}

# 역매핑 (snake_case → camelCase) — create_feature.py 호환용
FINANCIAL_COLUMN_MAP_REVERSE = {v: k for k, v in FINANCIAL_COLUMN_MAP.items()}


# ──────────────────────────────────────────────────────────────────────────────
# ORM 모델
# ──────────────────────────────────────────────────────────────────────────────

class FmpTicker(FmpBase):
    __tablename__ = "fmp_tickers"

    ticker_id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False)
    company_name = Column(String(200))
    exchange = Column(String(50))
    industry = Column(String(100))
    sector = Column(String(100))
    country = Column(String(10))
    currency = Column(String(10))
    ipo_date = Column(Date)
    is_actively_trading = Column(Boolean)
    is_etf = Column(Boolean)
    is_adr = Column(Boolean)
    is_fund = Column(Boolean)
    cik = Column(String(20))
    isin = Column(String(20))
    cusip = Column(String(20))
    retrieved_at = Column(DateTime(timezone=True), default=_utc_now)

    prices = relationship("FmpDailyPrice", back_populates="ticker", cascade="all, delete-orphan")
    market_caps = relationship("FmpMarketCap", back_populates="ticker", cascade="all, delete-orphan")
    financials = relationship("FmpFinancial", back_populates="ticker", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<FmpTicker(symbol='{self.symbol}')>"


class FmpDailyPrice(FmpBase):
    __tablename__ = "fmp_daily_prices"

    price_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey("fmp_tickers.ticker_id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)
    change = Column(Float)
    change_percent = Column(Float)
    vwap = Column(Float)

    ticker = relationship("FmpTicker", back_populates="prices")

    __table_args__ = (
        UniqueConstraint("ticker_id", "date", name="uix_fmp_price_ticker_date"),
        Index("idx_fmp_price_ticker_date", "ticker_id", "date"),
    )


class FmpMarketCap(FmpBase):
    __tablename__ = "fmp_market_caps"

    mcap_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey("fmp_tickers.ticker_id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    market_cap = Column(Float)

    ticker = relationship("FmpTicker", back_populates="market_caps")

    __table_args__ = (
        UniqueConstraint("ticker_id", "date", name="uix_fmp_mcap_ticker_date"),
        Index("idx_fmp_mcap_ticker_date", "ticker_id", "date"),
    )


class FmpFinancial(FmpBase):
    __tablename__ = "fmp_financials"

    financial_id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey("fmp_tickers.ticker_id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    period = Column(String(5), nullable=False)
    fiscal_year = Column(Integer)
    reported_currency = Column(String(10))
    filing_date = Column(Date)
    accepted_date = Column(String(30))

    # ── 수치 컬럼 119개 (Income / Balance Sheet / Cash Flow) ──
    account_payables = Column(Float)
    accounts_payables = Column(Float)
    accounts_receivables = Column(Float)
    accrued_expenses = Column(Float)
    accumulated_other_comprehensive_income_loss = Column(Float)
    acquisitions_net = Column(Float)
    additional_paid_in_capital = Column(Float)
    bottom_line_net_income = Column(Float)
    capital_expenditure = Column(Float)
    capital_lease_obligations = Column(Float)
    capital_lease_obligations_current = Column(Float)
    capital_lease_obligations_non_current = Column(Float)
    cash_and_cash_equivalents = Column(Float)
    cash_and_short_term_investments = Column(Float)
    cash_at_beginning_of_period = Column(Float)
    cash_at_end_of_period = Column(Float)
    change_in_working_capital = Column(Float)
    common_dividends_paid = Column(Float)
    common_stock = Column(Float)
    common_stock_issuance = Column(Float)
    common_stock_repurchased = Column(Float)
    cost_and_expenses = Column(Float)
    cost_of_revenue = Column(Float)
    deferred_income_tax = Column(Float)
    deferred_revenue = Column(Float)
    deferred_revenue_non_current = Column(Float)
    deferred_tax_liabilities_non_current = Column(Float)
    depreciation_and_amortization = Column(Float)
    ebit = Column(Float)
    ebitda = Column(Float)
    effect_of_forex_changes_on_cash = Column(Float)
    eps = Column(Float)
    eps_diluted = Column(Float)
    free_cash_flow = Column(Float)
    general_and_administrative_expenses = Column(Float)
    goodwill = Column(Float)
    goodwill_and_intangible_assets = Column(Float)
    gross_profit = Column(Float)
    income_before_tax = Column(Float)
    income_tax_expense = Column(Float)
    income_taxes_paid = Column(Float)
    intangible_assets = Column(Float)
    interest_expense = Column(Float)
    interest_income = Column(Float)
    interest_paid = Column(Float)
    inventory = Column(Float)
    investments_in_property_plant_and_equipment = Column(Float)
    long_term_debt = Column(Float)
    long_term_investments = Column(Float)
    long_term_net_debt_issuance = Column(Float)
    minority_interest = Column(Float)
    net_cash_provided_by_financing_activities = Column(Float)
    net_cash_provided_by_investing_activities = Column(Float)
    net_cash_provided_by_operating_activities = Column(Float)
    net_change_in_cash = Column(Float)
    net_common_stock_issuance = Column(Float)
    net_debt = Column(Float)
    net_debt_issuance = Column(Float)
    net_dividends_paid = Column(Float)
    net_income = Column(Float)
    net_income_deductions = Column(Float)
    net_income_from_continuing_operations = Column(Float)
    net_income_from_discontinued_operations = Column(Float)
    net_interest_income = Column(Float)
    net_preferred_stock_issuance = Column(Float)
    net_receivables = Column(Float)
    net_stock_issuance = Column(Float)
    non_operating_income_excluding_interest = Column(Float)
    operating_cash_flow = Column(Float)
    operating_expenses = Column(Float)
    operating_income = Column(Float)
    other_adjustments_to_net_income = Column(Float)
    other_assets = Column(Float)
    other_current_assets = Column(Float)
    other_current_liabilities = Column(Float)
    other_expenses = Column(Float)
    other_financing_activities = Column(Float)
    other_investing_activities = Column(Float)
    other_liabilities = Column(Float)
    other_non_cash_items = Column(Float)
    other_non_current_assets = Column(Float)
    other_non_current_liabilities = Column(Float)
    other_payables = Column(Float)
    other_receivables = Column(Float)
    other_total_stockholders_equity = Column(Float)
    other_working_capital = Column(Float)
    preferred_dividends_paid = Column(Float)
    preferred_stock = Column(Float)
    prepaids = Column(Float)
    property_plant_equipment_net = Column(Float)
    purchases_of_investments = Column(Float)
    research_and_development_expenses = Column(Float)
    retained_earnings = Column(Float)
    revenue = Column(Float)
    sales_maturities_of_investments = Column(Float)
    selling_and_marketing_expenses = Column(Float)
    selling_general_and_administrative_expenses = Column(Float)
    short_term_debt = Column(Float)
    short_term_investments = Column(Float)
    short_term_net_debt_issuance = Column(Float)
    stock_based_compensation = Column(Float)
    tax_assets = Column(Float)
    tax_payables = Column(Float)
    total_assets = Column(Float)
    total_current_assets = Column(Float)
    total_current_liabilities = Column(Float)
    total_debt = Column(Float)
    total_equity = Column(Float)
    total_investments = Column(Float)
    total_liabilities = Column(Float)
    total_liabilities_and_total_equity = Column(Float)
    total_non_current_assets = Column(Float)
    total_non_current_liabilities = Column(Float)
    total_other_income_expenses_net = Column(Float)
    total_payables = Column(Float)
    total_stockholders_equity = Column(Float)
    treasury_stock = Column(Float)
    weighted_average_shs_out = Column(Float)
    weighted_average_shs_out_dil = Column(Float)

    ticker = relationship("FmpTicker", back_populates="financials")

    __table_args__ = (
        UniqueConstraint("ticker_id", "date", "period", name="uix_fmp_fin_ticker_date_period"),
        Index("idx_fmp_fin_ticker_date", "ticker_id", "date"),
    )


class FmpTreasuryRate(FmpBase):
    __tablename__ = "fmp_treasury_rates"

    treasury_id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, unique=True, nullable=False)
    month_1 = Column(Float)
    month_2 = Column(Float)
    month_3 = Column(Float)
    month_6 = Column(Float)
    year_1 = Column(Float)
    year_2 = Column(Float)
    year_3 = Column(Float)
    year_5 = Column(Float)
    year_7 = Column(Float)
    year_10 = Column(Float)
    year_20 = Column(Float)
    year_30 = Column(Float)

    __table_args__ = (
        Index("idx_fmp_treasury_date", "date"),
    )


class FmpUniverseCalendar(FmpBase):
    __tablename__ = "fmp_universe_calendar"

    calendar_id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    symbol = Column(String(20), nullable=False)

    __table_args__ = (
        UniqueConstraint("date", "symbol", name="uix_fmp_cal_date_symbol"),
        Index("idx_fmp_cal_date", "date"),
        Index("idx_fmp_cal_symbol", "symbol"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 테이블 생성 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def create_fmp_tables(db_path: str = "data/database/fmp_stocks.db"):
    """FMP 전용 테이블을 생성한다."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    FmpBase.metadata.create_all(engine)
    print(f"FMP tables created: {db_path}")
    return engine


if __name__ == "__main__":
    create_fmp_tables()
