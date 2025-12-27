"""
확장된 DataPipeline._coverage_ok (max_consecutive_missing 제거 버전) 유닛 테스트

이 테스트는 프로젝트의 무거운 의존성(예: DB, SQLAlchemy 등)이 없어도 돌아가도록
src/data/pipeline.py를 importlib로 직접 로드하고,
pipeline.py가 import하는 일부 모듈을 스텁(stub)으로 대체한다.

실행:
  pytest -q tests/unit/test_coverage_ok.py
"""

from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


# ------------------------------
# Helpers: pipeline.py를 안전하게 로드하기
# ------------------------------

def _install_stub_project_modules() -> None:
    """pipeline.py를 단독으로 import하기 위해, 무거운 프로젝트 모듈을 스텁으로 대체한다."""
    # 'src'와 'src.data' 패키지 틀을 만든다
    src_pkg = types.ModuleType("src")
    data_pkg = types.ModuleType("src.data")
    data_pkg.__path__ = []  # 패키지로 인식시키기 위한 설정

    sys.modules.setdefault("src", src_pkg)
    sys.modules.setdefault("src.data", data_pkg)

    # 스텁 1) src.data.data_fetcher
    data_fetcher_mod = types.ModuleType("src.data.data_fetcher")

    class _StubStockDataFetcher:
        def __init__(self, *args, **kwargs):
            # _coverage_ok에서 참조하는 속성만 최소로 제공
            self.calendar_cache = {}
            self.custom_holidays = {}

        def _get_exchange_code(self, ticker_code: str):
            # 티커가 무엇이든 테스트용 거래소 코드로 고정
            return "FAKE_EXCHANGE"

    data_fetcher_mod.StockDataFetcher = _StubStockDataFetcher
    sys.modules.setdefault("src.data.data_fetcher", data_fetcher_mod)

    # 스텁 2) src.data.db_manager
    db_manager_mod = types.ModuleType("src.data.db_manager")

    class _StubDatabaseManager:
        def __init__(self, *args, **kwargs):
            pass

    db_manager_mod.DatabaseManager = _StubDatabaseManager
    sys.modules.setdefault("src.data.db_manager", db_manager_mod)

    # 스텁 3) src.data.indicator_calculator
    ind_mod = types.ModuleType("src.data.indicator_calculator")

    class _StubIndicatorCalculator:
        @staticmethod
        def get_lookback_days(indicator_list):
            return 0

    ind_mod.IndicatorCalculator = _StubIndicatorCalculator
    sys.modules.setdefault("src.data.indicator_calculator", ind_mod)


def _load_DataPipeline_class():
    """
    src/data/pipeline.py에서 DataPipeline 클래스를 importlib로 직접 로드한다.
    (패키지 __init__ 실행/의존성 로딩을 최소화하기 위함)
    """
    _install_stub_project_modules()

    # tests/unit/test_coverage_ok.py 기준으로 프로젝트 루트(/app)를 찾는다
    repo_root = Path(__file__).resolve().parents[2]
    pipeline_path = repo_root / "src" / "data" / "pipeline.py"
    assert pipeline_path.exists(), f"pipeline.py not found at {pipeline_path}"

    spec = importlib.util.spec_from_file_location("pipeline_under_test", pipeline_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod.DataPipeline


# ------------------------------
# Helpers: exchange_calendars 가짜 모듈/캘린더 만들기
# ------------------------------

class FakeCalendar:
    """
    exchange_calendars의 캘린더 객체를 아주 최소 기능으로 흉내낸 클래스.
    sessions_in_range(start, end)만 제공한다.
    """

    def __init__(self, sessions):
        self._sessions = pd.DatetimeIndex(pd.to_datetime(sessions))

    def sessions_in_range(self, start, end):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        # start~end 포함(inclusive) 범위로 세션만 리턴
        return self._sessions[(self._sessions >= start) & (self._sessions <= end)]


def _install_fake_exchange_calendars(fake_calendar: FakeCalendar):
    """
    sys.modules에 'exchange_calendars'라는 가짜 모듈을 주입한다.
    _coverage_ok 내부에서 import exchange_calendars가 호출되면 이 가짜 모듈이 로드된다.
    """
    xcals = types.ModuleType("exchange_calendars")

    def get_calendar(exchange_code: str):
        # 어떤 exchange_code가 오든 fake_calendar를 반환
        return fake_calendar

    xcals.get_calendar = get_calendar
    sys.modules["exchange_calendars"] = xcals


def _make_pipeline_instance(DataPipeline):
    """
    DataPipeline.__init__을 실행하지 않고 인스턴스를 만든 뒤,
    _coverage_ok에서 필요한 최소 속성만 붙여준다.
    """
    p = DataPipeline.__new__(DataPipeline)
    p._to_datetime_safe = lambda x: None if x is None else pd.to_datetime(x)

    # _coverage_ok에서 사용하는 fetcher 속성만 최소로 구성
    p.fetcher = SimpleNamespace(
        calendar_cache={},
        custom_holidays={},
        _get_exchange_code=lambda ticker: "FAKE_EXCHANGE",
    )
    return p


# ------------------------------
# Tests
# ------------------------------

@pytest.fixture(scope="module")
def DataPipeline():
    """테스트 대상 DataPipeline 클래스를 로드한다."""
    return _load_DataPipeline_class()


def test_returns_false_on_empty_or_missing_date(DataPipeline):
    """
    df가 None/비어있음/date 컬럼 없음이면 _coverage_ok는 False를 반환해야 한다.
    """
    p = _make_pipeline_instance(DataPipeline)

    assert p._coverage_ok(None, "2023-01-01", "2023-01-10", ticker_code="AAA") is False
    assert p._coverage_ok(pd.DataFrame(), "2023-01-01", "2023-01-10", ticker_code="AAA") is False
    assert p._coverage_ok(pd.DataFrame({"x": [1, 2]}), "2023-01-01", "2023-01-10", ticker_code="AAA") is False


def test_endpoints_only_check_when_check_gaps_false(DataPipeline):
    """
    check_gaps=False이면 '양끝(min/max 날짜)'만 보고 판단한다.
    즉, 캘린더 기반 gap 체크는 하지 않는다.
    """
    p = _make_pipeline_instance(DataPipeline)

    df = pd.DataFrame({
        "date": ["2023-01-05", "2023-01-06"],
        "close": [100, 101],
    })

    # start가 df.min보다 이르지만, tol_days=7이므로 양끝 커버 조건을 통과할 수 있다
    assert p._coverage_ok(df, "2023-01-01", "2023-01-06", tol_days=7, check_gaps=False) is True

    # start가 너무 늦으면 양끝 커버리지에서 실패해야 한다
    assert p._coverage_ok(df, "2023-01-20", "2023-01-30", tol_days=0, check_gaps=False) is False


def test_calendar_gap_detects_missing_trading_day(DataPipeline):
    """
    캘린더 기반 gap 체크:
    expected(거래일) 중 actual(df 날짜)에 없는 날이 있으면 missing으로 잡아야 한다.
    """
    p = _make_pipeline_instance(DataPipeline)

    # 테스트용 거래일(세션) 목록: 1/2~1/10 사이 거래일이라고 가정
    fake_sessions = [
        "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05",
        "2023-01-06", "2023-01-09", "2023-01-10",
    ]
    _install_fake_exchange_calendars(FakeCalendar(fake_sessions))

    # 실제 df에서는 2023-01-05가 빠져 있다고 가정
    df = pd.DataFrame({
        "date": ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-06", "2023-01-09", "2023-01-10"],
        "close": [1, 2, 3, 4, 5, 6],
    })

    # 기본값: 누락 허용 0 → 거래일 1개라도 missing이면 False
    assert p._coverage_ok(
        df,
        start_date="2023-01-02",
        end_date="2023-01-10",
        tol_days=0,
        ticker_code="AAA.KS",
        max_missing_trading_days=0,
        max_missing_ratio=0.0,
    ) is False

    # 누락 1개 허용하면 True가 되어야 한다
    assert p._coverage_ok(
        df,
        start_date="2023-01-02",
        end_date="2023-01-10",
        tol_days=0,
        ticker_code="AAA.KS",
        max_missing_trading_days=1,
        max_missing_ratio=0.5,
    ) is True


def test_nan_row_is_counted_as_missing_for_price_data(DataPipeline):
    """
    가격데이터(OHLCV)가 있고, 특정 날짜가 '모든 가격 컬럼이 NaN'이면
    그 날짜는 사실상 데이터가 없는 것으로 간주해 missing에 포함되어야 한다.
    """
    p = _make_pipeline_instance(DataPipeline)

    fake_sessions = [
        "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05",
        "2023-01-06", "2023-01-09", "2023-01-10",
    ]
    _install_fake_exchange_calendars(FakeCalendar(fake_sessions))

    # 날짜는 모두 존재하지만 2023-01-05의 OHLCV가 전부 NaN(None)인 케이스
    df = pd.DataFrame({
        "date": fake_sessions,
        "open":   [1, 1, 1, None, 1, 1, 1],
        "high":   [1, 1, 1, None, 1, 1, 1],
        "low":    [1, 1, 1, None, 1, 1, 1],
        "close":  [1, 1, 1, None, 1, 1, 1],
        "volume": [1, 1, 1, None, 1, 1, 1],
    })

    # 누락 허용 0이면 NaN row(1/5)가 missing으로 잡혀 False가 되어야 한다
    assert p._coverage_ok(
        df,
        start_date="2023-01-02",
        end_date="2023-01-10",
        tol_days=0,
        ticker_code="AAA.KS",
        max_missing_trading_days=0,
        max_missing_ratio=0.0,
    ) is False


def test_fallback_gap_detection_when_calendar_unavailable(DataPipeline):
    """
    exchange_calendars를 사용할 수 없으면 fallback 로직(날짜 간격 기반)을 사용해야 한다.
    - 날짜 간격이 너무 크면(False)
    - 작으면(True)
    """
    p = _make_pipeline_instance(DataPipeline)

    # exchange_calendars 모듈을 제거해서, _coverage_ok가 try 블록에서 실패 → fallback으로 내려가게 만든다
    sys.modules.pop("exchange_calendars", None)

    # 날짜 간격이 큰 경우: max_gap_days_fallback=10이면 실패해야 한다
    df_big_gap = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-25"],
        "close": [1, 2],
    })

    assert p._coverage_ok(
        df_big_gap,
        start_date="2023-01-01",
        end_date="2023-01-25",
        tol_days=0,
        ticker_code="AAA.KS",
        max_gap_days_fallback=10,
    ) is False

    # 날짜 간격이 작은 경우는 통과해야 한다
    df_small_gap = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-05"],
        "close": [1, 2],
    })

    assert p._coverage_ok(
        df_small_gap,
        start_date="2023-01-01",
        end_date="2023-01-05",
        tol_days=0,
        ticker_code="AAA.KS",
        max_gap_days_fallback=10,
    ) is True



def test_allow_early_end_truncates_gap_check_range(DataPipeline):
    p = _make_pipeline_instance(DataPipeline)

    fake_sessions = ["2023-01-02","2023-01-03","2023-01-04","2023-01-05","2023-01-06","2023-01-09","2023-01-10"]
    _install_fake_exchange_calendars(FakeCalendar(fake_sessions))

    df = pd.DataFrame({
        "date": ["2023-01-02","2023-01-03","2023-01-04"],  # dmax=01/04
        "close": [1,2,3],
    })

    assert p._coverage_ok(
        df,
        start_date="2023-01-01",
        end_date="2023-01-10",
        tol_days=0,
        ticker_code="AAA.KS",
        allow_early_end=True,
        max_missing_trading_days=0,
        max_missing_ratio=0.0,
    ) is True

def test_allow_early_end_does_not_ignore_internal_gaps(DataPipeline):
    p = _make_pipeline_instance(DataPipeline)

    fake_sessions = ["2023-01-02","2023-01-03","2023-01-04","2023-01-05","2023-01-06","2023-01-09","2023-01-10"]
    _install_fake_exchange_calendars(FakeCalendar(fake_sessions))

    df = pd.DataFrame({
        "date": ["2023-01-02","2023-01-04"],  # 내부에 01/03 누락
        "close": [1,3],
    })

    assert p._coverage_ok(
        df,
        start_date="2023-01-01",
        end_date="2023-01-10",
        tol_days=0,
        ticker_code="AAA.KS",
        allow_early_end=True,
        max_missing_trading_days=0,
        max_missing_ratio=0.0,
    ) is False
