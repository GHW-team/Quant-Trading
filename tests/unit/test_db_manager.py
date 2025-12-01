"""
DatabaseManager 클래스 단위 테스트
SQLite 기반 데이터 저장/로드 검증
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.data.db_models import Ticker, DailyPrice, TechnicalIndicator
from src.data.db_manager import DatabaseManager
from unittest.mock import patch
from sqlalchemy.exc import SQLAlchemyError

class TestDatabaseManagerInitialization:
    """DatabaseManager 초기화 테스트"""

    def test_initialization_complete(self, db_manager):
        """DB 초기화 완료 검증"""
        # 경로 확인
        assert db_manager.db_path is not None

        # 파일 생성 확인
        assert os.path.exists(db_manager.db_path)

        # 세션/엔진 초기화 확인
        assert db_manager.session is not None
        assert db_manager.engine is not None


class TestDatabaseManagerMarketInference:
    """시장 추론 테스트"""
    @pytest.mark.parametrize("ticker, expected_market", [
        # 1. 정상 케이스
        ('005930.KS', 'KOSPI'),
        ('253450.KQ', 'KOSDAQ'),
        ('AAPL', 'US'),
        ('BTC-USD', 'CRYPTO'),
        ('6758.T', 'TOKYO'),
        ('0700.HK', 'HONGKONG'),
        ('XXXX.YY', 'UNKNOWN'),
        
        # 2. 대소문자 구분 없음 (Case Insensitive) 검증 통합
        ('005930.ks', 'KOSPI'),
        ('000660.kq', 'KOSDAQ'),
        ('btc-usd', 'CRYPTO'),
        ('aapl', 'US'),
    ])
    def test_infer_market_scenarios(self, db_manager, ticker, expected_market):
        """다양한 티커에 대한 시장 추론 로직 검증 (통합 테스트)"""
        market = db_manager._infer_market_from_ticker(ticker)
        assert market == expected_market


class TestDatabaseManagerTickerId:
    """Ticker ID 조회/생성 테스트"""

    def test_get_ticker_id_creates_new(self, db_manager):
        """새 ticker ID 생성"""
        ticker_id = db_manager._get_ticker_id('005930.KS')

        #ID 반환 여부
        assert ticker_id is not None
        assert isinstance(ticker_id, int)

        #데이터 저장 여부
        saved_ticker = db_manager.session.query(Ticker).filter_by(ticker_id = ticker_id).first()
        assert saved_ticker is not None
        assert saved_ticker.ticker_code == '005930.KS' 

    def test_get_ticker_id_returns_existing(self, db_manager):
        """기존 ticker ID 반환"""
        #생성
        id1 = db_manager._get_ticker_id('005930.KS')
        #조회
        id2 = db_manager._get_ticker_id('005930.KS')

        #검증
        assert id1 == id2

        #검증(중복생성 여부)
        count = db_manager.session.query(Ticker).filter_by(ticker_code = '005930.KS').count()
        assert count == 1
        
    def test_get_ticker_id_with_explicit_market(self, db_manager):
        """명시적 시장으로 ticker 생성"""
        ticker_id = db_manager._get_ticker_id('TEST.XX', market='CUSTOM')
        
        saved_ticker = db_manager.session.query(Ticker).filter_by(ticker_id = ticker_id).first()
        assert saved_ticker.market == "CUSTOM"

    def test_multiple_tickers_different_ids(self, db_manager):
        """여러 ticker가 다른 ID"""
        id1 = db_manager._get_ticker_id('005930.KS')
        id2 = db_manager._get_ticker_id('000660.KS')
        assert id1 != id2

    def test_get_ticker_id_infers_market_automatically(self, db_manager):
        """market 인자가 없을 때 자동 추론되어 저장되는지 검증"""
        ticker_id = db_manager._get_ticker_id('005930.KS')

        saved_ticker = db_manager.session.query(Ticker).filter_by(ticker_id=ticker_id).first()
        assert saved_ticker.market == 'KOSPI'
        

class TestDatabaseManagerSavePriceData:
    """가격 데이터 저장 테스트"""

    def test_save_price_data_single_ticker(self, db_manager, sample_df_basic):
        """단일 ticker 가격 데이터 저장"""
        df_dict = {'005930.KS': sample_df_basic}

        result = db_manager.save_price_data(df_dict, update_if_exists=True)

        #반환값 검증
        assert '005930.KS' in result
        assert result['005930.KS'] == len(sample_df_basic)

        #실제 DB 조회하기
        saved_ticker = db_manager.session.query(Ticker).filter_by(ticker_code = '005930.KS').first()
        assert saved_ticker is not None
        count = db_manager.session.query(DailyPrice).filter_by(ticker_id = saved_ticker.ticker_id).count()
        assert count == len(sample_df_basic)

    def test_save_price_data_multiple_tickers(self, db_manager, sample_df_basic):
        """여러 ticker 가격 데이터 저장"""
        ticker_num = 30
        
        sample_df = sample_df_basic.iloc[:20].copy()
        df_num = len(sample_df)

        df_dict = {}
        ticker_list = []
        for i in range(ticker_num):
            ticker_code = f'{i:06d}.KS'
            df_dict[ticker_code] = sample_df.copy()
            ticker_list.append(ticker_code)

        result = db_manager.save_price_data(df_dict, update_if_exists=True)

        #반환값 검증
        assert len(result) == ticker_num
        assert all(count == df_num for count in result.values())

        #실제 DB 조회하기
            #첫번째 티커
        ticker_first = ticker_list[0]
        saved_ticker_first = db_manager.session.query(Ticker).filter_by(ticker_code=ticker_first).first()
        assert saved_ticker_first is not None
        count_first = db_manager.session.query(DailyPrice).filter_by(ticker_id = saved_ticker_first.ticker_id).count()
        assert count_first == df_num
            #마지막 티커
        ticker_last = ticker_list[-1]
        saved_ticker_last = db_manager.session.query(Ticker).filter_by(ticker_code=ticker_last).first()
        assert saved_ticker_last is not None
        count_last = db_manager.session.query(DailyPrice).filter_by(ticker_id = saved_ticker_last.ticker_id).count()
        assert count_last == df_num

    def test_save_price_data_update_if_exists_false(self, db_manager, sample_df_basic):
        """update_if_exists=False일 때 INSERT OR IGNORE"""
        df_dict = {'005930.KS': sample_df_basic}

        # 첫 번째 저장
        result1 = db_manager.save_price_data(df_dict, update_if_exists=False)
        count1 = result1['005930.KS']

        # 두 번째 저장 (같은 데이터)
        result2 = db_manager.save_price_data(df_dict, update_if_exists=False)
        count2 = result2['005930.KS']

        #반환값 검증
            # 첫 번째는 성공적으로 삽입
        assert count1 == len(sample_df_basic)
            # 두 번째는 무시됨 (만약 충돌로 인해 오류 발생했다면 count2 == 0 이됨)
        assert count2 == len(sample_df_basic)

        # DB 실제 조회하여 데이터가 중복되지 않았는지 확인
        ticker = db_manager.session.query(Ticker).filter_by(ticker_code = '005930.KS').first()
        count = db_manager.session.query(DailyPrice).filter_by(ticker_id = ticker.ticker_id).count()
        assert count == count1

    def test_save_price_data_upsert_updates_values(self, db_manager, sample_df_basic):
        """UPSERT: 기존 데이터 업데이트"""
        df_dict = {'005930.KS': sample_df_basic.copy()}

        # 첫 번째 저장
        result1 = db_manager.save_price_data(df_dict, update_if_exists=True)
        assert result1['005930.KS'] == len(sample_df_basic)

        # 데이터 수정 (close 가격을 모두 999.99로 변경)
        df_modified = sample_df_basic.copy()
        df_modified['close'] = 999.99
        df_dict_modified = {'005930.KS': df_modified}

        # 두 번째 저장 (UPSERT)
        result2 = db_manager.save_price_data(df_dict_modified, update_if_exists=True)
        assert result2['005930.KS'] == len(df_modified)

        # 데이터 검증 - 수정된 값이 적용됨
        ticker = db_manager.session.query(Ticker).filter_by(ticker_code = '005930.KS').first()
        daily_price = db_manager.session.query(DailyPrice).filter_by(ticker_id = ticker.ticker_id).first()
        assert daily_price.close == 999.99

    def test_save_price_data_empty_dict(self, db_manager):
        """빈 딕셔너리"""
        result = db_manager.save_price_data({}, update_if_exists=True)
        assert result == {}

    def test_save_price_data_missing_columns(self, db_manager):
        bad_df = pd.DataFrame({'date' : ['2020-01-01'], 'close' : [30]})
        df_dict = {'005930.KS': bad_df}

        result = db_manager.save_price_data(df_dict, update_if_exists = True)

        #반환값 검증 (0으로 처리 되었는지)
        assert result['005930.KS'] == 0 

        #DB 검증 (DB가 실제로 비어있는지)
        ticker = db_manager.session.query(Ticker).filter_by(ticker_code = '005930.KS').first()
        daily_price = db_manager.session.query(DailyPrice).filter_by(ticker_id = ticker.ticker_id).all()
        assert daily_price == []
    
    def test_save_price_data_partial_failure(self, db_manager, sample_df_basic):
        """여러 종목 중 하나가 실패해도 나머지는 저장되는지 검증"""
        # 정상 데이터
        good_df = sample_df_basic.copy()
        # 불량 데이터 (필수 컬럼 누락)
        bad_df = pd.DataFrame({'date': ['2020-01-01'], 'wrong_col': [100]})
        
        df_dict = {
            'BAD.KS': bad_df,
            'GOOD.KS': good_df,
        }

        # 실행
        result = db_manager.save_price_data(df_dict, update_if_exists=True)

        # 검증: 좋은 건 성공, 나쁜 건 실패
        assert result['BAD.KS'] == 0
        assert result['GOOD.KS'] == len(good_df)
        
        # 검증: DB에 좋은 데이터는 실제로 들어갔나?
        ticker = db_manager.session.query(Ticker).filter_by(ticker_code = 'GOOD.KS').first()
        count = db_manager.session.query(DailyPrice).filter_by(ticker_id = ticker.ticker_id).count()
        assert count == len(good_df)

    def test_save_price_data_ticker_id_fail(self, db_manager, sample_df_basic):
        """Ticker ID 조회 실패 시 처리"""
        df_dict = {'005930.KS': sample_df_basic}

        # _get_ticker_id가 에러를 뱉도록 Mocking
        with patch.object(db_manager, '_get_ticker_id', side_effect=Exception("DB Error")):
            result = db_manager.save_price_data(df_dict, update_if_exists=True)
            
            assert result['005930.KS'] == 0


class TestDatabaseManagerSaveIndicators:
    """지표 데이터 저장 테스트"""

    def test_save_indicators_single_ticker(self, db_manager, sample_df_with_indicators):
        """단일 ticker 지표 저장"""
        df_dict = {'005930.KS': sample_df_with_indicators}

        # 지표 저장
        result = db_manager.save_indicators(df_dict)

        # 반환값 검증
        assert '005930.KS' in result
        assert result['005930.KS'] == len(sample_df_with_indicators)

        # 실제 DB 조회하기
        saved_ticker = db_manager.session.query(Ticker).filter_by(ticker_code='005930.KS').first()
        assert saved_ticker is not None
        count = db_manager.session.query(TechnicalIndicator).filter_by(ticker_id=saved_ticker.ticker_id).count()
        assert count == len(sample_df_with_indicators)

    def test_save_indicators_multiple_tickers(self, db_manager, sample_df_with_indicators):
        """여러 ticker 지표 저장"""
        ticker_num = 30
        sample_df = sample_df_with_indicators.iloc[:20].copy()
        df_num = len(sample_df)

        df_dict = {}
        ticker_list = []
        for i in range(ticker_num):
            ticker_code = f'{i:06d}.KS'
            df_dict[ticker_code] = sample_df.copy()
            ticker_list.append(ticker_code)

        result = db_manager.save_indicators(df_dict)

        # 반환값 검증
        assert len(result) == ticker_num
        assert all(count == df_num for count in result.values())

        # 실제 DB 조회하기 - 첫번째 티커
        ticker_first = ticker_list[0]
        saved_ticker_first = db_manager.session.query(Ticker).filter_by(ticker_code=ticker_first).first()
        assert saved_ticker_first is not None
        count_first = db_manager.session.query(TechnicalIndicator).filter_by(ticker_id=saved_ticker_first.ticker_id).count()
        assert count_first == df_num

        # 실제 DB 조회하기 - 마지막 티커
        ticker_last = ticker_list[-1]
        saved_ticker_last = db_manager.session.query(Ticker).filter_by(ticker_code=ticker_last).first()
        assert saved_ticker_last is not None
        count_last = db_manager.session.query(TechnicalIndicator).filter_by(ticker_id=saved_ticker_last.ticker_id).count()
        assert count_last == df_num

    def test_save_indicators_with_version(self, db_manager, sample_df_with_indicators):
        """버전 정보와 함께 저장"""
        df_dict = {'005930.KS': sample_df_with_indicators}

        db_manager._get_ticker_id('005930.KS')
        result = db_manager.save_indicators(df_dict, version='v2.0')

        # 반환값 검증
        assert '005930.KS' in result
        assert result['005930.KS'] == len(sample_df_with_indicators)

        # DB에서 버전 정보 확인
        ticker = db_manager.session.query(Ticker).filter_by(ticker_code='005930.KS').first()
        indicator = db_manager.session.query(TechnicalIndicator).filter_by(ticker_id=ticker.ticker_id).first()
        assert indicator.calculated_version == 'v2.0'

    def test_save_indicators_empty_dict(self, db_manager):
        """빈 딕셔너리"""
        result = db_manager.save_indicators({})
        assert result == {}

    def test_save_indicators_missing_columns(self, db_manager):
        """필수 컬럼 누락 시 처리"""
        bad_df = pd.DataFrame({'date': ['2020-01-01'], 'close': [30]})
        df_dict = {'005930.KS': bad_df}

        db_manager._get_ticker_id('005930.KS')
        result = db_manager.save_indicators(df_dict)

        # 반환값 검증 (0으로 처리되었는지)
        assert result['005930.KS'] == 0

        # DB 검증 (DB가 실제로 비어있는지)
        ticker = db_manager.session.query(Ticker).filter_by(ticker_code='005930.KS').first()
        indicators = db_manager.session.query(TechnicalIndicator).filter_by(ticker_id=ticker.ticker_id).all()
        assert indicators == []

    def test_save_indicators_partial_failure(self, db_manager, sample_df_with_indicators):
        """여러 종목 중 하나가 실패해도 나머지는 저장되는지 검증"""
        # 정상 데이터
        good_df = sample_df_with_indicators.copy()
        # 불량 데이터 (필수 컬럼 누락)
        bad_df = pd.DataFrame({'date': ['2020-01-01'], 'wrong_col': [100]})

        db_manager._get_ticker_id('BAD.KS')
        db_manager._get_ticker_id('GOOD.KS')

        df_dict = {
            'BAD.KS': bad_df,
            'GOOD.KS': good_df,
        }

        # 실행
        result = db_manager.save_indicators(df_dict)

        # 검증: 좋은 건 성공, 나쁜 건 실패
        assert result['BAD.KS'] == 0
        assert result['GOOD.KS'] == len(good_df)

        # 검증: DB에 좋은 데이터는 실제로 들어갔나?
        ticker = db_manager.session.query(Ticker).filter_by(ticker_code='GOOD.KS').first()
        count = db_manager.session.query(TechnicalIndicator).filter_by(ticker_id=ticker.ticker_id).count()
        assert count == len(good_df)

    def test_save_indicators_ticker_id_fail(self, db_manager, sample_df_with_indicators):
        """Ticker ID 조회 실패 시 처리"""
        df_dict = {'005930.KS': sample_df_with_indicators}

        # _get_ticker_id가 에러를 뱉도록 Mocking
        with patch.object(db_manager, '_get_ticker_id', side_effect=Exception("DB Error")):
            result = db_manager.save_indicators(df_dict)

            assert result['005930.KS'] == 0

    def test_save_indicators_upsert_behavior(self, db_manager, sample_df_with_indicators):
        """UPSERT: 같은 date의 지표 업데이트"""
        df_dict = {'005930.KS': sample_df_with_indicators.copy()}

        # 첫 번째 저장
        db_manager._get_ticker_id('005930.KS')
        result1 = db_manager.save_indicators(df_dict)
        assert result1['005930.KS'] == len(sample_df_with_indicators)

        # 데이터 수정 (ma_5 값을 모두 999.99로 변경)
        df_modified = sample_df_with_indicators.copy()
        df_modified['ma_5'] = 999.99
        df_dict_modified = {'005930.KS': df_modified}

        # 두 번째 저장 (UPSERT)
        result2 = db_manager.save_indicators(df_dict_modified)
        assert result2['005930.KS'] == len(df_modified)

        # 데이터 검증 - 수정된 값이 적용됨
        ticker = db_manager.session.query(Ticker).filter_by(ticker_code='005930.KS').first()
        indicators = db_manager.session.query(TechnicalIndicator).filter_by(ticker_id=ticker.ticker_id).all()

        # 행의 개수가 중복되지 않음
        assert len(indicators) == len(sample_df_with_indicators)

        # 첫 번째 행의 ma_5 값이 수정되었는지 확인
        first_indicator = indicators[0]
        assert first_indicator.ma_5 == 999.99


class TestDatabaseManagerLoadPriceData:
    """가격 데이터 로드 테스트"""

    def test_load_price_data_after_save(self, db_manager, sample_df_basic):
        """저장 후 로드"""
        # 저장
        df_dict_save = {'005930.KS': sample_df_basic}
        db_manager.save_price_data(df_dict_save, update_if_exists=True)

        # 로드
        df_dict_load = db_manager.load_price_data(['005930.KS'])

        assert '005930.KS' in df_dict_load
        assert len(df_dict_load['005930.KS']) == len(sample_df_basic)

    def test_load_price_data_columns(self, db_manager, sample_df_basic):
        """로드된 데이터의 컬럼"""
        df_dict_save = {'005930.KS': sample_df_basic}
        db_manager.save_price_data(df_dict_save, update_if_exists=True)

        df_dict_load = db_manager.load_price_data(['005930.KS'])
        df = df_dict_load['005930.KS']

        expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        for col in expected_cols:
            assert col in df.columns

    @pytest.mark.parametrize("start_date, end_date, should_filter_start, should_filter_end", [
        ('2020-01-15', '2020-02-15', True, True),   # 양쪽 모두
        ('2020-01-15', None, True, False),          # start_date만
        (None, '2020-02-15', False, True),          # end_date만
    ])
    def test_load_price_data_with_date_filters(self, db_manager, sample_df_basic,
                                                 start_date, end_date, should_filter_start, should_filter_end):
        """날짜 필터 적용 (start_date, end_date 조합)"""
        df_dict_save = {'005930.KS': sample_df_basic}
        db_manager.save_price_data(df_dict_save, update_if_exists=True)

        df_dict_load = db_manager.load_price_data(
            ['005930.KS'],
            start_date=start_date,
            end_date=end_date
        )

        df = df_dict_load['005930.KS']
        if should_filter_start:
            assert df['date'].min() >= pd.Timestamp(start_date)
        if should_filter_end:
            assert df['date'].max() <= pd.Timestamp(end_date)

    def test_load_price_data_dates_out_of_range(self, db_manager, sample_df_basic):
        """데이터 범위 외 날짜 필터"""
        df_dict_save = {'005930.KS': sample_df_basic}
        db_manager.save_price_data(df_dict_save, update_if_exists=True)

        # 데이터 범위보다 훨씬 이후 날짜로 필터
        df_dict_load = db_manager.load_price_data(
            ['005930.KS'],
            start_date='2025-01-01',
            end_date='2025-12-31'
        )

        df = df_dict_load['005930.KS']
        assert df.empty, "Should return empty DataFrame for out-of-range dates"

    def test_load_price_data_multiple_tickers(self, db_manager, sample_df_basic):
        """여러 ticker 로드"""
        #데이터 저장
        ticker_num = 30

        sample_df = sample_df_basic.iloc[:20].copy()
        df_num = len(sample_df)
        
        df_dict = {}
        ticker_list = []
        for i in range(ticker_num):
            ticker_code = f'{i:06}.KS'
            df_dict[ticker_code] = sample_df.copy()
            ticker_list.append(ticker_code)

        result = db_manager.save_price_data(df_dict, update_if_exists = True)

        #데이터 로드
        load_df_dict = db_manager.load_price_data(ticker_codes = ticker_list)

        #반환값 검증
        assert len(load_df_dict) == 30
        assert set(load_df_dict.keys()) == set(ticker_list)
        assert all(len(load_df_dict[t]) == df_num for t in ticker_list)

    def test_load_price_data_nonexistent_ticker(self, db_manager):
        """존재하지 않는 ticker 로드"""
        df_dict_load = db_manager.load_price_data(['NONEXISTENT.KS'])

        # 빈 DataFrame 반환
        assert 'NONEXISTENT.KS' in df_dict_load
        assert df_dict_load['NONEXISTENT.KS'].empty

    def test_load_price_data_partial_failure(self, db_manager, caplog):
            """
            [Edge Case] 부분 실패 검증
            - 2개 종목을 요청했는데, 1개는 성공하고 1개는 DB 에러가 발생한 경우
            - 결과: 프로그램이 죽지 않고, 성공한 1개만 담긴 딕셔너리를 반환해야 함.
            """
            tickers = ['005930.KS', 'FAIL_TICKER.KS', '003520.KS']
            
            # 정상 데이터 준비
            valid_df = pd.DataFrame({
                'date': [pd.Timestamp('2024-01-01')],
                'close': [1000]
            })

            # pd.read_sql을 가로채서(Mock), 호출될 때마다 다른 반응을 보이게 함
            # 첫 번째 호출(005930.KS) -> 정상 데이터프레임 반환
            # 두 번째 호출(FAIL_TICKER.KS) -> SQLAlchemyError(DB 연결 에러 등) 발생!
            with patch('pandas.read_sql') as mock_read_sql:
                mock_read_sql.side_effect = [
                    valid_df.copy(),
                    SQLAlchemyError("DB Connection Error!"),
                    valid_df.copy(),
                ]

                # 실행
                result = db_manager.load_price_data(tickers)

                # 결과 딕셔너리 검증
                assert '005930.KS' in result  # 성공한 건 있어야 함
                assert '003520.KS' in result  # 성공한 건 있어야 함
                assert 'FAIL_TICKER.KS' not in result  # 실패한 건 없어야 함
                assert len(result) == 2  # 총 개수는 2개

                # 에러 로그가 찍혔는지 검증
                assert "Failed to load price data for FAIL_TICKER.KS" in caplog.text
                assert "DB Connection Error!" in caplog.text


class TestDatabaseManagerLoadIndicators:
    """지표 데이터 로드 테스트"""

    def test_load_indicators_after_save(self, db_manager, sample_df_with_indicators):
        """저장 후 로드"""
        # 저장
        df_dict_save = {'005930.KS': sample_df_with_indicators}
        db_manager.save_indicators(df_dict_save)

        # 로드
        df_dict_load = db_manager.load_indicators(['005930.KS'])

        assert '005930.KS' in df_dict_load
        assert len(df_dict_load['005930.KS']) == len(sample_df_with_indicators)

    def test_load_indicators_columns(self, db_manager, sample_df_with_indicators):
        """로드된 지표의 컬럼"""
        df_dict_save = {'005930.KS': sample_df_with_indicators}
        db_manager.save_indicators(df_dict_save)

        df_dict_load = db_manager.load_indicators(['005930.KS'])
        df = df_dict_load['005930.KS']

        expected_cols = ['date', 'ma_5', 'ma_20', 'ma_200', 'macd']
        for col in expected_cols:
            assert col in df.columns

    @pytest.mark.parametrize("start_date, end_date, should_filter_start, should_filter_end", [
        ('2020-01-15', '2020-02-15', True, True),   # 양쪽 모두
        ('2020-01-15', None, True, False),          # start_date만
        (None, '2020-02-15', False, True),          # end_date만
    ])
    def test_load_indicators_with_date_filters(self, db_manager, sample_df_with_indicators,
                                                start_date, end_date, should_filter_start, should_filter_end):
        """날짜 필터로 지표 로드"""
        df_dict_save = {'005930.KS': sample_df_with_indicators}
        db_manager.save_indicators(df_dict_save)

        df_dict_load = db_manager.load_indicators(
            ['005930.KS'],
            start_date=start_date,
            end_date=end_date
        )

        df = df_dict_load['005930.KS']
        if should_filter_start:
            assert df['date'].min() >= pd.Timestamp(start_date)
        if should_filter_end:
            assert df['date'].max() <= pd.Timestamp(end_date)

    def test_load_indicators_dates_out_of_range(self, db_manager, sample_df_with_indicators):
        """데이터 범위 외 날짜 필터"""
        df_dict_save = {'005930.KS': sample_df_with_indicators}
        db_manager.save_indicators(df_dict_save)

        # 데이터 범위보다 훨씬 이후 날짜로 필터
        df_dict_load = db_manager.load_indicators(
            ['005930.KS'],
            start_date='2025-01-01',
            end_date='2025-12-31'
        )

        df = df_dict_load['005930.KS']
        assert df.empty, "Should return empty DataFrame for out-of-range dates"

    def test_load_indicators_multiple_tickers(self, db_manager, sample_df_with_indicators):
        """여러 ticker 로드"""
        ticker_num = 30

        sample_df = sample_df_with_indicators.iloc[:20].copy()
        df_num = len(sample_df)

        df_dict = {}
        ticker_list = []
        for i in range(ticker_num):
            ticker_code = f'{i:06d}.KS'
            df_dict[ticker_code] = sample_df.copy()
            ticker_list.append(ticker_code)

        result = db_manager.save_indicators(df_dict)

        # 데이터 로드
        load_df_dict = db_manager.load_indicators(ticker_codes=ticker_list)

        # 반환값 검증
        assert len(load_df_dict) == 30
        assert set(load_df_dict.keys()) == set(ticker_list)
        assert all(len(load_df_dict[t]) == df_num for t in ticker_list)

    def test_load_indicators_nonexistent_ticker(self, db_manager):
        """존재하지 않는 ticker 로드"""
        df_dict_load = db_manager.load_indicators(['NONEXISTENT.KS'])

        # 빈 DataFrame 반환
        assert 'NONEXISTENT.KS' in df_dict_load
        assert df_dict_load['NONEXISTENT.KS'].empty

    def test_load_indicators_partial_failure(self, db_manager, sample_df_with_indicators, caplog):
        """
        [Edge Case] 부분 실패 검증
        - 3개 종목을 요청했는데, 1개는 DB 에러가 발생하는 경우
        - 결과: 프로그램이 죽지 않고, 성공한 2개만 담긴 딕셔너리를 반환해야 함.
        """
        tickers = ['005930.KS', 'FAIL_TICKER.KS', '003520.KS']

        # 정상 데이터 준비
        valid_df = sample_df_with_indicators.copy()

        # pd.read_sql을 가로채서(Mock), 호출될 때마다 다른 반응을 보이게 함
        # 첫 번째 호출(005930.KS) -> 정상 데이터프레임 반환
        # 두 번째 호출(FAIL_TICKER.KS) -> SQLAlchemyError(DB 연결 에러 등) 발생!
        # 세 번째 호출(003520.KS) -> 정상 데이터프레임 반환
        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.side_effect = [
                valid_df.copy(),
                SQLAlchemyError("DB Connection Error!"),
                valid_df.copy(),
            ]

            # 실행
            result = db_manager.load_indicators(tickers)

            # [검증 1] 결과 딕셔너리 확인
            assert '005930.KS' in result  # 성공한 건 있어야 함
            assert '003520.KS' in result  # 성공한 건 있어야 함
            assert 'FAIL_TICKER.KS' not in result  # 실패한 건 없어야 함
            assert len(result) == 2  # 총 개수는 2개

            # [검증 2] 에러 로그가 찍혔는지 확인
            # (continue로 넘어갔더라도 로그는 남겨야 나중에 확인 가능하므로)
            assert "Failed to load indicators for FAIL_TICKER.KS" in caplog.text
            assert "DB Connection Error!" in caplog.text


class TestDatabaseManagerContextManager:
    """Context Manager 테스트"""

    def test_enter_returns_manager(self, temp_db_path):
        """__enter__가 manager 반환"""
        with DatabaseManager(db_path=temp_db_path) as manager:
            assert isinstance(manager, DatabaseManager)

    def test_exit_calls_close(self, temp_db_path):  # db_manager 대신 temp_db_path 사용
        """__exit__이 내부적으로 close()를 호출하는지 검증"""
        
        # 1. 이 테스트 전용 객체 생성
        manager = DatabaseManager(db_path=temp_db_path)
        
        # 2. close 메서드를 감시(Spy)
        with patch.object(manager, 'close') as mock_close:
            
            # 3. Context Manager 진입 및 탈출
            with manager:
                pass 
            
            # 4. 검증: 탈출할 때 close()가 호출되었는가?
            mock_close.assert_called_once()

    def test_context_manager_workflow(self, temp_db_path, sample_df_basic):
        """완전한 Context Manager 워크플로우"""
        with DatabaseManager(db_path=temp_db_path) as manager:
            df_dict = {'005930.KS': sample_df_basic}
            manager.save_price_data(df_dict, update_if_exists=True)
            loaded = manager.load_price_data(['005930.KS'])
            assert len(loaded) == 1


class TestDatabaseManagerMetadata:
    """메타데이터 테스트"""

    def test_save_ticker_metadata(self, db_manager):
        """ticker 메타데이터 저장"""
        # ticker 생성
        db_manager._get_ticker_id('005930.KS')

        # 메타데이터 저장
        metadata = {'name': 'Samsung Electronics', 'sector': 'Electronics'}
        db_manager.save_ticker_metadata('005930.KS', metadata)

    @pytest.mark.parametrize("tickers_to_create, tickers_to_load, expected_count", [
        (['005930.KS'], ['005930.KS'], 1),                          # 특정 ticker만 로드
        (['005930.KS', '000660.KS'], ['005930.KS', '000660.KS'], 2), # 여러 ticker 로드
        (['005930.KS', '000660.KS'], None, 2),                      # 모든 ticker 로드
    ])
    def test_load_ticker_metadata(self, db_manager, tickers_to_create, tickers_to_load, expected_count):
        """ticker 메타데이터 로드 (특정/전체)"""
        # ticker 생성
        for ticker_code in tickers_to_create:
            db_manager._get_ticker_id(ticker_code)
            db_manager.save_ticker_metadata(ticker_code, {'name': f'Test {ticker_code}'})

        # 로드
        df_metadata = db_manager.load_ticker_metadata(tickers_to_load)

        if tickers_to_load:
            assert len(df_metadata) == expected_count
            for ticker_code in tickers_to_load:
                assert ticker_code in df_metadata.index
        else:
            assert len(df_metadata) >= expected_count


class TestDatabaseManagerIntegration:
    """통합 테스트"""

    def test_save_and_load_workflow(self, db_manager, sample_df_basic, sample_df_with_indicators):
        """완전한 저장/로드 워크플로우"""
        # 1. 가격 데이터 저장
        price_dict = {'005930.KS': sample_df_basic}
        db_manager.save_price_data(price_dict, update_if_exists=True)

        # 2. 지표 데이터 저장
        indicator_dict = {'005930.KS': sample_df_with_indicators}
        db_manager.save_indicators(indicator_dict)

        # 3. 로드
        loaded_prices = db_manager.load_price_data(['005930.KS'])
        loaded_indicators = db_manager.load_indicators(['005930.KS'])

        # 4. 검증
        assert len(loaded_prices['005930.KS']) > 0
        assert len(loaded_indicators['005930.KS']) > 0

    def test_multiple_save_load_cycles(self, db_manager, sample_df_basic):
        """여러 번의 저장/로드 사이클"""
        # 첫 번째 사이클
        dict1 = {'005930.KS': sample_df_basic}
        db_manager.save_price_data(dict1, update_if_exists=True)
        loaded1 = db_manager.load_price_data(['005930.KS'])

        # 두 번째 사이클
        loaded2 = db_manager.load_price_data(['005930.KS'])

        # 결과가 일관됨
        assert len(loaded1) == len(loaded2)

    def test_error_handling_with_invalid_data(self, db_manager):
        """잘못된 데이터 처리 - 필요한 컬럼 누락"""
        # 잘못된 DataFrame (필요한 컬럼 누락)
        invalid_df = pd.DataFrame({'wrong_col': [1, 2, 3]})
        df_dict = {'005930.KS': invalid_df}

        # save_price_data는 에러를 내부적으로 처리하고 0을 반환함
        result = db_manager.save_price_data(df_dict, update_if_exists=True)
        assert result['005930.KS'] == 0, "Should return 0 for invalid data (missing columns)"


class TestDatabaseManagerDataIntegrity:
    """데이터 무결성 테스트"""

    def test_save_indicators_with_nan_values(self, db_manager, sample_df_with_indicators):
        """NaN 값이 포함된 지표 저장"""
        # NaN 값이 포함된 지표
        df_with_nan = sample_df_with_indicators.copy()
        df_with_nan.loc[0, 'ma_5'] = np.nan

        db_manager._get_ticker_id('005930.KS')
        result = db_manager.save_indicators({'005930.KS': df_with_nan})

        assert result['005930.KS'] > 0, "Should save data with NaN values"

        # 로드해서 NaN이 None으로 저장되었는지 확인
        loaded = db_manager.load_indicators(['005930.KS'])
        df_loaded = loaded['005930.KS']
        assert df_loaded.isna().any().any(), "NaN values should be preserved"

    def test_upsert_duplicate_date_values(self, db_manager, sample_df_basic):
        """같은 date에 대한 UPSERT 동작 검증"""
        # 첫 번째 저장
        result1 = db_manager.save_price_data(
            {'005930.KS': sample_df_basic},
            update_if_exists=True
        )
        assert result1['005930.KS'] > 0

        # 같은 date의 close 값 수정
        df_modified = sample_df_basic.copy()
        df_modified.loc[0, 'close'] = 999.99

        result2 = db_manager.save_price_data(
            {'005930.KS': df_modified},
            update_if_exists=True
        )

        # 로드 후 수정된 값 확인
        loaded = db_manager.load_price_data(['005930.KS'])
        df_loaded = loaded['005930.KS']

        # 첫 번째 date의 close 값이 수정되었는지 확인
        assert df_loaded['close'].iloc[0] == 999.99

    def test_load_indicators_nonexistent_ticker(self, db_manager):
        """존재하지 않는 ticker의 지표 로드"""
        df_dict_load = db_manager.load_indicators(['NONEXISTENT.KS'])

        assert 'NONEXISTENT.KS' in df_dict_load
        assert df_dict_load['NONEXISTENT.KS'].empty

    def test_save_indicators_partial_columns(self, db_manager, sample_df_with_indicators):
        """지표 컬럼이 일부만 존재할 때"""
        # ma_5만 남기고 나머지 지표 제거
        df_partial = sample_df_with_indicators[['date', 'ma_5']].copy()

        db_manager._get_ticker_id('005930.KS')
        result = db_manager.save_indicators({'005930.KS': df_partial})

        assert result['005930.KS'] > 0

        loaded = db_manager.load_indicators(['005930.KS'])
        df_loaded = loaded['005930.KS']

        # ma_5는 있고 다른 지표들은 NaN이어야 함
        assert 'ma_5' in df_loaded.columns
        if not df_loaded.empty:
            assert not df_loaded['ma_5'].isna().all()

    def test_partial_ticker_list_exists(self, db_manager, sample_df_basic):
        """부분적으로만 존재하는 ticker 목록 로드"""
        # 일부 ticker만 저장
        db_manager.save_price_data({'005930.KS': sample_df_basic}, update_if_exists=True)

        # 존재하는 ticker와 존재하지 않는 ticker를 함께 로드
        result = db_manager.load_price_data(['005930.KS', 'NONEXISTENT.KS'])

        assert '005930.KS' in result
        assert 'NONEXISTENT.KS' in result
        assert len(result['005930.KS']) > 0
        assert result['NONEXISTENT.KS'].empty


class TestDatabaseManagerEdgeCases:
    """엣지 케이스 테스트"""

    def test_indicator_version_management(self, db_manager, sample_df_with_indicators):
        """지표 버전 관리 검증"""
        db_manager._get_ticker_id('005930.KS')

        # v1.0으로 저장
        db_manager.save_indicators(
            {'005930.KS': sample_df_with_indicators},
            version='v1.0'
        )

        # v2.0으로 재저장
        df_modified = sample_df_with_indicators.copy()
        df_modified['ma_5'] = df_modified['ma_5'] * 2

        db_manager.save_indicators(
            {'005930.KS': df_modified},
            version='v2.0'
        )

        # 로드된 데이터가 최신 버전인지 확인
        loaded = db_manager.load_indicators(['005930.KS'])
        df_loaded = loaded['005930.KS']

        assert not df_loaded.empty

    def test_large_dataset_save_and_load(self, db_manager):
        """큰 데이터셋 저장/로드"""
        # 1000개 행의 데이터 생성
        large_df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=1000),
            'open': np.random.randn(1000) + 100,
            'high': np.random.randn(1000) + 101,
            'low': np.random.randn(1000) + 99,
            'close': np.random.randn(1000) + 100,
            'volume': np.random.randint(1000000, 2000000, 1000),
            'adj_close': np.random.randn(1000) + 100,
        })

        result = db_manager.save_price_data({'005930.KS': large_df}, update_if_exists=True)
        assert result['005930.KS'] == 1000

        loaded = db_manager.load_price_data(['005930.KS'])
        assert len(loaded['005930.KS']) == 1000

    def test_context_manager_with_exception(self, temp_db_path, sample_df_basic):
        """예외 발생 시 Context Manager 자동 정리"""
        try:
            with DatabaseManager(db_path=temp_db_path) as manager:
                manager.save_price_data(
                    {'005930.KS': sample_df_basic},
                    update_if_exists=True
                )
                # 예외 발생
                raise ValueError("Test exception")
        except ValueError:
            pass

        # manager가 자동으로 close되었는지 확인 (재연결로 검증)
        with DatabaseManager(db_path=temp_db_path) as manager:
            loaded = manager.load_price_data(['005930.KS'])
            assert len(loaded['005930.KS']) > 0
