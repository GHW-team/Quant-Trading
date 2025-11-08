# scripts/run_pipeline.py - 데이터 파이프라인 실행 예시

import sys
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pipeline import DataPipeline

# ============================================
# 로깅 설정
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# =====================================================
# Case 1: 기본값으로 자동 실행
# =====================================================
def case1_default():
    """
    .env 파일의 설정값으로 전체 파이프라인 실행

    실행:
        python scripts/run_pipeline.py --case 1
    """
    logger.info("=" * 70)
    logger.info("Case 1: 기본값으로 자동 실행")
    logger.info("=" * 70)

    pipeline = DataPipeline()
    results = pipeline.run_full_pipeline()

    logger.info("\n결과:")
    logger.info(f"Fetched: {results['summary']['total_fetched_records']} records")
    logger.info(f"Saved prices: {results['summary']['total_saved_prices']} records")
    logger.info(f"Saved indicators: {results['summary']['total_saved_indicators']} records")

    return results


# =====================================================
# Case 2: 커스텀 인자로 실행
# =====================================================
def case2_custom():
    """
    특정 종목, 기간, 지표로 실행

    실행:
        python scripts/run_pipeline.py --case 2
    """
    logger.info("=" * 70)
    logger.info("Case 2: 커스텀 인자로 실행")
    logger.info("=" * 70)

    pipeline = DataPipeline()
    results = pipeline.run_full_pipeline(
        ticker_list=['005930.KS', '000660.KS'],
        period='2y',
        interval='1d',
        indicators=['ma_5', 'ma_20', 'ma_200', 'macd'],
        update_if_exists=True
    )

    logger.info("\n결과:")
    logger.info(f"Fetched: {results['summary']['total_fetched_records']} records")
    logger.info(f"Saved prices: {results['summary']['total_saved_prices']} records")
    logger.info(f"Saved indicators: {results['summary']['total_saved_indicators']} records")

    return results


# =====================================================
# Case 3: 특정 날짜 범위로 실행 (백테스팅용)
# =====================================================
def case3_date_range():
    """
    특정 기간의 데이터로 지표 계산 (백테스팅용)

    실행:
        python scripts/run_pipeline.py --case 3
    """
    logger.info("=" * 70)
    logger.info("Case 3: 특정 날짜 범위로 실행 (백테스팅용)")
    logger.info("=" * 70)

    pipeline = DataPipeline()
    results = pipeline.run_full_pipeline(
        ticker_list=['005930.KS'],
        start_date='2024-01-01',
        end_date='2024-12-31',
        indicators=['ma_5', 'ma_20', 'ma_200', 'macd']
    )

    logger.info("\n결과:")
    logger.info(f"Fetched: {results['summary']['total_fetched_records']} records")
    logger.info(f"Saved prices: {results['summary']['total_saved_prices']} records")
    logger.info(f"Saved indicators: {results['summary']['total_saved_indicators']} records")

    return results


# =====================================================
# Case 4: 단일 종목만 처리
# =====================================================
def case4_single_ticker():
    """
    한 종목만 처리

    실행:
        python scripts/run_pipeline.py --case 4
    """
    logger.info("=" * 70)
    logger.info("Case 4: 단일 종목만 처리")
    logger.info("=" * 70)

    pipeline = DataPipeline()
    results = pipeline.run_full_pipeline(
        ticker_list=['005930.KS'],
        indicators=['ma_5', 'ma_20']
    )

    logger.info("\n결과:")
    logger.info(f"Fetched: {results['summary']['total_fetched_records']} records")
    logger.info(f"Saved prices: {results['summary']['total_saved_prices']} records")
    logger.info(f"Saved indicators: {results['summary']['total_saved_indicators']} records")

    return results


# =====================================================
# Case 5: 단계별 선택적 실행
# =====================================================
def case5_step_by_step():
    """
    필요한 단계만 선택해서 실행

    실행:
        python scripts/run_pipeline.py --case 5
    """
    logger.info("=" * 70)
    logger.info("Case 5: 단계별 선택적 실행 (Step 1, 2만 실행)")
    logger.info("=" * 70)

    pipeline = DataPipeline()
    results = pipeline.run_step_by_step(
        ticker_list=['005930.KS', '000660.KS'],
        step_config={
            'step1_fetch': True,       # Step 1 실행
            'step2_save': True,        # Step 2 실행
            'step3_4_indicators': False,  # Step 3-4 스킵
            'period': '1y',
            'interval': '1d',
        }
    )

    logger.info("\n결과:")
    logger.info(f"Step 1 - Fetched: {results.get('step1_fetch', {})}")
    logger.info(f"Step 2 - Saved prices: {results.get('step2_save', {})}")
    logger.info(f"Step 3-4 - Indicators: {results.get('step3_4_indicators', {})}")

    return results


# =====================================================
# Case 6: 지표만 계산 및 저장 (데이터는 기존 DB 사용)
# =====================================================
def case6_indicators_only():
    """
    기존 DB의 가격 데이터를 이용해 지표만 계산 및 저장

    실행:
        python scripts/run_pipeline.py --case 6
    """
    logger.info("=" * 70)
    logger.info("Case 6: 지표만 계산 및 저장 (Step 3-4만 실행)")
    logger.info("=" * 70)

    pipeline = DataPipeline()
    results = pipeline.run_step_by_step(
        ticker_list=['005930.KS', '000660.KS'],
        step_config={
            'step1_fetch': False,      # Step 1 스킵
            'step2_save': False,       # Step 2 스킵
            'step3_4_indicators': True,  # Step 3-4만 실행
            'indicators': ['ma_5', 'ma_20', 'ma_200', 'macd'],
        }
    )

    logger.info("\n결과:")
    logger.info(f"Step 3-4 - Saved indicators: {results.get('step3_4_indicators', {})}")

    return results


# =====================================================
# 사용 가능한 모든 케이스
# =====================================================
CASES = {
    '1': ('기본값으로 자동 실행', case1_default),
    '2': ('커스텀 인자로 실행', case2_custom),
    '3': ('특정 날짜 범위로 실행 (백테스팅용)', case3_date_range),
    '4': ('단일 종목만 처리', case4_single_ticker),
    '5': ('단계별 선택적 실행', case5_step_by_step),
    '6': ('지표만 계산 및 저장', case6_indicators_only),
}


def print_usage():
    """사용법 출력"""
    print("\n" + "=" * 70)
    print("데이터 파이프라인 실행 예시")
    print("=" * 70)
    print("\n사용법:")
    print("  python scripts/run_pipeline.py --case <번호>")
    print("  또는")
    print("  python scripts/run_pipeline.py <번호>")
    print("\n사용 가능한 케이스:")

    for case_num, (description, _) in CASES.items():
        print(f"  {case_num}: {description}")

    print("\n예시:")
    print("  python scripts/run_pipeline.py 1     # 기본값으로 실행")
    print("  python scripts/run_pipeline.py --case 2  # 커스텀 인자로 실행")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='데이터 파이프라인 실행 스크립트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python scripts/run_pipeline.py 1
  python scripts/run_pipeline.py --case 2
  python scripts/run_pipeline.py --help
        """
    )

    parser.add_argument(
        'case',
        nargs='?',
        default=None,
        help='실행할 케이스 번호 (1-6)'
    )
    parser.add_argument(
        '--case',
        dest='case_arg',
        help='실행할 케이스 번호 (--case 1 형식)'
    )

    args = parser.parse_args()

    # 케이스 번호 결정
    case_num = args.case_arg or args.case

    if not case_num:
        print_usage()
        sys.exit(0)

    if case_num not in CASES:
        print(f"❌ 존재하지 않는 케이스: {case_num}")
        print_usage()
        sys.exit(1)

    # 선택한 케이스 실행
    description, case_func = CASES[case_num]

    try:
        logger.info(f"\n실행: Case {case_num} - {description}")
        case_func()
        logger.info("✓ 파이프라인 완료")
    except Exception as e:
        logger.error(f"✗ 파이프라인 실패: {e}", exc_info=True)
        sys.exit(1)
