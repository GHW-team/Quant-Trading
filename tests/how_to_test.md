# 테스트 이론 종합 가이드

> ML 퀀트 트레이딩 시스템 프로젝트

---

## 목차
1. [V-Model](#1-v-model)
2. [테스트 레벨 (ISTQB 4단계)](#2-테스트-레벨-istqb-4단계)
3. [테스트 설계 기법](#3-테스트-설계-기법)
4. [테스트 방법론](#4-테스트-방법론)
5. [테스트 작성 패턴](#5-테스트-작성-패턴)
6. [통합 구분](#6-통합-구분)
7. [커버리지 목표](#7-커버리지-목표)
8. [실전 적용 전략](#8-실전-적용-전략)

---

## 1. V-Model

### 핵심 개념
- **개발 단계와 테스트 단계가 1:1 대응**
- 왼쪽(개발)과 오른쪽(테스트)이 연결됨

### 구조 예시
```
요구사항 정의    ↔  인수 테스트
시스템 설계      ↔  시스템 테스트
상세설계 설계    ↔  통합 테스트
코딩 설계        ↔  단위 테스트
       ↓ 구현 완료 ↑
```

### 프로젝트 적용
- 데이터 수집 설계 시 → 단위 통합 테스트 준비
- Labeler 클래스 설계 시 → 단위 테스트 준비
- 전체 파이프라인 요구사항 → 인수 테스트 기준 정의

---

## 2. 테스트 레벨 (ISTQB 4단계)

### 2.1 단위 테스트 (Component Testing)
- **범위**: 개별 함수/클래스/모듈
- **담당자**: 개발자
- **특징**: 빠름, 독립 실행 가능
- **예시**: `test_labeler.py`, `test_indicator_calculator.py`

### 2.2 통합 테스트 (Integration Testing)
- **범위**: 컴포넌트 간 상호작용
- **담당자**: 개발자/QA
- **특징**: 데이터베이스 연결, 일부 Mock 사용
- **예시**: `test_pipeline_full.py` (Fetcher + DB + Calculator)

### 2.3 시스템 테스트 (System Testing)
- **범위**: 전체 시스템 E2E
- **담당자**: QA
- **특징**: 실제 환경, Mock 최소화
- **예시**: 실제 수집 → 지표 계산 (데이터 수집 → 지표 DB 저장 → 모델 학습 → ML)

### 2.4 인수 테스트 (Acceptance Testing)
- **범위**: 사용자 요구사항
- **담당자**: 사용자/PM
- **특징**: 비즈니스 시나리오, UAT
- **예시**: "삼성전자 최근 전체기간 데이터 15% 상승 예측 확인"

---

## 3. 테스트 설계 기법

### 3.1 명세 기반 테스트
**목적**: 요구 작성 기반, 블랙박스 방식

**주요 기법**:

#### 동등 분할 (Equivalence Partitioning)
- 입력/출력 값을 유효/무효로 분류
- 대표값 하나만 선택해 테스트
- 예: `period` 값의 유효('1y', '6m'), 무효('invalid', '-1y')

#### 경계값 분석 (Boundary Value Analysis)
- 경계 근처값 집중 테스트
- 예: `threshold` 값 (0, 0.01, -0.01, 100)

#### 결정 테이블 (Decision Table)
- 조건 조합 테스트
- 예: (tickers=None, exchanges=None) 시 Error

### 3.2 구조 기반 테스트
**목적**: 코드 구조 작성 기준 화이트박스 테스트

**주요 기법**:
- **Statement Coverage**: 모든 구문이 최소 한 번 실행 (**80% 목표**)
- **Branch Coverage**: 모든 if/else 분기 실행 (**70% 목표**)
- **Path Coverage**: 모든 경로 완주 (현실적으로 매우 어려움)

**도구**: `pytest --cov=src --cov-report=html`

### 3.3 경험 기반 테스트
**목적**: 개발자 직관으로 엣지케이스 찾기

**예시**:
- DB 컬럼 타입 edge case 테스트 (NULL, 빈 값)
- API 작성 타입 병렬 테스트 (max_workers=0)

---

## 4. 테스트 방법론

### 4.1 TDD (Test-Driven Development)

**프로세스 (Red-Green-Refactor)**:
```
1. RED (빨강)
    먼저(먼저) 테스트를 작성 → 실패
    아직 구현 코드가 없으므로 당연
    "무엇을 만들지 명확히 정의" 단계

2. GREEN (녹색)
    테스트를 통과시킬 최소한의 코드 작성
    일단 빠르게 동작만 (품질 고려 OK)
    "작동하게 만들기"

3. REFACTOR (리팩토링)
    코드 품질 향상
    중복 제거, 가독성 향상
    테스트는 그대로 통과해야 함

※ 이후 반복: RED → GREEN → REFACTOR → ...
```

**장점 강조**:
- 요구 기능만 개발 (전체 개발 시간 단축)
- 요구사항 변경에 유연
- 리팩토링 자신감 확보

**단점**: 러닝 커브 높음, 설계 변경 시, 리팩토링 비용

### 4.2 BDD (Behavior-Driven Development)

**프로세스**:
```gherkin
Given [조건]
When [동작]
Then [결과]
```

**도구**: `pytest-bdd`, Gherkin

**장점 강조**:
- 사용자 친화적 용어
- 요구사항 공유 용이

**단점 추가**: 러닝 커브 (초보 팀 부적합)

### 4.3 Mutation Testing

**목적**: 테스트의 강건성 검증

**방식**:
1. 코드에 고의로 버그 생성 (Mutation)
2. 테스트 실행
3. 테스트 통과 실패(실패 OK, 통과시 테스트 품질 낮음)

**도구**: `mutmut`

**장점 강조**:
- 높은 신뢰도 (코드 변화, 버그 감지)
- 테스트 커버리지 허점 탐색 가능 준비

### 4.4 Property-Based Testing

**목적**: 불변성(invariant) 검증

**방식**:
- 무작위 값으로 자동 생성 케이스 실행
- Hypothesis 라이브러리로 edge case 발견

**도구**: `Hypothesis`

**예시 코드**:
```python
@given(st.floats(min_value=-1.0, max_value=10.0))
def test_returns_always_above_minus_100_percent(return_value):
    assert return_value >= -1.0  # 수익률 하한선
```

**적용 강조**:
- 수익률 검증 (예: -100% ~ +무한대 범위)
- 비중 검증 (예: 0 이상 1)

---

### 4가지 방법론 비교표

| 방법론 | 주요 목적 | 적용 | 장점 | 적용 시점 |
|--------|------|------|------|-----------|
| TDD | 설계 주도   | 구조화 설계, 공유 문서 | 기능 중심, 빠른 개발  | 요구 기능 개발 |
| BDD | 요구사항 중심 | 사용자 요구, 협업 | 가독 향상, 설계력 | 비즈니스 시나리오 |
| Mutation | 테스트 품질 검증 | 완성 테스트 강화 | 높은 신뢰, 버그 | 코드 완성도 검증 |
| Property-Based | 경계 조건 자동 탐색 | 수치형, 조합 | 불변성 확인 자동화 | 수익 신뢰도 검증 |

---

### ML 퀀트 트레이딩 시스템 방법론 적용 예시

```

            ML 퀀트 트레이딩 시스템 방법론 적용 예시


1. 데이터 계층
    TDD: 스키마 데이터 수집 검증 테스트
    BDD: 데이터 수집 시나리오
    Mutation: DB 조회/저장 로직
    Property-Based: 날짜 범위 검증

2. ML 계층
    TDD: 비중 로직 검증
    BDD: 예측 결과 시나리오
    Mutation: 학습 알고리즘
    Property-Based: 제약조건, 조합 비중

3. 전략
    TDD: 전략 코드 검증 범위
    BDD: 전략 시나리오 (예: 듀얼모멘텀)
    Mutation: 결과 로직
    Property-Based: 비즈니스 룰

4. 백테스팅 계층
    TDD: 계산 로직 검증
    BDD: 계산 시나리오
    Mutation: 슬리피지 로직 (예: 듀얼모멘텀)
    Property-Based: 수익률 제약
```

---

## 5. 테스트 작성 패턴

### 5.1 AAA 패턴 (Arrange-Act-Assert)

```python
def test_example():
    # Arrange: 준비 단계 (데이터, Mock, Fixture)
    labeler = Labeler(horizon=5, threshold=0.02)
    data = pd.DataFrame({'close': [100, 102, 105]})

    # Act: 실행 (테스트 대상 함수 호출)
    result = labeler.label_data(data)

    # Assert: 검증 (결과 확인)
    assert 'label' in result.columns
```

**주의점**:
- **Act는 가능한 한 줄**: 여러 함수 호출은 통합 테스트에 적합
- **Assert는 구체적이고 명확**: 모호한 assert는 OK, 디버깅 어려워짐

### 5.2 FIRST 원칙

| 원칙 | 의미 | 적용 방법 |
|------|------|-----------|
| **F**ast | 빠른 실행 | Mock 사용, 최소 데이터, 외부 API 호출 제거 |
| **I**ndependent | 독립 실행 | Fixture 사용, 테스트 간 순서 의존 제거 |
| **R**epeatable | 반복 가능 실행 | 고정 시드, Mock 대체 API, 무작위 제거 |
| **S**elf-validating | 자동 검증 | assert 사용, 수동 확인 필요 제거 |
| **T**imely | 적시 작성 | TDD(코드 전) 또는 코드 작성 직후 |

---

## 6. 통합 구분

### 통합의 정의

**"의존성에 외부 컴포넌트"**를 사용하는가 여부:

#### 내부 컴포넌트 (우리가 만든 클래스)
```
StockDataFetcher     (데이터 수집 클래스)
IndicatorCalculator  (지표 범위 클래스)
Labeler              (비중 클래스)
DatabaseManager      (저장 클래스)
DataPipeline         (전체)
```

#### 외부 컴포넌트 (외부 시스템)
```
SQLite               (데이터베이스 엔진)
Yahoo Finance Server (외부도메인 호출 서버)
```

### 단위 vs 통합 테스트 구분

**핵심 원칙**: "외부 경계(Boundary)를 넘는가?"

#### 단위 테스트 (Unit Test)
- **개별 클래스(class)의 내부 로직만 검증**
- 예시: 지표계산 실행 어려움, 빠름
- 예:
  ```python
  def test_labeler_creates_labels(sample_df_basic):
      # Arrange
      labeler = Labeler(horizon=5, threshold=0.02)

      # Act
      result = labeler.label_data(sample_df_basic)

      # Assert
      assert 'label' in result.columns
  ```
  - 의존성 1개: `Labeler` 클래스
  - 외부 의존 없음, 메모리만 사용 완료

#### 통합 테스트 (Integration Test)
- **여러 컴포넌트 간 상호작용(class)이나, 외부 시스템(DB, API)과 통신**
- 예시: 실제 데이터 흐름 검증
- 예 1 - 내부 클래스 + 외부 시스템:
  ```python
  def test_db_saves_data(temp_db_path, sample_df_basic):
      # Arrange
      db = DatabaseManager(temp_db_path)

      # Act
      db.save_price_data({'005930.KS': sample_df_basic})

      # Assert
      loaded = db.load_price_data('005930.KS')
      assert len(loaded) == len(sample_df_basic)
  ```
  - 의존성 2개: `DatabaseManager` + `SQLite 엔진 시스템`
  - 외부 의존(파일 I/O 데이터베이스 파일 읽기/쓰기) → 이것이 통합 테스트

- 예 2 - 내부 클래스 + 내부 클래스:
  ```python
  def test_pipeline_full_workflow(temp_db_path):
      # Arrange
      pipeline = DataPipeline(db_path=temp_db_path)

      # Act (여러 컴포넌트 협력)
      result = pipeline.run_full_pipeline(...)

      # Assert
      assert result['saved'] > 0
  ```
  - 의존성 3개: `DataPipeline` + `StockDataFetcher` + `IndicatorCalculator`
  - 여러 클래스가 협업 흐름 검증 → 이것이 통합 테스트

### 프로젝트 예시

```
Labeler, IndicatorCalculator:
  - 입력을 받아 변환만 수행
  - 계산로직만 의존성이 없는 순수 어려움
  =→ 1개 의존성 = Unit Test

DatabaseManager:
  - 자체적으로 외부에 의존하는 SQLite를 호출 사용
  - 실제 DB 파일 사용: 2개 의존성 (클래스 + DB)
  =→ Integration Test
  - Mock/In-memory DB 사용 시: 1개 의존성
  =→ Unit Test (현실적 어려움 있음)

DataPipeline:
  - 다수 클래스를 호출 결합
  - 여러 컴포넌트 (수집 + 계산 + 저장)
  =→ Integration Test
```

---

## 7. 커버리지 목표

| 코드 유형   | Statement | Branch | 상황 |
|-----------|-----------|--------|------|
| 핵심 신뢰도 로직 | 80%+ | 70%+ | 학습, 결과, 버그 범위 |
| 일반 코드 | 60%+ | 50%+ | 데이터 수집, 수집 |
| 간단 유틸 | 40%+ | 30%+ | 로깅 어려움 |

**실행 방법**:
```bash
# 기본 커버리지
pytest --cov=src

# HTML 리포트 생성
pytest --cov=src --cov-report=html

# 이후 브라우저 htmlcov/index.html 열기
```

**주의사항**:
- 커버리지 높다고 테스트 품질 좋다는게 아님 단순
- 의미 없는 assert 제거 필요
- 100% 커버리지는 목표가 아님 (80% 달성이 현실 적정)

---

## 8. 실전 적용 전략

### 3단계 우선 순위 전략

#### 필수 (즉시 적용) 항목
1. **AAA 패턴** - 모든 테스트에 적용
2. **단위 테스트** - 핵심 신뢰도 로직, 50% 커버리지
3. **Fast + Independent** - FIRST 원칙 중 최소 2가지
4. **명세 기반 기법** - 동등 분할 경계값

#### 권장 (점진 도입) 항목
5. **통합 테스트** - 주요 수집 3가지
6. **구조 기반 커버리지 측정** - pytest-cov
7. **TDD** - 전략 로직 개발

#### 고급 옵션 (여력 시) 항목
8. **Property-Based Testing** - 수익률/비중 범위 검증
9. **Mutation Testing** - 핵심 신뢰도 로직

#### 보류 항목
- V-Model 전체 (프로젝트 완성 후)
- BDD (협업 팀 없는 경우)
- 시스템/인수 테스트 (외부 의뢰)

---

## 프로젝트 현황 평가

### 단위 테스트 작성 현황

```
tests/
  unit/
    test_labeler.py           단위 테스트 (OK)
    test_indicator_calc.py    단위 테스트 (OK)
    test_db_manager.py       단위 vs통합 논쟁 통합 테스트
    test_pipeline.py          단위 테스트 (OK)

  integration/
    test_pipeline_full.py     통합 테스트 (OK)
    test_train_model_full.py  통합 테스트 (OK)
```

### 개선 방향

1. **파일 이동**: `test_db_manager.py` 를 `integration/`
   - 실제 SQLite 엔진 사용 → 명확한 통합 테스트

2. **AAA 패턴 적용 확인**: 모든 테스트에 적용
   ```python
   def test_something():
       # Arrange
       ...
       # Act
       ...
       # Assert
       ...
   ```

3. **커버리지 측정**: `pytest-cov` 설치 및 실행
   ```bash
   pip install pytest-cov
   pytest --cov=src --cov-report=html
   ```

4. **느린 테스트 마킹**:
   ```python
   @pytest.mark.slow
   def test_full_pipeline():
       ...

   # 빠른 테스트만 실행: pytest -m "not slow"
   ```

---

## 빠른 참고용 체크리스트

### 단위 테스트 작성 전
- [ ] AAA 작성 준비됐나?
- [ ] Act 부분 1줄로 가능한가?
- [ ] Fast한가? (1초 이내)
- [ ] Independent한가? (순서 무관)
- [ ] Repeatable한가? (datetime.now() 사용 안함)

### 코드 품질 체크
- [ ] 단위/통합 구분이 명확한가?
- [ ] 명세 기반 테스트인가?
- [ ] 커버리지 50% 이상인가?
- [ ] Mock을 올바르게 사용?

### 최종 검증
- [ ] 단위 테스트 완료
- [ ] 핵심 신뢰도 로직 80% 커버리지
- [ ] 통합 테스트 완료
- [ ] 느린 테스트에 마크 적용 확인

---

## 추가 학습 참고 자료

- **ISTQB Foundation**: [https://www.istqb.org/](https://www.istqb.org/)
- **Pytest 공식 문서**: [https://docs.pytest.org/](https://docs.pytest.org/)
- **pytest-cov**: [https://pytest-cov.readthedocs.io/](https://pytest-cov.readthedocs.io/)
- **Hypothesis**: [https://hypothesis.readthedocs.io/](https://hypothesis.readthedocs.io/)
- **mutmut**: [https://mutmut.readthedocs.io/](https://mutmut.readthedocs.io/)

---

**문서 최종 업데이트**: 2025-12-16
**프로젝트**: ML 퀀트 트레이딩 시스템
