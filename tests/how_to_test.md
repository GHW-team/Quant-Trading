# 테스트 이론 핵심 정리

> 빠른 참고를 위한 치트시트

---

## 1. V-Model

### 핵심 개념
- **개발 단계와 테스트 단계 1:1 대응**
- 왼쪽(개발) → 오른쪽(테스트) 매핑

### 적용 방법
```
요구사항 분석    ↔  인수 테스트
시스템 설계      ↔  시스템 테스트
아키텍처 설계    ↔  통합 테스트
모듈 설계        ↔  단위 테스트
       ↓ 코딩 ↓
```

### 프로젝트 적용
- 데이터 파이프라인 설계 시 → 통합 테스트 계획
- Labeler 클래스 설계 시 → 단위 테스트 계획
- 백테스팅 시스템 요구사항 → 인수 테스트 기준

---

## 2. 테스트 레벨 (ISTQB 4단계)

### 2.1 단위 테스트 (Component Testing)
- **대상**: 개별 함수/메서드/클래스
- **책임**: 개발자
- **특징**: 빠름, 외부 의존성 없음
- **예시**: `test_labeler.py`, `test_indicator_calculator.py`

### 2.2 통합 테스트 (Integration Testing)
- **대상**: 컴포넌트 간 상호작용
- **책임**: 개발자/QA
- **특징**: 인터페이스 검증, 일부 Mock 사용
- **예시**: `test_pipeline_full.py` (Fetcher + DB + Calculator)

### 2.3 시스템 테스트 (System Testing)
- **대상**: 전체 시스템 E2E
- **책임**: QA
- **특징**: 실제 환경, Mock 최소화
- **예시**: 전체 파이프라인 실행 (데이터 수집 → DB → 지표 → 라벨링 → ML)

### 2.4 인수 테스트 (Acceptance Testing)
- **대상**: 비즈니스 요구사항
- **책임**: 고객/PM
- **특징**: 사용자 시나리오, UAT
- **예시**: "모멘텀 전략이 백테스트에서 15% 수익률을 달성한다"

---

## 3. 테스트 설계 기법

### 3.1 블랙박스 테스트
**관점**: 내부 구조 모름, 입출력만 봄

**주요 기법**:

#### 동등 분할 (Equivalence Partitioning)
- 유효/무효 입력 그룹으로 나눔
- 각 그룹에서 대표값 1개씩 테스트
- 예: `period` → 유효('1y', '6m'), 무효('invalid', '-1y')

#### 경계값 분석 (Boundary Value Analysis)
- 경계 근처 값 테스트
- 예: `threshold` → (0, 0.01, -0.01, 100)

#### 결정 테이블 (Decision Table)
- 조건 조합 테스트
- 예: (tickers=None, exchanges=None) → Error

### 3.2 화이트박스 테스트
**관점**: 내부 코드 구조 알고 테스트

**주요 기법**:
- **Statement Coverage**: 모든 코드 라인 실행 (**80% 목표**)
- **Branch Coverage**: 모든 if/else 분기 실행 (**70% 목표**)
- **Path Coverage**: 모든 실행 경로 (거의 측정 안 함)

**도구**: `pytest --cov=src --cov-report=html`

### 3.3 그레이박스 테스트
**관점**: 일부 내부 지식 활용

**예시**:
- DB 스키마 알고 edge case 테스트 (NULL, 중복 키)
- API 구조 알고 경계값 테스트 (max_workers=0)

---

## 4. 테스트 방법론

### 4.1 TDD (Test-Driven Development)

**프로세스**:
1. **RED**: 실패하는 테스트 작성
2. **GREEN**: 최소 코드로 통과
3. **REFACTOR**: 코드 개선

**적용 시기**:
- 새 기능 개발 시 (백테스팅 계층 구현 시)
- 요구사항이 명확할 때
- 복잡한 로직일 때

**장점**: 버그 조기 발견, 설계 개선, 리팩토링 안전

### 4.2 BDD (Behavior-Driven Development)

**프로세스**:
```gherkin
Given [전제조건]
When [행동]
Then [결과]
```

**도구**: `pytest-bdd`, Gherkin

**적용 시기**:
- 비개발자와 협업 시
- 요구사항 문서화 필요 시

**현재 프로젝트**: 스킵 (팀이 너무 작음)

### 4.3 Mutation Testing

**목적**: 테스트의 품질 검증

**방법**:
1. 코드에 버그 심기 (Mutation)
2. 테스트 실행
3. 테스트가 실패하면 OK, 통과하면 테스트 품질 문제

**도구**: `mutmut`

**적용 시기**:
- 핵심 로직 (수익률 계산, 리스크 계산)
- 테스트 커버리지 높은데 버그가 계속 나올 때

### 4.4 Property-Based Testing

**목적**: 속성(invariant) 검증

**방법**:
- 랜덤 입력으로 불변 속성 확인
- Hypothesis가 자동으로 edge case 생성

**도구**: `Hypothesis`

**적용 예시**:
```python
@given(st.floats(min_value=-1.0, max_value=10.0))
def test_returns_always_above_minus_100_percent(return_value):
    assert return_value >= -1.0  # 수익률 하한
```

**적용 시기**:
- 수익률 계산 (항상 -100% ~ +무한대)
- 라벨링 (항상 0 또는 1)

---

## 5. 테스트 구조 패턴

### 5.1 AAA 패턴 (Arrange-Act-Assert)

```python
def test_example():
    # Arrange: 준비 (데이터, Mock, Fixture)
    labeler = Labeler(horizon=5, threshold=0.02)
    data = pd.DataFrame({'close': [100, 102, 105]})

    # Act: 실행 (테스트 대상 함수 호출)
    result = labeler.label_data(data)

    # Assert: 검증 (결과 확인)
    assert 'label' in result.columns
```

**규칙**:
- **Act는 한 번만**: 여러 함수 호출은 통합 테스트의 신호
- **Assert는 하나의 개념만**: 여러 assert는 OK, 단 같은 개념이어야 함

### 5.2 FIRST 원칙

| 원칙 | 의미 | 적용 방법 |
|------|------|-----------|
| **F**ast | 빠르게 실행 | Mock 사용, 최소 데이터, 실제 API 호출 금지 |
| **I**ndependent | 독립적 실행 | Fixture 사용, 테스트 간 순서 무관하게 |
| **R**epeatable | 반복 가능 | 고정 날짜, Mock 외부 API, 랜덤 제거 |
| **S**elf-validating | 자체 검증 | assert 사용, 수동 로그 확인 금지 |
| **T**imely | 적시 작성 | TDD(코드 전) 또는 코드 작성 직후 |

---

## 커버리지 목표

| 코드 유형 | Statement | Branch | 비고 |
|-----------|-----------|--------|------|
| 핵심 로직 | 80%+ | 70%+ | 결제, 거래, 리스크 계산 |
| 일반 코드 | 60%+ | 50%+ | 데이터 처리, 파이프라인 |
| 유틸리티 | 40%+ | 30%+ | 헬퍼 함수 |

**측정 방법**:
```bash
# 기본 커버리지
pytest --cov=src

# HTML 리포트 생성
pytest --cov=src --cov-report=html

# 브라우저에서 htmlcov/index.html 열기
```

---


1.V-Model
	-요구사항 분석 - 시스템 설계 - 아키텍처 설계 - 모듈 설계 - 코딩
	- "개발 단계"와 "테스트 단계"가 1:1로 대응

2.테스트 레벨
	-단위 테스트
	-통합 테스트
	-시스템 테스트
	-인수 테스트

3.테스트 설계 기법
	-블랙박스
	-화이트박스
	-그레이박스

4.테스트 방법론
	-TDD
	-BDD
	-Mutation Testing
	-Property-Based Testing

5.구조 패턴
	-AAA
	-FIRST

위 테스트 이론을

1.일반적인 ml퀀트 시스템의 구조
2.현재 내 프로젝트 폴더(현재 구현된 내용 + 구현할 내용)의 구조

위 두가지에 대해 적용시키면 

어떻게 되는지 

구체적 예시를 여러개 들어서
아주 자세히 설명해줘.

5.1 4가지 방법론 비교표
방법론	목적	장점	단점	적용 시점
TDD	설계 개선	깔끔한 설계, 문서화 효과	초기 느림, 학습 곡선	새 기능 개발
BDD	요구사항 검증	비즈니스 언어, 협업	도구 복잡, 오버헤드	사용자 시나리오
Mutation	테스트 품질 검증	약한 테스트 발견	실행 느림, 리소스	코드 안정화 후
Property-Based	엣지 케이스 발견	자동화, 광범위	속성 정의 어려움	핵심 로직


┌──────────────────────────────────────────────────────────────┐
│            ML 퀀트 시스템 방법론 적용 전략                      │
└──────────────────────────────────────────────────────────────┘

1. 데이터 계층
   ├─ TDD: 새로운 데이터 소스 추가 시
   ├─ BDD: 데이터 파이프라인 워크플로우
   ├─ Mutation: DB 저장/로드 로직
   └─ Property-Based: 지표 계산 함수

2. ML 계층
   ├─ TDD: 라벨링 로직 개발
   ├─ BDD: 모델 학습 시나리오
   ├─ Mutation: 특징 엔지니어링
   └─ Property-Based: 정규화, 스케일링

3. 백테스팅
   ├─ TDD: 성능 메트릭 계산
   ├─ BDD: 백테스트 시나리오 (✨ 필수)
   ├─ Mutation: 거래 로직
   └─ Property-Based: 포트폴리오 제약

4. 실행 계층
   ├─ TDD: 주문 실행 로직
   ├─ BDD: 주문 워크플로우
   ├─ Mutation: 자금 관리 (✨ 필수)
   └─ Property-Based: 포지션 크기

┌──────────────────────────────────────────────────────────────┐
│                  TDD 사이클 (Red-Green-Refactor)              │
└──────────────────────────────────────────────────────────────┘

1. RED (빨강)
   ├─ 실패하는 테스트를 먼저 작성
   ├─ 아직 구현 코드가 없으므로 실패
   └─ "무엇을 만들 것인가" 정의

2. GREEN (녹색)
   ├─ 테스트를 통과하는 최소한의 코드 작성
   ├─ 일단 동작하게만 만듦 (지저분해도 OK)
   └─ "어떻게든 동작하게"

3. REFACTOR (리팩토링)
   ├─ 코드 품질 개선
   ├─ 중복 제거, 가독성 향상
   └─ 테스트는 계속 통과해야 함

반복 → 다음 기능 → RED → GREEN → REFACTOR → ...

=================================

1. 컴포넌트의 실체 (님의 프로젝트 기준)
님의 프로젝트에서 **"하나의 컴포넌트"**라고 부를 수 있는 것들은 이 네모 박스 하나하나입니다.

소프트웨어 컴포넌트 (내가 짠 클래스)

StockDataFetcher (데이터 수집 담당)

IndicatorCalculator (지표 계산 담당)

Labeler (라벨링 담당)

DatabaseManager (저장 담당)

DataPipeline (지휘자)

인프라 컴포넌트 (외부 시스템)

SQLite (데이터베이스 파일)

Yahoo Finance Server (인터넷 너머의 서버)