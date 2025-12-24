# 🚀 ML 퀀트 트레이딩 시스템

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/GHW-team/Quant-Trading/workflows/Tests/badge.svg)](https://github.com/GHW-team/Quant-Trading/actions)
[![Code Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/GHW-team/Quant-Trading)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

> **절대 모멘텀 전략 기반 머신러닝 한국 주식 트레이딩 시스템**
> *대학생 3명이 처음부터 구축하는 퀀트 트레이딩 학습 프로젝트*

---

## 📋 프로젝트 개요

### 팀 구성
경제 전문가 + 데이터 사이언티스트 + 머신러닝 엔지니어 지망 대학생 3명

### 프로젝트 목표
간단하지만 **구조적으로 완전한** 머신러닝 퀀트 트레이딩 시스템 구축을 통한 실전 능력 배양

### 투자 전략
- **전략**: 절대 모멘텀 (Absolute Momentum)
- **대상**: 한국 주식 시장 (KOSPI)
- **방법**: 19개 기술적 지표 기반 머신러닝 예측
- **모델**: Logistic Regression (향후 LightGBM/XGBoost 확장 예정)

---

## ✨ 핵심 기능

### 🎯 현재 구현 완료
- ✅ **Production-ready 데이터 파이프라인** (병렬 처리, 자동 재시도, 속도 제한)
- ✅ **19개 기술적 지표 자동 계산** (MA, MACD, RSI, Bollinger Bands, ATR, Stochastic 등)
- ✅ **SQLite 기반 데이터 관리** (3개 테이블, SQLAlchemy ORM)
- ✅ **머신러닝 모델 학습** (GridSearchCV + TimeSeriesSplit)
- ✅ **백테스팅 프레임워크** (Backtrader 통합, 성과 지표 계산)
- ✅ **포괄적 테스트 커버리지** (95%+, 8,620줄 테스트 코드)
- ✅ **CI/CD 파이프라인** (GitHub Actions)

### 🔜 다음 단계
- ⏳ 키움증권 API 연동 (실전 매매)
- ⏳ LightGBM/XGBoost 모델 추가
- ⏳ 고급 피처 엔지니어링
- ⏳ 자동 스케줄링 및 알림 시스템

---

## 🏗️ 시스템 아키텍처

6대 계층으로 구성된 완전한 퀀트 트레이딩 시스템

```
┌─────────────────────────────────────────────────────────────┐
│  1. 데이터 계층 (Data Layer)                      ✅ 완성    │
│     yfinance → SQLite → 19개 기술지표 계산                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 전략/모델링 계층 (Strategy & Modeling)        ⚠️  기본   │
│     절대 모멘텀 전략 → 피처 생성 → ML 학습                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 백테스팅 계층 (Backtesting)                  ✅ 작동     │
│     Backtrader → 성과 분석 (Sharpe, MDD, 승률)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. 포트폴리오 관리 (Portfolio & Risk)            ⚠️  부분   │
│     동일 가중 포지션 사이징 → 기본 Stop-Loss                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  5. 실행 계층 (Order Execution)                   ❌ 예정    │
│     키움증권 API → 시장가 주문 → 보안 관리                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  6. 자동화/모니터링 (Automation)                  ❌ 예정    │
│     스케줄링 → 로깅 → 알림 시스템                           │
└─────────────────────────────────────────────────────────────┘
```

**범례**: ✅ 완성 | ⚠️ 부분 완료 | ❌ 미구현

---

## 🛠️ 기술 스택

| 계층 | 기술 | 상태 |
|------|------|------|
| **데이터 수집** | yfinance | ✅ |
| **데이터베이스** | SQLite + SQLAlchemy ORM | ✅ |
| **기술 지표** | pandas-ta-classic (19개 지표) | ✅ |
| **머신러닝** | scikit-learn (Logistic Regression) | ✅ |
| **ML (계획)** | LightGBM, XGBoost | ⏳ |
| **백테스팅** | Backtrader | ✅ |
| **실전 매매** | 키움증권 API | ⏳ |
| **테스팅** | pytest, pytest-cov (95%+ 커버리지) | ✅ |
| **코드 품질** | black, flake8, isort | ✅ |
| **CI/CD** | GitHub Actions | ✅ |
| **개발 환경** | Docker + docker-compose | ✅ |

---

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone https://github.com/your-org/Quant-Trading.git
cd Quant-Trading
```

### 2. Docker로 실행 (권장)
```bash
cd docker
cp .env.example .env  # 환경 변수 설정
docker-compose up --build
```

### 3. 로컬 환경 설정
```bash
# Python 3.11 필요
pip install -r requirements.txt

# 데이터 파이프라인 실행
python scripts/run_pipeline.py

# 모델 학습
python scripts/train_model.py

# 백테스트 실행
python scripts/run_backtest.py
```

---

## 📂 프로젝트 구조

```
Quant-Trading/
├── src/                         # 소스 코드 (3,286줄)
│   ├── data/                    # 데이터 계층 ✅
│   │   ├── data_fetcher.py     #   yfinance 데이터 수집 (병렬 처리, 재시도 로직)
│   │   ├── db_manager.py       #   SQLite CRUD 관리
│   │   ├── db_models.py        #   데이터베이스 스키마 (3 테이블)
│   │   ├── indicator_calculator.py  #   19개 기술 지표 계산
│   │   ├── pipeline.py         #   End-to-end 데이터 파이프라인
│   │   ├── all_ticker.py       #   티커 유니버스 관리 (KOSPI)
│   │   └── visualized_db.py    #   데이터 시각화 유틸
│   ├── ml/                      # ML 계층 ⚠️
│   │   ├── labeler.py          #   모멘텀 레이블링
│   │   └── logistic_regression.py  #   모델 학습 & 예측
│   └── backtest/                # 백테스팅 계층 ✅
│       ├── strategy.py         #   ML 신호 기반 전략
│       ├── runner.py           #   백테스트 실행 엔진
│       ├── data_feed.py        #   Backtrader 데이터 피드
│       └── analyzer.py         #   성과 지표 분석
├── tests/                       # 테스트 코드 (8,620줄, 95%+ 커버리지)
│   ├── unit/                   #   단위 테스트 (7 파일)
│   ├── integration/            #   통합 테스트 (3 파일)
│   └── fixtures/               #   테스트 유틸리티
├── scripts/                     # CLI 실행 스크립트
│   ├── run_pipeline.py         #   데이터 파이프라인 CLI
│   ├── train_model.py          #   모델 학습 CLI
│   └── run_backtest.py         #   백테스트 CLI
├── config/                      # YAML 설정 파일
│   ├── config.yaml             #   마스터 설정
│   ├── run_pipeline.yaml       #   파이프라인 설정
│   └── backtest.yaml           #   백테스트 설정
├── docker/                      # Docker 환경
│   ├── Dockerfile
│   └── docker-compose.yml
├── notebooks/                   # Jupyter 노트북 (분석/실험)
├── docs/                        # 문서
└── data/                        # 데이터 저장소
    ├── trading.db              #   SQLite 데이터베이스
    └── backtest_results/       #   백테스트 결과
```

---

## 📊 개발 현황

### ✅ 완료 (Production-Ready)
- [x] **데이터 계층**
  - yfinance 통합 (병렬 처리, 재시도, 속도 제한)
  - SQLite 데이터베이스 (Ticker, DailyPrice, TechnicalIndicator)
  - 19개 기술 지표 자동 계산
  - 시장 달력 검증
- [x] **백테스팅 프레임워크** 
  - Backtrader 통합
  - 성과 지표 (Sharpe Ratio, Max Drawdown, Win Rate 등)
  - 수수료/슬리피지 시뮬레이션
  - 거래 내역 로깅
- [x] **ML 모델 학습** 
  - Logistic Regression
  - TimeSeriesSplit 교차 검증
  - 모델 직렬화 (joblib)
- [x] **테스트 인프라**
  - 20개 테스트 모듈 
  - Unit + Integration 테스트
  - GitHub Actions CI/CD

### ⚠️ 부분 완료
- [ ] **포트폴리오 관리** (40%)
  - ✅ 동일 가중 포지션 사이징
  - ✅ 기본 Stop-Loss
  - ❌ 고급 리스크 관리
  - ❌ 포트폴리오 최적화
- [ ] **피처 엔지니어링** (60%)
  - ✅ 기본 기술 지표 (11개 피처)
  - ❌ 고급 복합 지표
  - ❌ 볼륨 기반 피처

### ❌ 예정
- [ ] **실행 계층** (0%)
  - 키움증권 API 통합
  - 주문 실행 로직
  - API 키 보안 관리
- [ ] **자동화/모니터링** (0%)
  - 스케줄링 (일일 자동 실행)
  - 알림 시스템 (Slack/Email)
  - 로깅 대시보드
- [ ] **고급 ML 모델** (0%)
  - LightGBM 구현
  - XGBoost 구현
  - 모델 앙상블

---

## 🧪 테스트 및 품질

### 테스트 커버리지
```bash
# 전체 테스트 실행
pytest

# 커버리지 리포트
pytest --cov=src --cov-report=html

# 특정 마커만 실행
pytest -m unit          # 단위 테스트만
pytest -m integration   # 통합 테스트만
pytest -m "not slow"    # 빠른 테스트만
```

### 통계
- **소스 코드**: 3,286줄 (15개 모듈)
- **테스트 코드**: 8,620줄 (20개 모듈)
- **테스트 비율**: 2.6:1 (업계 표준 초과)
- **커버리지**: 95%+

---

## 🎯 다음 단계 (우선순위)

### 🔴 High Priority
1. **LightGBM/XGBoost 모델 구현** - 성과 향상의 핵심
2. **키움증권 API 연동** - 실전 투자 준비
3. **고급 피처 엔지니어링** - 복합 모멘텀 지표, 볼륨 분석

### 🟡 Medium Priority
4. **모델 평가 고도화** - Feature Importance, SHAP 분석
5. **리스크 관리 강화** - 포트폴리오 레벨 제어
6. **문서화** - API 문서, 아키텍처 가이드

### 🟢 Low Priority
7. **성능 최적화** - 백테스트 병렬 처리
8. **시각화 도구** - 거래 대시보드
9. **알림 시스템** - 에러/성과 알림

---

이 프로젝트는 실전 투자 시 발생하는 손실에 대해 책임지지 않습니다.

---

<div align="center">

### 👥 팀원

**경제 전문가** | **데이터 사이언티스트** | **머신러닝 엔지니어**

---

**Built by College Students Learning Quant Trading 📈**

*"최종 목표는 고도화된 성능 좋은 ML 퀀트 시스템 구축,
현재는 그를 위한 능력을 키우는 첫 번째 토이 프로젝트"*

</div>
