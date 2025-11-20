# 🚀 ML 퀀트 트레이딩 시스템

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

> **절대 모멘텀 전략 기반 한국 주식 자동매매 시스템 (개발 중)**

대학생 3명이 처음 만드는 머신러닝 퀀트 트레이딩 토이 프로젝트

---

## 📋 프로젝트 개요

**투자 전략**: 절대 모멘텀 (Absolute Momentum)
- 기술적 지표(이동평균선, MACD)로 모멘텀 포착
- 머신러닝(Logistic Regression)으로 매수/매도 예측

**개발 목표**
1. 데이터 수집 및 저장 자동화
2. 기술 지표 계산 및 피처 생성
3. ML 모델 학습 및 백테스팅
4. 키움증권 API 연동 실전 매매

**팀**: 경제학 + 데이터 사이언스 + 머신러닝 지망 대학생

---

## 🏗️ 시스템 구조

```
yfinance (주가 수집)
    ↓
SQLite (OHLCV + 기술지표 저장)
    ↓
Feature Engineering (피처 생성)
    ↓
ML Model (모멘텀 예측)
    ↓
Backtrader (백테스팅)
    ↓
키움증권 API (실전 매매)
```

---

## 🛠️ 기술 스택

| 분야 | 기술 |
|------|------|
| **데이터 수집** | yfinance |
| **데이터베이스** | SQLite + SQLAlchemy |
| **기술 지표** | pandas-ta-classic |
| **머신러닝** | scikit-learn |
| **백테스팅** | Backtrader |
| **실전 매매** | 키움증권 API |
| **개발 환경** | Docker |

---

## 📂 프로젝트 구조

```
src/
├── data/              # 데이터 수집 및 처리
│   ├── data_fetcher.py       # yfinance 주가 수집
│   ├── db_manager.py         # SQLite 관리
│   ├── indicator_calculator.py  # 기술 지표 계산
│   └── pipeline.py           # 자동화 파이프라인
├── models/            # ML 모델 (개발 예정)
├── backtest/          # 백테스팅 (개발 예정)
└── execution/         # 실전 매매 (개발 예정)
```

---

## 🚀 빠른 시작

### Docker로 실행
```bash
git clone https://github.com/your-team/Quant-Trading.git
cd Quant-Trading/docker
cp .env.example .env
docker-compose up --build

## 📝 개발 진행 상황

### ✅ 완료
- Docker 환경 구축

### 🔄 진행 중
- yfinance 데이터 수집 모듈
- SQLite 데이터베이스 (OHLCV, 기술지표)
- 기술 지표 계산 (MA, MACD, RSI, Bollinger Bands)
- 자동화 데이터 파이프라인

### 🔜 예정
- 피처 엔지니어링
- 모멘텀 레이블링
- 백테스팅 (거래 비용, 성과 지표)
- 키움증권 API 연동
- 자동매매 시스템

---

## 👥 팀원

경제 전문가 | 데이터 사이언티스트 | 머신러닝 엔지니어

---

<div align="center">

**Built by College Students Learning Quant Trading 📈**

</div>
