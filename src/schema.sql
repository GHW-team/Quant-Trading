PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- 1) 원천
CREATE TABLE IF NOT EXISTS securities (
  ticker     TEXT PRIMARY KEY,
  name       TEXT,
  exchange   TEXT
);

CREATE TABLE IF NOT EXISTS price_daily (
  ticker      TEXT NOT NULL,
  date        TEXT NOT NULL,
  open        REAL,
  high        REAL,
  low         REAL,
  close       REAL,
  adj_close   REAL,
  volume      INTEGER,
  PRIMARY KEY (ticker, date),
  FOREIGN KEY (ticker) REFERENCES securities(ticker)
);

-- 2) 피처 & 라벨
CREATE TABLE IF NOT EXISTS features_daily (
  ticker   TEXT NOT NULL,
  date     TEXT NOT NULL,
  ma_5     REAL,
  ma_20    REAL,
  ma_200   REAL,
  ret_1d   REAL,
  ret_5d   REAL,
  gap_o_c  REAL,
  version  TEXT DEFAULT 'v1',
  PRIMARY KEY (ticker, date, version),
  FOREIGN KEY (ticker) REFERENCES securities(ticker)
);

CREATE TABLE IF NOT EXISTS labels_daily (
  ticker    TEXT NOT NULL,
  date      TEXT NOT NULL,
  label     INTEGER NOT NULL,        -- 0/1
  rule      TEXT NOT NULL,           -- 예: 'ret_20d>0.05'
  horizon   INTEGER NOT NULL,        -- 예: 20
  version   TEXT DEFAULT 'v1',
  PRIMARY KEY (ticker, date, rule, horizon, version),
  FOREIGN KEY (ticker) REFERENCES securities(ticker)
);

/*
-- 3) 모델 & 예측
CREATE TABLE IF NOT EXISTS model_registry (
  model_id      TEXT PRIMARY KEY,    -- 'logreg_YYYYMMDD_hhmm'
  algo          TEXT NOT NULL,       -- 'logreg'
  created_at    TEXT NOT NULL,
  features_ver  TEXT NOT NULL,
  label_rule    TEXT NOT NULL,
  label_horizon INTEGER NOT NULL,
  train_start   TEXT NOT NULL,
  train_end     TEXT NOT NULL,
  params_json   TEXT,
  metrics_json  TEXT
);

CREATE TABLE IF NOT EXISTS predictions_daily (
  model_id  TEXT NOT NULL,
  ticker    TEXT NOT NULL,
  date      TEXT NOT NULL,           -- 시그널 기준일
  score     REAL NOT NULL,           -- p(y=1|x)
  signal    INTEGER NOT NULL,        -- 임계값 통과 여부(0/1)
  PRIMARY KEY (model_id, ticker, date),
  FOREIGN KEY (model_id) REFERENCES model_registry(model_id),
  FOREIGN KEY (ticker)   REFERENCES securities(ticker)
);

-- 4) 백테스트
CREATE TABLE IF NOT EXISTS backtests (
  bt_id        TEXT PRIMARY KEY,
  model_id     TEXT NOT NULL,
  universe     TEXT NOT NULL,        -- 'KR' 등
  start_date   TEXT NOT NULL,
  end_date     TEXT NOT NULL,
  cost_bps     REAL NOT NULL,
  tax_bps      REAL DEFAULT 0,
  rules_json   TEXT,
  metrics_json TEXT,
  FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
);

CREATE TABLE IF NOT EXISTS bt_trades (
  bt_id    TEXT NOT NULL,
  trade_id INTEGER NOT NULL,
  date     TEXT NOT NULL,
  ticker   TEXT NOT NULL,
  side     TEXT NOT NULL,            -- 'BUY'/'SELL'
  price    REAL NOT NULL,
  qty      REAL NOT NULL,
  fee      REAL DEFAULT 0,
  tax      REAL DEFAULT 0,
  pnl      REAL,
  PRIMARY KEY (bt_id, trade_id),
  FOREIGN KEY (bt_id)  REFERENCES backtests(bt_id),
  FOREIGN KEY (ticker) REFERENCES securities(ticker)
);

CREATE TABLE IF NOT EXISTS bt_daily_perf (
  bt_id     TEXT NOT NULL,
  date      TEXT NOT NULL,
  nav       REAL NOT NULL,
  gross_expo REAL,
  net_expo   REAL,
  drawdown   REAL,
  PRIMARY KEY (bt_id, date),
  FOREIGN KEY (bt_id) REFERENCES backtests(bt_id)
);

-- 5) 운영(옵션)
CREATE TABLE IF NOT EXISTS job_runs (
  job_id     TEXT NOT NULL,          -- 'fetch','features','train','predict','rebalance'…
  run_id     TEXT NOT NULL,          -- UUID/타임스탬프
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status     TEXT NOT NULL,          -- 'OK'|'FAIL'
  msg        TEXT,
  PRIMARY KEY (job_id, run_id)
);

CREATE TABLE IF NOT EXISTS alerts (
  alert_id   TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  level      TEXT NOT NULL,          -- 'INFO'|'WARN'|'ERROR'
  channel    TEXT NOT NULL,          -- 'telegram'|'email'…
  text       TEXT NOT NULL,
  sent_at    TEXT
);
*/