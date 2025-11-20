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
  PRIMARY KEY (ticker, date),
  FOREIGN KEY (ticker) REFERENCES securities(ticker)
);


CREATE TABLE IF NOT EXISTS labels_daily (
  ticker    TEXT NOT NULL,
  date      TEXT NOT NULL,
  label     INTEGER NOT NULL,        -- 0/1
  horizon   INTEGER NOT NULL,        -- 예: 20
  PRIMARY KEY (ticker, date, horizon),
  FOREIGN KEY (ticker) REFERENCES securities(ticker)
);
