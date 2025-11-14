import sqlite3, pathlib

root = pathlib.Path(__file__).resolve().parents[1]
db_path = root / "data" / "market.db"
sql_path = (root / "src" / "schema.sql").read_text(encoding="utf-8")

# db.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(db_path) as conn:
    conn.executescript(sql_path)

print("OK:", db_path)
