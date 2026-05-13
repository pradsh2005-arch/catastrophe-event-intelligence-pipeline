import sqlite3
import pandas as pd

DB_PATH = "data/processed/cat_events.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("\nRow count in events table:")
cur.execute("SELECT COUNT(*) FROM events")
print(cur.fetchone())

print("\nTables and views:")
cur.execute("""
SELECT name, type
FROM sqlite_master
WHERE type IN ('table', 'view')
ORDER BY type, name;
""")
print(cur.fetchall())

print("\nTop 10 high-risk events:")
query = """
SELECT
    event_name,
    region,
    cat_risk_score,
    cat_risk_bucket,
    ils_exposure_zone
FROM events
WHERE cat_risk_bucket = 'High'
ORDER BY cat_risk_score DESC
LIMIT 10;
"""

print(pd.read_sql_query(query, conn))

conn.close()