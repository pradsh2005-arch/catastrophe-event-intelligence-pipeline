import sqlite3
from pathlib import Path

import pandas as pd

from config import PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "final_catastrophe_event_dataset.csv"
DATABASE_FILE = PROCESSED_DIR / "cat_events.db"
QUERY_OUTPUT_DIR = PROCESSED_DIR / "sql_query_outputs"

QUERY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_final_dataset() -> pd.DataFrame:
    """
    Load the final enriched catastrophe event dataset.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing final dataset: {INPUT_FILE}. Run src/run_pipeline.py first."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print(f"Loaded final dataset: {len(df)} rows, {len(df.columns)} columns")

    return df


def clean_for_sqlite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe values so they store cleanly in SQLite.

    SQLite can store strings, numbers and nulls, but not Python lists directly.
    Some NLP columns may contain list-like values, so we convert object columns
    safely to strings where needed.
    """

    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict, tuple, set)) else x
            )

    return df


def export_events_table(df: pd.DataFrame, connection: sqlite3.Connection) -> None:
    """
    Export the final event dataset into a SQLite events table.
    """

    df.to_sql(
        name="events",
        con=connection,
        if_exists="replace",
        index=False,
    )

    print("Created table: events")


def create_summary_tables(connection: sqlite3.Connection) -> None:
    """
    Create useful SQL summary tables/views for quick analysis.
    """

    cursor = connection.cursor()

    cursor.execute("DROP VIEW IF EXISTS exposure_zone_summary;")
    cursor.execute(
        """
        CREATE VIEW exposure_zone_summary AS
        SELECT
            ils_exposure_zone,
            COUNT(*) AS event_count,
            AVG(cat_risk_score) AS avg_risk_score,
            SUM(CASE WHEN cat_risk_bucket = 'High' THEN 1 ELSE 0 END) AS high_risk_count
        FROM events
        GROUP BY ils_exposure_zone
        ORDER BY event_count DESC;
        """
    )

    cursor.execute("DROP VIEW IF EXISTS source_summary;")
    cursor.execute(
        """
        CREATE VIEW source_summary AS
        SELECT
            source,
            COUNT(*) AS event_count,
            AVG(cat_risk_score) AS avg_risk_score,
            SUM(CASE WHEN cat_risk_bucket = 'High' THEN 1 ELSE 0 END) AS high_risk_count
        FROM events
        GROUP BY source
        ORDER BY event_count DESC;
        """
    )

    cursor.execute("DROP VIEW IF EXISTS high_risk_events;")
    cursor.execute(
        """
        CREATE VIEW high_risk_events AS
        SELECT
            source,
            event_type,
            event_name,
            region,
            country,
            cat_risk_score,
            cat_risk_bucket,
            ils_exposure_zone,
            regex_wind_speed_mph,
            regex_precipitation_inches,
            fatalities,
            injuries,
            economic_damage_usd,
            description
        FROM events
        WHERE cat_risk_bucket = 'High'
        ORDER BY cat_risk_score DESC;
        """
    )

    connection.commit()

    print("Created views: exposure_zone_summary, source_summary, high_risk_events")


def run_query(
    connection: sqlite3.Connection,
    query_name: str,
    query: str,
    output_filename: str,
) -> pd.DataFrame:
    """
    Run a SQL query, print the results and save them to CSV.
    """

    print("\n" + "=" * 80)
    print(query_name)
    print("=" * 80)

    result = pd.read_sql_query(query, connection)

    print(result.head(20).to_string(index=False))

    output_path = QUERY_OUTPUT_DIR / output_filename
    result.to_csv(output_path, index=False)

    print(f"\nSaved query output to {output_path}")

    return result


def run_example_queries(connection: sqlite3.Connection) -> None:
    """
    Run example SQL queries that demonstrate database/query capability.
    """

    query_1 = """
    SELECT
        ils_exposure_zone,
        COUNT(*) AS event_count,
        ROUND(AVG(cat_risk_score), 2) AS avg_risk_score,
        SUM(CASE WHEN cat_risk_bucket = 'High' THEN 1 ELSE 0 END) AS high_risk_count
    FROM events
    GROUP BY ils_exposure_zone
    ORDER BY event_count DESC;
    """

    run_query(
        connection=connection,
        query_name="Query 1: Events by ILS exposure zone",
        query=query_1,
        output_filename="events_by_exposure_zone.csv",
    )

    query_2 = """
    SELECT
        source,
        COUNT(*) AS event_count,
        SUM(CASE WHEN cat_risk_bucket = 'High' THEN 1 ELSE 0 END) AS high_risk_count,
        ROUND(AVG(cat_risk_score), 2) AS avg_risk_score
    FROM events
    GROUP BY source
    ORDER BY event_count DESC;
    """

    run_query(
        connection=connection,
        query_name="Query 2: Events and high-risk counts by source",
        query=query_2,
        output_filename="events_by_source.csv",
    )

    query_3 = """
    SELECT
        event_name,
        region,
        cat_risk_score,
        cat_risk_bucket,
        regex_wind_speed_mph,
        regex_precipitation_inches,
        fatalities,
        injuries,
        description
    FROM events
    WHERE cat_risk_bucket = 'High'
      AND ils_exposure_zone = 'US Winter Storm'
    ORDER BY cat_risk_score DESC
    LIMIT 25;
    """

    run_query(
        connection=connection,
        query_name="Query 3: High-risk US Winter Storm events",
        query=query_3,
        output_filename="high_risk_us_winter_storm_events.csv",
    )

    query_4 = """
    SELECT
        event_type,
        COUNT(*) AS event_count,
        ROUND(AVG(cat_risk_score), 2) AS avg_risk_score,
        MAX(cat_risk_score) AS max_risk_score
    FROM events
    GROUP BY event_type
    ORDER BY max_risk_score DESC, event_count DESC
    LIMIT 30;
    """

    run_query(
        connection=connection,
        query_name="Query 4: Event types ranked by maximum risk score",
        query=query_4,
        output_filename="event_types_by_risk.csv",
    )

    query_5 = """
    SELECT
        source,
        event_type,
        event_name,
        region,
        country,
        cat_risk_score,
        cat_risk_bucket,
        ils_exposure_zone,
        regex_wind_speed_mph,
        regex_precipitation_inches,
        fatalities,
        injuries,
        economic_damage_usd
    FROM events
    ORDER BY cat_risk_score DESC
    LIMIT 50;
    """

    run_query(
        connection=connection,
        query_name="Query 5: Top 50 events by risk score",
        query=query_5,
        output_filename="top_50_events_by_risk_score.csv",
    )


def print_database_summary(connection: sqlite3.Connection) -> None:
    """
    Print a small database summary.
    """

    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM events;")
    row_count = cursor.fetchone()[0]

    cursor.execute("PRAGMA table_info(events);")
    columns = cursor.fetchall()

    cursor.execute(
        """
        SELECT name, type
        FROM sqlite_master
        WHERE type IN ('table', 'view')
        ORDER BY type, name;
        """
    )
    objects = cursor.fetchall()

    print("\n" + "=" * 80)
    print("SQLite database summary")
    print("=" * 80)
    print(f"Database file: {DATABASE_FILE}")
    print(f"Events table rows: {row_count}")
    print(f"Events table columns: {len(columns)}")

    print("\nTables/views:")
    for name, object_type in objects:
        print(f"- {object_type}: {name}")


def main() -> None:
    df = load_final_dataset()
    df = clean_for_sqlite(df)

    connection = sqlite3.connect(DATABASE_FILE)

    try:
        export_events_table(df, connection)
        create_summary_tables(connection)
        run_example_queries(connection)
        print_database_summary(connection)
    finally:
        connection.close()

    print("\nSQLite export complete.")


if __name__ == "__main__":
    main()