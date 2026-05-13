import pandas as pd
from datetime import datetime, timezone

from config import RAW_DIR, PROCESSED_DIR

from fetch_usgs import fetch_usgs_earthquakes
from fetch_gdacs import fetch_gdacs_events
from fetch_noaa import fetch_noaa_storm_events, find_latest_noaa_year

from normalise import COMMON_COLUMNS, validate_schema, align_schema
from regex_extract import add_regex_features
from nlp_extract import add_nlp_features
from risk_scoring import add_risk_scores
from exposure_tagging import add_exposure_zones


# -----------------------------
# Pipeline settings
# -----------------------------

NOAA_YEAR = "latest"  # Set to "latest" to auto-detect the most recent year with data
USGS_DAYS_BACK = 14
USGS_MIN_MAGNITUDE = 4.5

RUN_NOAA = True
RUN_NLP = True

# Keep this at 10,000 during development.
# Later you can set this to None for the full dataset.
MAX_NLP_ROWS = 10_000


def save_raw_sources(
    usgs_df: pd.DataFrame,
    gdacs_df: pd.DataFrame,
    noaa_df: pd.DataFrame | None = None,
    noaa_year: int | None = None,
) -> None:
    """
    Save raw source outputs to data/raw.
    """

    usgs_path = RAW_DIR / "usgs_earthquakes.csv"
    gdacs_path = RAW_DIR / "gdacs_events.csv"

    usgs_df.to_csv(usgs_path, index=False)
    gdacs_df.to_csv(gdacs_path, index=False)

    print(f"Saved USGS raw data to {usgs_path}")
    print(f"Saved GDACS raw data to {gdacs_path}")

    if noaa_df is not None:
        if noaa_year is None:
            noaa_year = "unknown"

        noaa_path = RAW_DIR / f"noaa_storm_events_{noaa_year}.csv"
        noaa_df.to_csv(noaa_path, index=False)
        print(f"Saved NOAA raw data to {noaa_path}")


def combine_sources(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Validate and combine source dataframes into the common catastrophe schema.
    """

    aligned_frames = []

    for df in frames:
        if df.empty:
            continue

        source_name = str(df["source"].iloc[0]) if "source" in df.columns else "UNKNOWN"

        validate_schema(df, source_name)
        aligned = align_schema(df)

        aligned_frames.append(aligned)

    combined = pd.concat(aligned_frames, ignore_index=True)

    combined = combined.drop_duplicates(
        subset=["source", "event_id"],
        keep="last"
    )

    return combined


def save_processed_stage(df: pd.DataFrame, filename: str) -> None:
    """
    Save an intermediate or final processed pipeline file.
    """

    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved {filename}: {len(df)} rows")


def run_pipeline() -> pd.DataFrame:
    """
    Run the full catastrophe event extraction pipeline.

    Stages:
    1. Fetch USGS
    2. Fetch GDACS
    3. Fetch NOAA
    4. Combine into common schema
    5. Apply regex extraction
    6. Apply optional spaCy NLP extraction
    7. Apply risk scoring
    8. Apply ILS exposure-zone tagging
    9. Save final enriched dataset
    """

    start_time = datetime.now(timezone.utc)

    print("=" * 80)
    print("Starting catastrophe event extraction pipeline")
    print(f"Started at: {start_time.isoformat()}")
    print("=" * 80)

    frames = []

    # -----------------------------
    # 1. USGS
    # -----------------------------
    print("\n[1/8] Fetching USGS earthquake events...")

    usgs_df = fetch_usgs_earthquakes(
        days_back=USGS_DAYS_BACK,
        min_magnitude=USGS_MIN_MAGNITUDE,
    )

    print(f"USGS rows: {len(usgs_df)}")
    frames.append(usgs_df)

    # -----------------------------
    # 2. GDACS
    # -----------------------------
    print("\n[2/8] Fetching GDACS live disaster alerts...")

    gdacs_df = fetch_gdacs_events()

    print(f"GDACS rows: {len(gdacs_df)}")
    frames.append(gdacs_df)

    # -----------------------------
    # 3. NOAA
    # -----------------------------
    noaa_df = None

    if RUN_NOAA:
        if NOAA_YEAR == "latest":
            selected_noaa_year = find_latest_noaa_year()
        else:
            selected_noaa_year = int(NOAA_YEAR)

        print(f"\n[3/8] Fetching NOAA Storm Events for {selected_noaa_year}...")

        noaa_df = fetch_noaa_storm_events(year=selected_noaa_year)

        print(f"NOAA rows: {len(noaa_df)}")
        frames.append(noaa_df)
    else:
        print("\n[3/8] Skipping NOAA fetch.")

    save_raw_sources(usgs_df, gdacs_df, noaa_df, selected_noaa_year if RUN_NOAA else None)
    # -----------------------------
    # 4. Combine
    # -----------------------------
    print("\n[4/8] Combining sources into common schema...")

    combined_df = combine_sources(frames)

    print(f"Combined rows: {len(combined_df)}")
    print("\nSource breakdown:")
    print(combined_df["source"].value_counts())

    save_processed_stage(combined_df, "combined_cat_events.csv")

    # -----------------------------
    # 5. Regex extraction
    # -----------------------------
    print("\n[5/8] Applying regex extraction...")

    regex_df = add_regex_features(combined_df)

    regex_cols = [
        "regex_damage_usd",
        "regex_wind_speed_mph",
        "regex_precipitation_inches",
        "regex_magnitude",
        "regex_depth_km",
        "regex_fatalities",
        "regex_injuries",
        "regex_displaced",
    ]

    print("\nRegex extraction coverage:")
    for col in regex_cols:
        count = regex_df[col].notna().sum()
        pct = count / len(regex_df) * 100
        print(f"{col}: {count} rows ({pct:.2f}%)")

    save_processed_stage(regex_df, "cat_events_with_regex.csv")

    # -----------------------------
    # 6. NLP extraction
    # -----------------------------
    if RUN_NLP:
        print("\n[6/8] Applying spaCy NLP extraction...")

        if MAX_NLP_ROWS is not None and len(regex_df) > MAX_NLP_ROWS:
            print(
                f"Using first {MAX_NLP_ROWS} rows for NLP development run "
                f"out of {len(regex_df)} total rows."
            )
            nlp_input_df = regex_df.head(MAX_NLP_ROWS).copy()
        else:
            nlp_input_df = regex_df.copy()

        nlp_df = add_nlp_features(nlp_input_df)

        nlp_cols = [
            "nlp_locations",
            "nlp_organisations",
            "nlp_dates",
            "nlp_cardinal_numbers",
            "extracted_keywords",
        ]

        print("\nNLP extraction coverage:")
        for col in nlp_cols:
            count = nlp_df[col].apply(lambda x: len(x) > 0).sum()
            pct = count / len(nlp_df) * 100
            print(f"{col}: {count} rows ({pct:.2f}%)")

        save_processed_stage(nlp_df, "cat_events_with_nlp.csv")
    else:
        print("\n[6/8] Skipping NLP extraction.")
        nlp_df = regex_df.copy()

    # -----------------------------
    # 7. Risk scoring
    # -----------------------------
    print("\n[7/8] Applying catastrophe risk scoring...")

    risk_df = add_risk_scores(nlp_df)

    print("\nRisk bucket breakdown:")
    print(risk_df["cat_risk_bucket"].value_counts())

    print("\nRisk score summary:")
    print(risk_df["cat_risk_score"].describe())

    save_processed_stage(risk_df, "cat_events_with_risk_scores.csv")

    # -----------------------------
    # 8. ILS exposure-zone tagging
    # -----------------------------
    print("\n[8/8] Applying ILS exposure-zone tagging...")

    final_df = add_exposure_zones(risk_df)

    print("\nILS exposure-zone breakdown:")
    print(final_df["ils_exposure_zone"].value_counts().head(30))

    save_processed_stage(final_df, "cat_events_with_exposure_zones.csv")

    final_output = PROCESSED_DIR / "final_catastrophe_event_dataset.csv"
    final_df.to_csv(final_output, index=False)

    end_time = datetime.now(timezone.utc)
    elapsed = end_time - start_time

    print("\n" + "=" * 80)
    print("Pipeline complete")
    print(f"Finished at: {end_time.isoformat()}")
    print(f"Elapsed time: {elapsed}")
    print(f"Final output: {final_output}")
    print(f"Final rows: {len(final_df)}")
    print(f"Final columns: {len(final_df.columns)}")
    print("=" * 80)

    return final_df


if __name__ == "__main__":
    run_pipeline()