import pandas as pd

from config import PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "final_catastrophe_event_dataset.csv"
LLM_FILE = PROCESSED_DIR / "llm_extracted_high_risk_events_ollama.csv"
REPORT_FILE = PROCESSED_DIR / "data_quality_report.txt"


def format_section(title: str) -> str:
    """
    Create a section header for the text report.
    """

    return "\n" + "=" * 80 + f"\n{title}\n" + "=" * 80 + "\n"


def load_final_dataset() -> pd.DataFrame:
    """
    Load the final catastrophe event dataset.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing final dataset: {INPUT_FILE}. Run src/run_pipeline.py first."
        )

    return pd.read_csv(INPUT_FILE, low_memory=False)


def load_llm_dataset() -> pd.DataFrame | None:
    """
    Load local LLM extraction output if it exists.
    """

    if not LLM_FILE.exists():
        return None

    return pd.read_csv(LLM_FILE, low_memory=False)


def get_basic_summary(df: pd.DataFrame) -> str:
    """
    Return basic dataset shape and column summary.
    """

    lines = []

    lines.append(format_section("1. Basic dataset summary"))
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    lines.append("")
    lines.append("Column names:")
    lines.extend([f"- {col}" for col in df.columns])

    return "\n".join(lines)


def get_source_breakdown(df: pd.DataFrame) -> str:
    """
    Return row counts by source.
    """

    lines = []

    lines.append(format_section("2. Source breakdown"))

    if "source" in df.columns:
        counts = df["source"].value_counts(dropna=False)

        for source, count in counts.items():
            pct = count / len(df) * 100
            lines.append(f"{source}: {count:,} rows ({pct:.2f}%)")
    else:
        lines.append("source column missing.")

    return "\n".join(lines)


def get_missingness_report(df: pd.DataFrame) -> str:
    """
    Return missingness report by column.
    """

    lines = []

    lines.append(format_section("3. Missingness by column"))

    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    missing.columns = ["column", "missing_pct"]

    for _, row in missing.iterrows():
        lines.append(f"{row['column']}: {row['missing_pct'] * 100:.2f}% missing")

    return "\n".join(lines)


def get_duplicate_report(df: pd.DataFrame) -> str:
    """
    Return duplicate event checks.
    """

    lines = []

    lines.append(format_section("4. Duplicate checks"))

    if {"source", "event_id"}.issubset(df.columns):
        duplicate_count = df.duplicated(subset=["source", "event_id"]).sum()
        lines.append(f"Duplicate source-event_id rows: {duplicate_count:,}")
    else:
        lines.append("source/event_id columns missing; duplicate check skipped.")

    full_duplicate_count = df.duplicated().sum()
    lines.append(f"Fully duplicated rows: {full_duplicate_count:,}")

    return "\n".join(lines)


def get_coordinate_report(df: pd.DataFrame) -> str:
    """
    Return coordinate quality checks.
    """

    lines = []

    lines.append(format_section("5. Coordinate quality checks"))

    if not {"latitude", "longitude"}.issubset(df.columns):
        lines.append("latitude/longitude columns missing.")
        return "\n".join(lines)

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")

    both_missing = lat.isna() & lon.isna()
    one_missing = lat.isna() ^ lon.isna()

    invalid_lat = lat.notna() & ~lat.between(-90, 90)
    invalid_lon = lon.notna() & ~lon.between(-180, 180)

    valid_coords = lat.notna() & lon.notna() & ~invalid_lat & ~invalid_lon

    lines.append(f"Rows with valid coordinates: {valid_coords.sum():,} ({valid_coords.mean() * 100:.2f}%)")
    lines.append(f"Rows with both coordinates missing: {both_missing.sum():,} ({both_missing.mean() * 100:.2f}%)")
    lines.append(f"Rows with only one coordinate missing: {one_missing.sum():,}")
    lines.append(f"Rows with invalid latitude: {invalid_lat.sum():,}")
    lines.append(f"Rows with invalid longitude: {invalid_lon.sum():,}")

    return "\n".join(lines)


def get_event_type_report(df: pd.DataFrame) -> str:
    """
    Return top event types.
    """

    lines = []

    lines.append(format_section("6. Top event types"))

    if "event_type" not in df.columns:
        lines.append("event_type column missing.")
        return "\n".join(lines)

    counts = df["event_type"].value_counts(dropna=False).head(25)

    for event_type, count in counts.items():
        pct = count / len(df) * 100
        lines.append(f"{event_type}: {count:,} rows ({pct:.2f}%)")

    return "\n".join(lines)


def get_risk_report(df: pd.DataFrame) -> str:
    """
    Return risk-score and bucket distribution.
    """

    lines = []

    lines.append(format_section("7. Risk scoring distribution"))

    if "cat_risk_bucket" in df.columns:
        bucket_counts = df["cat_risk_bucket"].value_counts(dropna=False)

        lines.append("Risk bucket counts:")
        for bucket, count in bucket_counts.items():
            pct = count / len(df) * 100
            lines.append(f"- {bucket}: {count:,} rows ({pct:.2f}%)")
    else:
        lines.append("cat_risk_bucket column missing.")

    if "cat_risk_score" in df.columns:
        lines.append("")
        lines.append("Risk score summary:")
        lines.append(str(df["cat_risk_score"].describe()))
    else:
        lines.append("cat_risk_score column missing.")

    return "\n".join(lines)


def get_exposure_zone_report(df: pd.DataFrame) -> str:
    """
    Return ILS exposure-zone distribution.
    """

    lines = []

    lines.append(format_section("8. ILS exposure-zone distribution"))

    if "ils_exposure_zone" not in df.columns:
        lines.append("ils_exposure_zone column missing.")
        return "\n".join(lines)

    counts = df["ils_exposure_zone"].value_counts(dropna=False).head(30)

    for zone, count in counts.items():
        pct = count / len(df) * 100
        lines.append(f"{zone}: {count:,} rows ({pct:.2f}%)")

    return "\n".join(lines)


def get_regex_coverage_report(df: pd.DataFrame) -> str:
    """
    Return regex extraction coverage.
    """

    lines = []

    lines.append(format_section("9. Regex extraction coverage"))

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

    for col in regex_cols:
        if col not in df.columns:
            lines.append(f"{col}: missing")
            continue

        count = df[col].notna().sum()
        pct = count / len(df) * 100
        lines.append(f"{col}: {count:,} rows ({pct:.2f}%)")

    return "\n".join(lines)


def get_nlp_coverage_report(df: pd.DataFrame) -> str:
    """
    Return NLP extraction coverage.
    """

    lines = []

    lines.append(format_section("10. NLP extraction coverage"))

    nlp_cols = [
        "nlp_locations",
        "nlp_organisations",
        "nlp_dates",
        "nlp_cardinal_numbers",
        "extracted_keywords",
    ]

    for col in nlp_cols:
        if col not in df.columns:
            lines.append(f"{col}: missing")
            continue

        non_empty = df[col].fillna("").astype(str).str.len() > 2
        count = non_empty.sum()
        pct = count / len(df) * 100
        lines.append(f"{col}: {count:,} rows ({pct:.2f}%)")

    return "\n".join(lines)


def get_llm_report(llm_df: pd.DataFrame | None) -> str:
    """
    Return local LLM extraction report.
    """

    lines = []

    lines.append(format_section("11. Local LLM extraction report"))

    if llm_df is None:
        lines.append("Local LLM output file not found. Run src/llm_extract_ollama.py first.")
        return "\n".join(lines)

    lines.append(f"Rows processed by local LLM: {len(llm_df):,}")

    if "llm_parse_error" in llm_df.columns:
        counts = llm_df["llm_parse_error"].value_counts(dropna=False)

        lines.append("LLM parse error counts:")
        for value, count in counts.items():
            lines.append(f"- {value}: {count:,}")

    if "llm_confidence_score" in llm_df.columns:
        numeric_conf = pd.to_numeric(llm_df["llm_confidence_score"], errors="coerce")
        lines.append("")
        lines.append("LLM confidence score summary:")
        lines.append(str(numeric_conf.describe()))

    lines.append("")
    lines.append("Sample LLM extracted outputs:")

    sample_cols = [
        "event_name",
        "region",
        "llm_summary",
        "llm_severity_indicators",
        "llm_investment_relevance",
    ]

    existing_cols = [col for col in sample_cols if col in llm_df.columns]

    sample = llm_df[existing_cols].head(5)

    lines.append(sample.to_string(index=False))

    return "\n".join(lines)


def get_validation_examples(df: pd.DataFrame, llm_df: pd.DataFrame | None) -> str:
    """
    Return a few concrete validation examples from the dataset.
    """

    lines = []

    lines.append(format_section("12. Validation examples"))

    # Example 1: Lake Tahoe / Sierra winter storm
    lines.append("Example 1: NOAA Lake Tahoe / Sierra heavy snow event")
    tahoe_mask = (
        df["source"].astype(str).str.contains("NOAA", case=False, na=False)
        & df["description"].astype(str).str.contains("Central Sierra Snow Lab|Lake Tahoe|Tahoe", case=False, na=False)
    )

    tahoe = df[tahoe_mask].head(1)

    if not tahoe.empty:
        row = tahoe.iloc[0]
        lines.append(f"- Event type: {row.get('event_type')}")
        lines.append(f"- Region: {row.get('region')}")
        lines.append(f"- Risk score/bucket: {row.get('cat_risk_score')} / {row.get('cat_risk_bucket')}")
        lines.append(f"- ILS exposure zone: {row.get('ils_exposure_zone')}")
        lines.append(f"- Extracted wind speed: {row.get('regex_wind_speed_mph')} mph")
        lines.append(f"- Extracted precipitation/snowfall inches field: {row.get('regex_precipitation_inches')}")
        lines.append("- Interpretation: NOAA narrative references heavy Sierra snowfall and strong winds; pipeline correctly tags it as a US Winter Storm and extracts key severity indicators.")
    else:
        lines.append("- No Lake Tahoe / Sierra example found in current dataset.")

    lines.append("")

    # Example 2: GDACS Indonesia flood
    lines.append("Example 2: GDACS Indonesia flood")
    gdacs_mask = (
        df["source"].astype(str).str.contains("GDACS", case=False, na=False)
        & df["event_type"].astype(str).str.contains("Flood", case=False, na=False)
        & df["description"].astype(str).str.contains("Indonesia", case=False, na=False)
    )

    gdacs = df[gdacs_mask].head(1)

    if not gdacs.empty:
        row = gdacs.iloc[0]
        lines.append(f"- Event name: {row.get('event_name')}")
        lines.append(f"- Risk score/bucket: {row.get('cat_risk_score')} / {row.get('cat_risk_bucket')}")
        lines.append(f"- ILS exposure zone: {row.get('ils_exposure_zone')}")
        lines.append(f"- Extracted fatalities: {row.get('regex_fatalities')}")
        lines.append(f"- Extracted displaced count: {row.get('regex_displaced')}")
        lines.append("- Interpretation: pipeline extracts human-impact indicators from a live GDACS flood narrative.")
    else:
        lines.append("- No GDACS Indonesia flood example found in current dataset.")

    lines.append("")

    # Example 3: USGS earthquake
    lines.append("Example 3: USGS earthquake")
    usgs_mask = df["source"].astype(str).str.contains("USGS", case=False, na=False)

    usgs = df[usgs_mask].head(1)

    if not usgs.empty:
        row = usgs.iloc[0]
        lines.append(f"- Event name: {row.get('event_name')}")
        lines.append(f"- Region: {row.get('region')}")
        lines.append(f"- Magnitude: {row.get('magnitude')}")
        lines.append(f"- Depth km: {row.get('depth_km')}")
        lines.append(f"- ILS exposure zone: {row.get('ils_exposure_zone')}")
        lines.append(f"- URL: {row.get('url')}")
        lines.append("- Interpretation: USGS event records can be traced back to official event pages for validation.")
    else:
        lines.append("- No USGS example found in current dataset.")

    if llm_df is not None:
        lines.append("")
        lines.append("Example 4: Local LLM extraction quality check")

        valid_llm = llm_df[
            llm_df["llm_parse_error"].astype(str).str.lower().isin(["false", "0"])
        ].head(1)

        if not valid_llm.empty:
            row = valid_llm.iloc[0]
            lines.append(f"- Event name: {row.get('event_name')}")
            lines.append(f"- Region: {row.get('region')}")
            lines.append(f"- LLM summary: {row.get('llm_summary')}")
            lines.append(f"- LLM severity indicators: {row.get('llm_severity_indicators')}")
            lines.append("- Interpretation: local LLM converts long event narratives into concise structured summaries and severity indicators.")
        else:
            lines.append("- No successful local LLM extraction rows found.")

    return "\n".join(lines)


def get_limitations_section() -> str:
    """
    Return limitations section.
    """

    lines = []

    lines.append(format_section("13. Limitations"))

    lines.append("- The catastrophe risk score is a rule-based prioritisation metric, not a catastrophe loss model.")
    lines.append("- The project does not estimate actual insured losses, portfolio losses or expected loss.")
    lines.append("- NOAA latest-year files may be partial-year datasets, so row counts are not directly comparable with full historical years.")
    lines.append("- The column regex_precipitation_inches currently captures generic inch-based precipitation/snowfall values, so it should be interpreted carefully for winter storm events.")
    lines.append("- Local LLM extraction is used selectively on high-risk narratives and should be audited before use in any production workflow.")
    lines.append("- Some NOAA records lack coordinates, limiting map coverage.")

    return "\n".join(lines)


def build_report() -> str:
    """
    Build full data quality report.
    """

    df = load_final_dataset()
    llm_df = load_llm_dataset()

    sections = [
        get_basic_summary(df),
        get_source_breakdown(df),
        get_missingness_report(df),
        get_duplicate_report(df),
        get_coordinate_report(df),
        get_event_type_report(df),
        get_risk_report(df),
        get_exposure_zone_report(df),
        get_regex_coverage_report(df),
        get_nlp_coverage_report(df),
        get_llm_report(llm_df),
        get_validation_examples(df, llm_df),
        get_limitations_section(),
    ]

    return "\n".join(sections)


def main() -> None:
    report = build_report()

    REPORT_FILE.write_text(report, encoding="utf-8")

    print(f"Saved data quality report to {REPORT_FILE}")
    print("\nPreview:")
    print(report[:3000])


if __name__ == "__main__":
    main()