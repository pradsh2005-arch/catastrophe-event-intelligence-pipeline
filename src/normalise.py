import pandas as pd

from config import RAW_DIR, PROCESSED_DIR


COMMON_COLUMNS = [
    "event_id",
    "source",
    "event_type",
    "event_name",
    "country",
    "region",
    "latitude",
    "longitude",
    "start_time",
    "end_time",
    "magnitude",
    "depth_km",
    "severity",
    "fatalities",
    "injuries",
    "economic_damage_usd",
    "description",
    "url",
]


def load_raw_source(filename: str) -> pd.DataFrame:
    """
    Load a raw source CSV from the data/raw folder.
    """

    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing raw source file: {path}")

    df = pd.read_csv(path)

    return df


def validate_schema(df: pd.DataFrame, source_name: str) -> None:
    """
    Check whether a dataframe contains all common catastrophe schema columns.
    """

    missing_columns = [col for col in COMMON_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{source_name} is missing columns: {missing_columns}"
        )


def align_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only common columns and order them consistently.
    """

    return df[COMMON_COLUMNS].copy()


def combine_raw_sources() -> pd.DataFrame:
    """
    Load USGS, GDACS and NOAA raw files and combine them into one dataset.
    """

    source_files = {
        "USGS": "usgs_earthquakes.csv",
        "GDACS": "gdacs_events.csv",
        "NOAA": "noaa_storm_events_2024.csv",
    }

    frames = []

    for source_name, filename in source_files.items():
        print(f"Loading {source_name}: {filename}")

        df = load_raw_source(filename)
        validate_schema(df, source_name)
        df = align_schema(df)

        print(f"{source_name} rows: {len(df)}")

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    print(f"\nCombined rows before duplicate removal: {len(combined)}")

    combined = combined.drop_duplicates(
        subset=["source", "event_id"],
        keep="last"
    )

    print(f"Combined rows after duplicate removal: {len(combined)}")

    return combined


def save_combined_events(df: pd.DataFrame) -> None:
    """
    Save combined catastrophe events to data/processed.
    """

    output_path = PROCESSED_DIR / "combined_cat_events.csv"

    df.to_csv(output_path, index=False)

    print(f"\nSaved combined catastrophe events to {output_path}")


if __name__ == "__main__":
    combined_events = combine_raw_sources()

    print("\nPreview:")
    print(combined_events.head())

    print("\nShape:")
    print(combined_events.shape)

    print("\nSource breakdown:")
    print(combined_events["source"].value_counts())

    print("\nEvent type breakdown:")
    print(combined_events["event_type"].value_counts().head(25))

    save_combined_events(combined_events)