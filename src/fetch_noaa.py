import gzip
import re
import requests
import pandas as pd
from io import BytesIO
from bs4 import BeautifulSoup

from config import RAW_DIR, NOAA_BASE_URL


NOAA_COLUMNS = [
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

def find_latest_noaa_year() -> int:
    """
    Find the latest available NOAA Storm Events details year
    from the NOAA public directory listing.
    """

    response = requests.get(NOAA_BASE_URL, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    years = []

    pattern = r"StormEvents_details-ftp_v1\.0_d(\d{4})_.*\.csv\.gz"

    for link in soup.find_all("a"):
        href = link.get("href")

        if not href:
            continue

        match = re.search(pattern, href)

        if match:
            years.append(int(match.group(1)))

    if not years:
        raise FileNotFoundError("No NOAA Storm Events detail files found.")

    return max(years)

def find_noaa_details_file(year: int) -> str:
    """
    Find the NOAA StormEvents details CSV file for a given year
    from the NOAA public directory listing.
    """

    response = requests.get(NOAA_BASE_URL, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    target_prefix = f"StormEvents_details-ftp_v1.0_d{year}_"
    matching_files = []

    for link in soup.find_all("a"):
        href = link.get("href")

        if href and href.startswith(target_prefix) and href.endswith(".csv.gz"):
            matching_files.append(href)

    if not matching_files:
        raise FileNotFoundError(f"No NOAA details file found for year {year}")

    # If multiple versions exist, take the latest filename alphabetically.
    return sorted(matching_files)[-1]


def parse_noaa_damage(value) -> float | None:
    """
    Convert NOAA damage strings such as:
    10K, 2.5M, 1B, 0.00K
    into numeric USD values.
    """

    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    if text in ["", "0", "0.00", "0K", "0.00K"]:
        return 0.0

    multiplier = 1.0

    if text.endswith("K"):
        multiplier = 1_000
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("B"):
        multiplier = 1_000_000_000
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return None


def combine_description(row: pd.Series) -> str | None:
    """
    Combine NOAA episode and event narratives into one description field.
    """

    parts = []

    for col in ["EPISODE_NARRATIVE", "EVENT_NARRATIVE"]:
        value = row.get(col)

        if pd.notna(value) and str(value).strip():
            parts.append(str(value).strip())

    if not parts:
        return None

    return " ".join(parts)


def normalise_noaa_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw NOAA Storm Events data into the common catastrophe schema.
    """

    records = []

    for _, row in raw_df.iterrows():
        event_id = row.get("EVENT_ID")
        state = row.get("STATE")
        cz_name = row.get("CZ_NAME")
        event_type = row.get("EVENT_TYPE")

        region_parts = []

        if pd.notna(state):
            region_parts.append(str(state).title())

        if pd.notna(cz_name):
            region_parts.append(str(cz_name).title())

        region = ", ".join(region_parts) if region_parts else None

        record = {
            "event_id": f"NOAA_{event_id}",
            "source": "NOAA",
            "event_type": event_type,
            "event_name": event_type,
            "country": "United States",
            "region": region,
            "latitude": row.get("BEGIN_LAT"),
            "longitude": row.get("BEGIN_LON"),
            "start_time": row.get("BEGIN_DATE_TIME"),
            "end_time": row.get("END_DATE_TIME"),
            "magnitude": row.get("MAGNITUDE"),
            "depth_km": None,
            "severity": row.get("TOR_F_SCALE"),
            "fatalities": row.get("DEATHS_DIRECT"),
            "injuries": row.get("INJURIES_DIRECT"),
            "economic_damage_usd": parse_noaa_damage(row.get("DAMAGE_PROPERTY")),
            "description": combine_description(row),
            "url": None,
        }

        records.append(record)

    df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame(columns=NOAA_COLUMNS)

    df = df[NOAA_COLUMNS]

    return df


def fetch_noaa_storm_events(year: int = 2024) -> pd.DataFrame:
    """
    Fetch NOAA Storm Events details data for one year.

    Parameters
    ----------
    year:
        NOAA storm event year to fetch.

    Returns
    -------
    pd.DataFrame
        Normalised NOAA storm events using the common catastrophe schema.
    """

    filename = find_noaa_details_file(year)
    file_url = NOAA_BASE_URL + filename

    print(f"Downloading NOAA file: {filename}")

    response = requests.get(file_url, timeout=120)
    response.raise_for_status()

    with gzip.open(BytesIO(response.content), mode="rt", encoding="utf-8", errors="replace") as file:
        raw_df = pd.read_csv(file)

    print(f"Raw NOAA rows: {len(raw_df)}")

    df = normalise_noaa_dataframe(raw_df)

    return df


def save_noaa_storm_events(df: pd.DataFrame, year: int) -> None:
    """
    Save NOAA storm events dataframe to the raw data folder.
    """

    output_path = RAW_DIR / f"noaa_storm_events_{year}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} NOAA storm events to {output_path}")


if __name__ == "__main__":
    year = 2024

    noaa_events = fetch_noaa_storm_events(year=year)

    print("\nPreview:")
    print(noaa_events.head())

    print("\nShape:")
    print(noaa_events.shape)

    print("\nColumns:")
    print(noaa_events.columns.tolist())

    print("\nTop event types:")
    print(noaa_events["event_type"].value_counts(dropna=False).head(20))

    print("\nDamage summary:")
    print(noaa_events["economic_damage_usd"].describe())

    save_noaa_storm_events(noaa_events, year=year)