import pandas as pd
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timezone

from config import RAW_DIR, GDACS_RSS_URL


GDACS_COLUMNS = [
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


def clean_html(text: str | None) -> str | None:
    """
    Remove HTML tags from RSS summaries.
    """

    if not text:
        return None

    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def infer_event_type(title: str | None, summary: str | None) -> str:
    """
    Infer a broad catastrophe event type from GDACS title/summary text.
    """

    text = f"{title or ''} {summary or ''}".lower()

    if "earthquake" in text:
        return "Earthquake"

    if (
        "tropical cyclone" in text
        or "cyclone" in text
        or "hurricane" in text
        or "typhoon" in text
    ):
        return "Tropical Cyclone"

    if "flood" in text:
        return "Flood"

    if (
        "forest fire" in text
        or "wildfire" in text
        or "bushfire" in text
        or "fire notification" in text
    ):
        return "Wildfire"

    if "volcano" in text or "volcanic" in text or "eruption" in text:
        return "Volcano"

    if "drought" in text:
        return "Drought"

    return "Other"

def extract_alert_level(title: str | None, summary: str | None) -> str | None:
    """
    Extract GDACS alert severity where visible in title/summary text.
    """

    text = f"{title or ''} {summary or ''}".lower().strip()

    if (
        "red alert" in text
        or "alert level: red" in text
        or text.startswith("red ")
        or " red " in text
    ):
        return "Red"

    if (
        "orange alert" in text
        or "alert level: orange" in text
        or text.startswith("orange ")
        or " orange " in text
    ):
        return "Orange"

    if (
        "green alert" in text
        or "alert level: green" in text
        or text.startswith("green ")
        or " green " in text
    ):
        return "Green"

    return None


def parse_published_datetime(entry) -> datetime | None:
    """
    Convert RSS published date into a timezone-aware datetime.
    """

    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

    return None


def fetch_gdacs_events() -> pd.DataFrame:
    """
    Fetch current disaster alerts from the GDACS RSS feed.

    Returns
    -------
    pd.DataFrame
        A structured dataframe of GDACS events using the common catastrophe schema.
    """

    feed = feedparser.parse(GDACS_RSS_URL)

    records = []

    for entry in feed.entries:
        title = entry.get("title")
        summary = clean_html(entry.get("summary"))
        link = entry.get("link")
        published_dt = parse_published_datetime(entry)

        event_type = infer_event_type(title, summary)
        severity = extract_alert_level(title, summary)

        latitude = entry.get("geo_lat")
        longitude = entry.get("geo_long")

        record = {
            "event_id": entry.get("id") or link or title,
            "source": "GDACS",
            "event_type": event_type,
            "event_name": title,
            "country": None,
            "region": None,
            "latitude": latitude,
            "longitude": longitude,
            "start_time": published_dt,
            "end_time": None,
            "magnitude": None,
            "depth_km": None,
            "severity": severity,
            "fatalities": None,
            "injuries": None,
            "economic_damage_usd": None,
            "description": summary,
            "url": link,
        }

        records.append(record)

    df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame(columns=GDACS_COLUMNS)

    df = df[GDACS_COLUMNS]

    return df


def save_gdacs_events(df: pd.DataFrame) -> None:
    """
    Save GDACS dataframe to the raw data folder.
    """

    output_path = RAW_DIR / "gdacs_events.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} GDACS events to {output_path}")


if __name__ == "__main__":
    gdacs_events = fetch_gdacs_events()

    print("\nPreview:")
    print(gdacs_events.head())

    print("\nShape:")
    print(gdacs_events.shape)

    print("\nColumns:")
    print(gdacs_events.columns.tolist())

    print("\nEvent types:")
    print(gdacs_events["event_type"].value_counts(dropna=False))

    print("\nSeverity levels:")
    print(gdacs_events["severity"].value_counts(dropna=False))

    save_gdacs_events(gdacs_events)