import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

from config import RAW_DIR, USGS_ENDPOINT


USGS_COLUMNS = [
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


def fetch_usgs_earthquakes(days_back: int = 7, min_magnitude: float = 4.5) -> pd.DataFrame:
    """
    Fetch recent earthquake events from the USGS earthquake API.

    Parameters
    ----------
    days_back:
        Number of days before today to search.
    min_magnitude:
        Minimum earthquake magnitude to include.

    Returns
    -------
    pd.DataFrame
        A structured dataframe of earthquake events using the common catastrophe schema.
    """

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)

    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%d"),
        "endtime": end_time.strftime("%Y-%m-%d"),
        "minmagnitude": min_magnitude,
        "orderby": "time",
    }

    response = requests.get(
        USGS_ENDPOINT,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()
    features = data.get("features", [])

    records = []

    for feature in features:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        coordinates = geometry.get("coordinates", [None, None, None])

        longitude = coordinates[0] if len(coordinates) > 0 else None
        latitude = coordinates[1] if len(coordinates) > 1 else None
        depth_km = coordinates[2] if len(coordinates) > 2 else None

        event_time_ms = properties.get("time")

        if event_time_ms is not None:
            event_time = datetime.fromtimestamp(
                event_time_ms / 1000,
                tz=timezone.utc,
            )
        else:
            event_time = None

        record = {
            "event_id": feature.get("id"),
            "source": "USGS",
            "event_type": "Earthquake",
            "event_name": properties.get("title"),
            "country": None,
            "region": properties.get("place"),
            "latitude": latitude,
            "longitude": longitude,
            "start_time": event_time,
            "end_time": None,
            "magnitude": properties.get("mag"),
            "depth_km": depth_km,
            "severity": None,
            "fatalities": None,
            "injuries": None,
            "economic_damage_usd": None,
            "description": properties.get("title"),
            "url": properties.get("url"),
        }

        records.append(record)

    df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame(columns=USGS_COLUMNS)

    df = df[USGS_COLUMNS]

    return df


def save_usgs_earthquakes(df: pd.DataFrame) -> None:
    """
    Save USGS earthquake dataframe to the raw data folder.
    """

    output_path = RAW_DIR / "usgs_earthquakes.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} USGS earthquake events to {output_path}")


if __name__ == "__main__":
    earthquakes = fetch_usgs_earthquakes(
        days_back=14,
        min_magnitude=4.5,
    )

    print("\nPreview:")
    print(earthquakes.head())

    print("\nShape:")
    print(earthquakes.shape)

    print("\nColumns:")
    print(earthquakes.columns.tolist())

    save_usgs_earthquakes(earthquakes)