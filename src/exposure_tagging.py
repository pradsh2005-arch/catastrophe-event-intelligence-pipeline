import pandas as pd

from config import PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "cat_events_with_risk_scores.csv"
OUTPUT_FILE = PROCESSED_DIR / "cat_events_with_exposure_zones.csv"


US_HURRICANE_STATES = [
    "Florida",
    "Texas",
    "Louisiana",
    "Mississippi",
    "Alabama",
    "Georgia",
    "South Carolina",
    "North Carolina",
    "Virginia",
]


US_WILDFIRE_STATES = [
    "California",
    "Oregon",
    "Washington",
    "Colorado",
    "Arizona",
    "New Mexico",
    "Nevada",
    "Idaho",
    "Montana",
    "Wyoming",
]


US_EARTHQUAKE_STATES = [
    "California",
    "Alaska",
    "Washington",
    "Oregon",
    "Nevada",
    "Utah",
]


def text_contains_any(text: str, keywords: list[str]) -> bool:
    """
    Return True if any keyword appears in the text.
    """

    text_lower = text.lower()

    return any(keyword.lower() in text_lower for keyword in keywords)


def build_search_text(row: pd.Series) -> str:
    """
    Combine useful fields into one searchable lowercase string.

    Handles strings, missing values and list-like NLP outputs.
    """

    fields = [
        row.get("event_type"),
        row.get("event_name"),
        row.get("country"),
        row.get("region"),
        row.get("description"),
        row.get("nlp_locations"),
        row.get("extracted_keywords"),
    ]

    cleaned_fields = []

    for field in fields:
        if field is None:
            continue

        if isinstance(field, float) and pd.isna(field):
            continue

        if isinstance(field, list):
            cleaned_fields.extend(str(item) for item in field if item is not None)
            continue

        cleaned_fields.append(str(field))

    return " ".join(cleaned_fields).lower()

def is_us_event(row: pd.Series, search_text: str) -> bool:
    """
    Identify likely US events.
    """

    country = str(row.get("country", "")).lower()
    source = str(row.get("source", "")).lower()

    if country == "united states":
        return True

    if source == "noaa":
        return True

    if "united states" in search_text or "usa" in search_text or "u.s." in search_text:
        return True

    return False


def tag_us_event(row: pd.Series, search_text: str) -> str | None:
    """
    Tag US catastrophe events into ILS-relevant zones.
    """

    event_type = str(row.get("event_type", "")).lower()

    # Earthquake
    if "earthquake" in event_type:
        if text_contains_any(search_text, ["california"]):
            return "US California Earthquake"
        if text_contains_any(search_text, ["alaska"]):
            return "US Alaska Earthquake"
        if text_contains_any(search_text, US_EARTHQUAKE_STATES):
            return "US Other Earthquake"
        return "US Earthquake"

    # Tropical cyclone / hurricane
    if text_contains_any(event_type, ["hurricane", "tropical storm", "tropical cyclone"]):
        if text_contains_any(search_text, US_HURRICANE_STATES):
            return "US Hurricane / Tropical Storm"
        return "US Tropical Cyclone"

    # Severe convective storm: tornado, hail, thunderstorm wind
    if text_contains_any(
        event_type,
        [
            "tornado",
            "hail",
            "thunderstorm wind",
            "marine thunderstorm wind",
            "funnel cloud",
            "waterspout",
        ],
    ):
        return "US Severe Convective Storm"

    # Winter storm
    if text_contains_any(
        event_type,
        [
            "winter storm",
            "winter weather",
            "heavy snow",
            "blizzard",
            "ice storm",
            "lake-effect snow",
            "cold/wind chill",
            "extreme cold/wind chill",
            "frost/freeze",
        ],
    ):
        return "US Winter Storm"

    # Wildfire
    if text_contains_any(event_type, ["wildfire", "forest fire"]):
        if text_contains_any(search_text, US_WILDFIRE_STATES):
            return "US Wildfire"
        return "US Wildfire / Fire"

    # Flood
    if text_contains_any(
        event_type,
        [
            "flood",
            "flash flood",
            "coastal flood",
            "lakeshore flood",
            "heavy rain",
        ],
    ):
        return "US Flood"

    # Wind
    if text_contains_any(event_type, ["high wind", "strong wind", "severe wind"]):
        return "US Windstorm"

    # Heat / drought
    if text_contains_any(event_type, ["heat", "excessive heat", "drought"]):
        return "US Heat / Drought"

    return None


def tag_non_us_event(row: pd.Series, search_text: str) -> str | None:
    """
    Tag non-US catastrophe events into broad ILS-relevant regions.
    """

    event_type = str(row.get("event_type", "")).lower()

    # Japan earthquake / typhoon
    if "japan" in search_text:
        if "earthquake" in event_type:
            return "Japan Earthquake"
        if text_contains_any(event_type, ["typhoon", "tropical cyclone", "tropical storm"]):
            return "Japan Typhoon"
        return "Japan Other"

    # Australia
    if "australia" in search_text:
        if text_contains_any(event_type, ["wildfire", "forest fire"]):
            return "Australia Wildfire"
        if "flood" in event_type:
            return "Australia Flood"
        if text_contains_any(event_type, ["tropical cyclone", "cyclone"]):
            return "Australia Cyclone"
        return "Australia Other"

    # Europe
    if text_contains_any(
        search_text,
        [
            "united kingdom",
            "uk",
            "ireland",
            "france",
            "germany",
            "netherlands",
            "belgium",
            "spain",
            "italy",
            "switzerland",
            "austria",
            "denmark",
            "norway",
            "sweden",
            "poland",
        ],
    ):
        if text_contains_any(event_type, ["wind", "storm"]):
            return "European Windstorm"
        if "flood" in event_type:
            return "European Flood"
        return "Europe Other"

    # Asia-Pacific typhoon regions
    if text_contains_any(
        search_text,
        [
            "philippines",
            "taiwan",
            "china",
            "hong kong",
            "vietnam",
            "south korea",
            "korea",
        ],
    ):
        if text_contains_any(event_type, ["typhoon", "tropical cyclone", "tropical storm"]):
            return "Asia Typhoon"
        if "earthquake" in event_type:
            return "Asia Earthquake"
        if "flood" in event_type:
            return "Asia Flood"
        return "Asia Other"

    # Latin America
    if text_contains_any(
        search_text,
        [
            "mexico",
            "chile",
            "peru",
            "argentina",
            "colombia",
            "ecuador",
            "brazil",
            "guatemala",
            "costa rica",
        ],
    ):
        if "earthquake" in event_type:
            return "Latin America Earthquake"
        if "flood" in event_type:
            return "Latin America Flood"
        if text_contains_any(event_type, ["hurricane", "tropical cyclone", "tropical storm"]):
            return "Latin America Hurricane"
        return "Latin America Other"

    # Caribbean
    if text_contains_any(
        search_text,
        [
            "bahamas",
            "jamaica",
            "cuba",
            "haiti",
            "dominican republic",
            "puerto rico",
            "barbados",
            "antigua",
            "grenada",
        ],
    ):
        if text_contains_any(event_type, ["hurricane", "tropical cyclone", "tropical storm"]):
            return "Caribbean Hurricane"
        if "earthquake" in event_type:
            return "Caribbean Earthquake"
        return "Caribbean Other"

    # Generic non-US peril buckets
    if "earthquake" in event_type:
        return "Global Earthquake"

    if text_contains_any(event_type, ["tropical cyclone", "hurricane", "typhoon", "tropical storm"]):
        return "Global Tropical Cyclone"

    if "flood" in event_type:
        return "Global Flood"

    if text_contains_any(event_type, ["wildfire", "forest fire"]):
        return "Global Wildfire"

    if "volcano" in event_type:
        return "Global Volcano"

    if "drought" in event_type:
        return "Global Drought"

    return None


def tag_ils_exposure_zone(row: pd.Series) -> str:
    """
    Assign a broad ILS-relevant exposure zone to an event.

    This is a rule-based exposure categorisation layer, not a portfolio loss model.
    """

    search_text = build_search_text(row)

    if is_us_event(row, search_text):
        us_tag = tag_us_event(row, search_text)

        if us_tag:
            return us_tag

    non_us_tag = tag_non_us_event(row, search_text)

    if non_us_tag:
        return non_us_tag

    return "Other"


def add_exposure_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ILS exposure-zone tags.
    """

    df = df.copy()

    df["ils_exposure_zone"] = df.apply(tag_ils_exposure_zone, axis=1)

    return df


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}. Run src/risk_scoring.py first."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print(f"Loaded {len(df)} risk-scored catastrophe events.")

    df = add_exposure_zones(df)

    print("\nILS exposure-zone breakdown:")
    print(df["ils_exposure_zone"].value_counts().head(30))

    print("\nHighest-risk events by exposure zone:")
    cols = [
        "source",
        "event_type",
        "event_name",
        "region",
        "country",
        "cat_risk_score",
        "cat_risk_bucket",
        "ils_exposure_zone",
        "description",
    ]

    existing_cols = [col for col in cols if col in df.columns]

    top_events = df.sort_values(
        "cat_risk_score",
        ascending=False
    )[existing_cols].head(20)

    print(top_events.to_string(index=False))

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved exposure-tagged events to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()