import re
import pandas as pd

from config import PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "combined_cat_events.csv"
OUTPUT_FILE = PROCESSED_DIR / "cat_events_with_regex.csv"


def parse_number_text(value: str) -> float | None:
    """
    Convert simple numeric strings and scale words into a float.

    Examples:
    - "4" -> 4
    - "1.8 million" -> 1800000
    - "2 thousand" -> 2000
    """

    if not value:
        return None

    text = value.lower().replace(",", "").strip()

    match = re.search(r"(\d+(?:\.\d+)?)\s*(thousand|million|billion)?", text)

    if not match:
        return None

    number = float(match.group(1))
    scale = match.group(2)

    if scale == "thousand":
        number *= 1_000
    elif scale == "million":
        number *= 1_000_000
    elif scale == "billion":
        number *= 1_000_000_000

    return number


def extract_damage_usd(text: str | None) -> float | None:
    """
    Extract monetary damage amounts from text.

    Examples:
    - "$2.5M"
    - "$10 million"
    - "property damage of 500K"
    """

    if not isinstance(text, str):
        return None

    patterns = [
        r"\$\s*(\d+(?:\.\d+)?)\s*(billion|million|thousand|bn|m|k)?",
        r"damage(?:s)?(?: estimated)?(?: at| of)?\s*\$?\s*(\d+(?:\.\d+)?)\s*(billion|million|thousand|bn|m|k)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            amount = float(match.group(1))
            scale = match.group(2).lower() if match.group(2) else None

            if scale in ["billion", "bn"]:
                amount *= 1_000_000_000
            elif scale in ["million", "m"]:
                amount *= 1_000_000
            elif scale in ["thousand", "k"]:
                amount *= 1_000

            return amount

    return None


def extract_wind_speed_mph(text: str | None) -> float | None:
    """
    Extract wind speed in mph.
    """

    if not isinstance(text, str):
        return None

    patterns = [
        r"(\d+(?:\.\d+)?)\s*mph",
        r"winds?(?: of| up to| around)?\s*(\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return float(match.group(1))

    return None


def extract_precipitation_inches(text: str | None) -> float | None:
    """
    Extract rainfall totals in inches.
    """

    if not isinstance(text, str):
        return None

    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:inches|inch|in\.)\s*(?:of)?\s*(?:rain|rainfall)?",
        r"rainfall(?: totals)?(?: of| around| near)?\s*(\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return float(match.group(1))

    return None


def extract_magnitude(text: str | None) -> float | None:
    """
    Extract earthquake magnitude from text.
    """

    if not isinstance(text, str):
        return None

    patterns = [
        r"magnitude\s*(\d+(?:\.\d+)?)",
        r"\bM\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*M\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return float(match.group(1))

    return None


def extract_depth_km(text: str | None) -> float | None:
    """
    Extract earthquake depth in km.
    """

    if not isinstance(text, str):
        return None

    patterns = [
        r"depth:?\s*(\d+(?:\.\d+)?)\s*km",
        r"depth\s*(?:of)?\s*(\d+(?:\.\d+)?)\s*km",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return float(match.group(1))

    return None


def extract_fatalities(text: str | None) -> float | None:
    """
    Extract fatalities/deaths from event text.
    """

    if not isinstance(text, str):
        return None

    patterns = [
        r"(\d+)\s*(?:deaths|fatalities|people killed|killed)",
        r"caused\s*(\d+)\s*deaths",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return float(match.group(1))

    return None


def extract_injuries(text: str | None) -> float | None:
    """
    Extract injuries from event text.
    """

    if not isinstance(text, str):
        return None

    patterns = [
        r"(\d+)\s*(?:injuries|injured|people injured)",
        r"caused\s*(\d+)\s*injuries",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return float(match.group(1))

    return None


def extract_displaced(text: str | None) -> float | None:
    """
    Extract displaced population counts.
    """

    if not isinstance(text, str):
        return None

    patterns = [
        r"(\d+(?:\.\d+)?\s*(?:thousand|million|billion)?)\s*displaced",
        r"displaced\s*(\d+(?:\.\d+)?\s*(?:thousand|million|billion)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return parse_number_text(match.group(1))

    return None


def add_regex_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regex-extracted fields to the catastrophe events dataframe.
    """

    df = df.copy()

    descriptions = df["description"].fillna("")

    df["regex_damage_usd"] = descriptions.apply(extract_damage_usd)
    df["regex_wind_speed_mph"] = descriptions.apply(extract_wind_speed_mph)
    df["regex_precipitation_inches"] = descriptions.apply(extract_precipitation_inches)
    df["regex_magnitude"] = descriptions.apply(extract_magnitude)
    df["regex_depth_km"] = descriptions.apply(extract_depth_km)
    df["regex_fatalities"] = descriptions.apply(extract_fatalities)
    df["regex_injuries"] = descriptions.apply(extract_injuries)
    df["regex_displaced"] = descriptions.apply(extract_displaced)

    return df


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}. Run src/normalise.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} combined catastrophe events.")

    df = add_regex_features(df)

    print("\nRegex extraction coverage:")
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
        count = df[col].notna().sum()
        pct = count / len(df) * 100
        print(f"{col}: {count} rows ({pct:.2f}%)")

    print("\nSample extracted earthquake fields:")
    earthquake_sample = df[
        df["regex_magnitude"].notna() | df["regex_depth_km"].notna()
    ][
        ["source", "event_type", "description", "regex_magnitude", "regex_depth_km"]
    ].head(10)

    print(earthquake_sample.to_string(index=False))

    print("\nSample extracted impact fields:")
    impact_sample = df[
        df["regex_fatalities"].notna()
        | df["regex_injuries"].notna()
        | df["regex_displaced"].notna()
    ][
        [
            "source",
            "event_type",
            "description",
            "regex_fatalities",
            "regex_injuries",
            "regex_displaced",
        ]
    ].head(10)

    print(impact_sample.to_string(index=False))

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved regex-enhanced events to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()