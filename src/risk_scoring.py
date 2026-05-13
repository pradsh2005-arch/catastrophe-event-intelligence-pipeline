import pandas as pd

from config import PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "cat_events_with_nlp.csv"
OUTPUT_FILE = PROCESSED_DIR / "cat_events_with_risk_scores.csv"


def safe_float(value) -> float | None:
    """
    Convert a value to float safely.
    """

    if pd.isna(value):
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def score_magnitude(row: pd.Series) -> int:
    """
    Score earthquake magnitude only.

    Important:
    NOAA's MAGNITUDE field does not always mean earthquake magnitude.
    For wind, hail and other storm events, it can refer to a peril-specific
    measurement. So this score should only apply to earthquake rows.
    """

    event_type = str(row.get("event_type", "")).lower()

    if "earthquake" not in event_type:
        return 0

    magnitude = safe_float(row.get("magnitude"))

    if magnitude is None:
        magnitude = safe_float(row.get("regex_magnitude"))

    if magnitude is None:
        return 0

    if magnitude >= 7.5:
        return 5
    if magnitude >= 7.0:
        return 4
    if magnitude >= 6.0:
        return 3
    if magnitude >= 5.0:
        return 2
    if magnitude >= 4.0:
        return 1

    return 0


def score_wind(row: pd.Series) -> int:
    """
    Score wind speed extracted from text.
    """

    wind_speed = safe_float(row.get("regex_wind_speed_mph"))

    if wind_speed is None:
        return 0

    if wind_speed >= 150:
        return 5
    if wind_speed >= 110:
        return 4
    if wind_speed >= 75:
        return 3
    if wind_speed >= 50:
        return 2
    if wind_speed >= 30:
        return 1

    return 0


def score_precipitation(row: pd.Series) -> int:
    """
    Score precipitation extracted from text.
    """

    precipitation = safe_float(row.get("regex_precipitation_inches"))

    if precipitation is None:
        return 0

    if precipitation >= 20:
        return 5
    if precipitation >= 12:
        return 4
    if precipitation >= 8:
        return 3
    if precipitation >= 4:
        return 2
    if precipitation >= 1:
        return 1

    return 0


def score_damage(row: pd.Series) -> int:
    """
    Score economic damage using NOAA damage or regex damage.
    """

    damage = safe_float(row.get("economic_damage_usd"))

    if damage is None or damage == 0:
        damage = safe_float(row.get("regex_damage_usd"))

    if damage is None:
        return 0

    if damage >= 1_000_000_000:
        return 5
    if damage >= 100_000_000:
        return 4
    if damage >= 10_000_000:
        return 3
    if damage >= 1_000_000:
        return 2
    if damage > 0:
        return 1

    return 0


def score_human_impact(row: pd.Series) -> int:
    """
    Score fatalities, injuries and displaced people.
    """

    fatalities = safe_float(row.get("fatalities"))
    injuries = safe_float(row.get("injuries"))
    displaced = safe_float(row.get("regex_displaced"))

    regex_fatalities = safe_float(row.get("regex_fatalities"))
    regex_injuries = safe_float(row.get("regex_injuries"))

    if fatalities is None:
        fatalities = regex_fatalities

    if injuries is None:
        injuries = regex_injuries

    score = 0

    if fatalities is not None:
        if fatalities >= 100:
            score += 5
        elif fatalities >= 25:
            score += 4
        elif fatalities >= 5:
            score += 3
        elif fatalities > 0:
            score += 2

    if injuries is not None:
        if injuries >= 500:
            score += 4
        elif injuries >= 100:
            score += 3
        elif injuries >= 10:
            score += 2
        elif injuries > 0:
            score += 1

    if displaced is not None:
        if displaced >= 100_000:
            score += 4
        elif displaced >= 10_000:
            score += 3
        elif displaced >= 1_000:
            score += 2
        elif displaced > 0:
            score += 1

    return min(score, 6)


def score_alert_level(row: pd.Series) -> int:
    """
    Score GDACS alert severity.
    """

    severity = str(row.get("severity", "")).lower()

    if severity == "red":
        return 5
    if severity == "orange":
        return 3
    if severity == "green":
        return 1

    return 0


def score_event_type(row: pd.Series) -> int:
    """
    Add a small base score for event types commonly relevant to catastrophe risk.
    """

    event_type = str(row.get("event_type", "")).lower()

    high_cat_types = [
        "earthquake",
        "tornado",
        "hurricane",
        "typhoon",
        "tropical cyclone",
        "tropical storm",
        "flash flood",
        "flood",
        "wildfire",
        "winter storm",
        "hail",
        "thunderstorm wind",
    ]

    for cat_type in high_cat_types:
        if cat_type in event_type:
            return 1

    return 0


def calculate_cat_risk_score(row: pd.Series) -> int:
    """
    Calculate total catastrophe risk score.

    This is a heuristic monitoring score, not a catastrophe loss model.
    """

    score = 0

    score += score_event_type(row)
    score += score_magnitude(row)
    score += score_wind(row)
    score += score_precipitation(row)
    score += score_damage(row)
    score += score_human_impact(row)
    score += score_alert_level(row)

    return int(score)


def assign_risk_bucket(score: int) -> str:
    """
    Convert numerical risk score into a categorical bucket.
    """

    if score >= 9:
        return "Severe"
    if score >= 6:
        return "High"
    if score >= 3:
        return "Moderate"
    return "Low"


def add_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add catastrophe risk scores and risk buckets.
    """

    df = df.copy()

    df["cat_risk_score"] = df.apply(calculate_cat_risk_score, axis=1)
    df["cat_risk_bucket"] = df["cat_risk_score"].apply(assign_risk_bucket)

    return df


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}. Run src/nlp_extract.py first."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print(f"Loaded {len(df)} NLP-enhanced catastrophe events.")

    df = add_risk_scores(df)

    print("\nRisk bucket breakdown:")
    print(df["cat_risk_bucket"].value_counts())

    print("\nRisk score summary:")
    print(df["cat_risk_score"].describe())

    print("\nHighest-risk events:")
    cols = [
        "source",
        "event_type",
        "event_name",
        "region",
        "magnitude",
        "economic_damage_usd",
        "regex_wind_speed_mph",
        "regex_precipitation_inches",
        "fatalities",
        "injuries",
        "regex_displaced",
        "cat_risk_score",
        "cat_risk_bucket",
        "description",
    ]

    existing_cols = [col for col in cols if col in df.columns]

    top_events = df.sort_values(
        "cat_risk_score",
        ascending=False
    )[existing_cols].head(15)

    print(top_events.to_string(index=False))

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved risk-scored events to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()