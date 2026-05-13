import pandas as pd
import spacy
from tqdm import tqdm

from config import PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "cat_events_with_regex.csv"
OUTPUT_FILE = PROCESSED_DIR / "cat_events_with_nlp.csv"


CATASTROPHE_KEYWORDS = [
    "earthquake",
    "aftershock",
    "tsunami",
    "hurricane",
    "typhoon",
    "cyclone",
    "tropical storm",
    "storm surge",
    "flood",
    "flash flood",
    "river flood",
    "coastal flood",
    "wildfire",
    "forest fire",
    "bushfire",
    "tornado",
    "hail",
    "thunderstorm",
    "severe wind",
    "high wind",
    "heavy rain",
    "rainfall",
    "landslide",
    "mudslide",
    "volcano",
    "eruption",
    "drought",
    "heat",
    "excessive heat",
    "winter storm",
    "snow",
    "blizzard",
]


def load_spacy_model():
    """
    Load spaCy English model.
    """

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise OSError(
            "spaCy model en_core_web_sm is not installed. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc


def extract_keywords(text: str | None) -> list[str]:
    """
    Extract catastrophe-related keywords using simple keyword matching.
    """

    if not isinstance(text, str):
        return []

    lowered = text.lower()

    return sorted(
        {
            keyword
            for keyword in CATASTROPHE_KEYWORDS
            if keyword in lowered
        }
    )


def extract_entities(doc) -> dict:
    """
    Extract selected named entities from a spaCy document.
    """

    locations = []
    organisations = []
    dates = []
    cardinal_numbers = []

    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            locations.append(ent.text)
        elif ent.label_ == "ORG":
            organisations.append(ent.text)
        elif ent.label_ == "DATE":
            dates.append(ent.text)
        elif ent.label_ == "CARDINAL":
            cardinal_numbers.append(ent.text)

    return {
        "nlp_locations": sorted(set(locations)),
        "nlp_organisations": sorted(set(organisations)),
        "nlp_dates": sorted(set(dates)),
        "nlp_cardinal_numbers": sorted(set(cardinal_numbers)),
    }


def add_nlp_features(df: pd.DataFrame, batch_size: int = 500) -> pd.DataFrame:
    """
    Add NLP features to the catastrophe events dataframe.

    Uses spaCy pipe for faster batch processing.
    """

    df = df.copy()
    nlp = load_spacy_model()

    descriptions = df["description"].fillna("").astype(str).tolist()

    locations_col = []
    organisations_col = []
    dates_col = []
    cardinal_numbers_col = []
    keywords_col = []

    print("Running spaCy NLP extraction...")

    for doc in tqdm(nlp.pipe(descriptions, batch_size=batch_size), total=len(descriptions)):
        entities = extract_entities(doc)

        locations_col.append(entities["nlp_locations"])
        organisations_col.append(entities["nlp_organisations"])
        dates_col.append(entities["nlp_dates"])
        cardinal_numbers_col.append(entities["nlp_cardinal_numbers"])
        keywords_col.append(extract_keywords(doc.text))

    df["nlp_locations"] = locations_col
    df["nlp_organisations"] = organisations_col
    df["nlp_dates"] = dates_col
    df["nlp_cardinal_numbers"] = cardinal_numbers_col
    df["extracted_keywords"] = keywords_col

    return df


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}. Run src/regex_extract.py first."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    # For development, only run NLP on a sample to keep runtime manageable.
    # Later, we can remove this limit for the final full pipeline.
    MAX_ROWS = 10000

    if len(df) > MAX_ROWS:
        df = df.head(MAX_ROWS).copy()

    ########################################################################

    print(f"Loaded {len(df)} regex-enhanced catastrophe events.")

    df = add_nlp_features(df)

    print("\nNLP extraction coverage:")

    nlp_cols = [
        "nlp_locations",
        "nlp_organisations",
        "nlp_dates",
        "nlp_cardinal_numbers",
        "extracted_keywords",
    ]

    for col in nlp_cols:
        count = df[col].apply(lambda x: len(x) > 0).sum()
        pct = count / len(df) * 100
        print(f"{col}: {count} rows ({pct:.2f}%)")

    print("\nSample NLP output:")
    sample = df[
        df["nlp_locations"].apply(lambda x: len(x) > 0)
        | df["extracted_keywords"].apply(lambda x: len(x) > 0)
    ][
        [
            "source",
            "event_type",
            "description",
            "nlp_locations",
            "nlp_dates",
            "extracted_keywords",
        ]
    ].head(10)

    print(sample.to_string(index=False))

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved NLP-enhanced events to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()