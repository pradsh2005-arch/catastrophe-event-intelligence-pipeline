import json
import os
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from config import PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "final_catastrophe_event_dataset.csv"
OUTPUT_FILE = PROCESSED_DIR / "llm_extracted_high_risk_events.csv"


# Keep this small. We only want selective LLM extraction, not a full-dataset run.
MAX_EVENTS = 20

# Use a relatively cheap/fast model for structured extraction.
MODEL_NAME = "gpt-4o-mini"


def load_client() -> OpenAI:
    """
    Load OpenAI client using OPENAI_API_KEY from .env.
    """

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to your .env file as OPENAI_API_KEY=your_key_here"
        )

    return OpenAI(api_key=api_key)


def load_high_risk_events() -> pd.DataFrame:
    """
    Load final dataset and select the highest-risk event narratives.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing final dataset: {INPUT_FILE}. Run src/run_pipeline.py first."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    if "cat_risk_score" not in df.columns:
        raise ValueError("cat_risk_score column missing. Run risk scoring first.")

    if "description" not in df.columns:
        raise ValueError("description column missing.")

    high_risk_df = (
        df[df["cat_risk_bucket"].isin(["High", "Severe"])]
        .dropna(subset=["description"])
        .sort_values("cat_risk_score", ascending=False)
        .head(MAX_EVENTS)
        .copy()
    )

    print(f"Loaded {len(df)} final events.")
    print(f"Selected {len(high_risk_df)} high-risk events for LLM extraction.")

    return high_risk_df


def build_prompt(row: pd.Series) -> str:
    """
    Build a compact prompt for the LLM.
    """

    event_context = {
        "source": row.get("source"),
        "event_type": row.get("event_type"),
        "event_name": row.get("event_name"),
        "region": row.get("region"),
        "country": row.get("country"),
        "cat_risk_score": row.get("cat_risk_score"),
        "cat_risk_bucket": row.get("cat_risk_bucket"),
        "ils_exposure_zone": row.get("ils_exposure_zone"),
        "regex_wind_speed_mph": row.get("regex_wind_speed_mph"),
        "regex_precipitation_inches": row.get("regex_precipitation_inches"),
        "fatalities": row.get("fatalities"),
        "injuries": row.get("injuries"),
        "economic_damage_usd": row.get("economic_damage_usd"),
        "description": row.get("description"),
    }

    return f"""
You are assisting an insurance-linked securities analyst.

Extract structured information from the catastrophe event record below.

Return JSON only with exactly these keys:
- summary: concise 1-2 sentence event summary
- affected_locations: list of affected locations
- peril_type: broad peril category, e.g. Winter Storm, Flood, Earthquake, Wildfire, Tropical Cyclone, Severe Convective Storm
- severity_indicators: list of quantitative or qualitative severity indicators found in the text
- human_impact: concise statement of deaths, injuries, displaced people or disruption if mentioned
- investment_relevance: concise statement of why this event may matter for ILS/catastrophe exposure monitoring
- confidence_score: number between 0 and 1 reflecting confidence in the extraction

Catastrophe event record:
{json.dumps(event_context, indent=2)}
""".strip()


def parse_json_safely(text: str) -> dict[str, Any]:
    """
    Parse JSON safely from model output.
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON between first { and last }
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

    return {
        "summary": None,
        "affected_locations": [],
        "peril_type": None,
        "severity_indicators": [],
        "human_impact": None,
        "investment_relevance": None,
        "confidence_score": None,
        "parse_error": True,
        "raw_output": text,
    }


def extract_with_llm(client: OpenAI, row: pd.Series) -> dict[str, Any]:
    """
    Call OpenAI model and return structured extraction.
    """

    prompt = build_prompt(row)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise catastrophe risk extraction assistant. "
                    "Return valid JSON only. Do not include markdown."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    if content is None:
        raise ValueError("Empty response from model.")

    return parse_json_safely(content)

def normalise_confidence_score(value: Any) -> float | None:
    """
    Normalise LLM confidence scores to the 0-1 range.

    Some local models may return 7 instead of 0.7.
    """

    if value is None:
        return None

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    if score > 1 and score <= 10:
        score = score / 10

    if score < 0:
        return 0.0

    if score > 1:
        return 1.0

    return score

def normalise_list(value: Any) -> str:
    """
    Convert list-like LLM fields into semicolon-separated strings for CSV storage.
    """

    if isinstance(value, list):
        return "; ".join(str(item) for item in value)

    if value is None:
        return ""

    return str(value)


def run_llm_extraction() -> pd.DataFrame:
    """
    Run LLM extraction on selected high-risk events.
    """

    client = load_client()
    events = load_high_risk_events()

    output_records = []

    for idx, row in events.iterrows():
        print(
            f"Extracting {len(output_records) + 1}/{len(events)}: "
            f"{row.get('event_name')} | {row.get('region')}"
        )

        try:
            extracted = extract_with_llm(client, row)
            parse_error = extracted.get("parse_error", False)

        except Exception as exc:
            extracted = {
                "summary": None,
                "affected_locations": [],
                "peril_type": None,
                "severity_indicators": [],
                "human_impact": None,
                "investment_relevance": None,
                "confidence_score": None,
                "parse_error": True,
                "raw_output": str(exc),
            }
            parse_error = True

        output_records.append(
            {
                "event_id": row.get("event_id"),
                "source": row.get("source"),
                "event_type": row.get("event_type"),
                "event_name": row.get("event_name"),
                "region": row.get("region"),
                "country": row.get("country"),
                "cat_risk_score": row.get("cat_risk_score"),
                "cat_risk_bucket": row.get("cat_risk_bucket"),
                "ils_exposure_zone": row.get("ils_exposure_zone"),
                "description": row.get("description"),
                "llm_summary": extracted.get("summary"),
                "llm_affected_locations": normalise_list(
                    extracted.get("affected_locations")
                ),
                "llm_peril_type": extracted.get("peril_type"),
                "llm_severity_indicators": normalise_list(
                    extracted.get("severity_indicators")
                ),
                "llm_human_impact": extracted.get("human_impact"),
                "llm_investment_relevance": extracted.get("investment_relevance"),
                "llm_confidence_score": normalise_confidence_score(extracted.get("confidence_score")),
                "llm_parse_error": parse_error,
                "llm_raw_output": extracted.get("raw_output"),
            }
        )

        # Small pause to be gentle with rate limits.
        time.sleep(0.25)

    output_df = pd.DataFrame(output_records)

    output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved LLM extraction output to {OUTPUT_FILE}")
    print(f"Rows: {len(output_df)}")

    print("\nParse error count:")
    print(output_df["llm_parse_error"].value_counts(dropna=False))

    print("\nSample LLM outputs:")
    sample_cols = [
        "event_name",
        "region",
        "cat_risk_score",
        "ils_exposure_zone",
        "llm_summary",
        "llm_severity_indicators",
        "llm_investment_relevance",
        "llm_confidence_score",
    ]

    print(output_df[sample_cols].head(10).to_string(index=False))

    return output_df


if __name__ == "__main__":
    run_llm_extraction()