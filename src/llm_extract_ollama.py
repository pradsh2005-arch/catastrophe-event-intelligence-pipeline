import json
import time
from typing import Any

import pandas as pd
import requests

from config import PROCESSED_DIR


INPUT_FILE = PROCESSED_DIR / "final_catastrophe_event_dataset.csv"
OUTPUT_FILE = PROCESSED_DIR / "llm_extracted_high_risk_events_ollama.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

MAX_EVENTS = 20


def load_high_risk_events() -> pd.DataFrame:
    """
    Load final dataset and select highest-risk event narratives.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing final dataset: {INPUT_FILE}. Run src/run_pipeline.py first."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    high_risk_df = (
        df[df["cat_risk_bucket"].isin(["High", "Severe"])]
        .dropna(subset=["description"])
        .sort_values("cat_risk_score", ascending=False)
        .head(MAX_EVENTS)
        .copy()
    )

    print(f"Loaded {len(df)} final events.")
    print(f"Selected {len(high_risk_df)} high-risk events for local LLM extraction.")

    return high_risk_df


def build_prompt(row: pd.Series) -> str:
    """
    Build prompt for local LLM JSON extraction.
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

Return valid JSON only. Do not include markdown. Do not include explanation.

Important extraction rules:
- The summary must be a full 1-2 sentence summary, not just the event name.
- For severity_indicators, extract exact quantitative indicators where available.
- Prioritise numbers such as mph winds, inches of snow/rain, deaths, injuries, displaced people, property damage, power outages, road closures, affected population and event duration.
- If no human impact is mentioned, say "No direct human impact mentioned in the event record."
- investment_relevance must explain why the event matters for ILS exposure monitoring.
- confidence_score must be between 0 and 1, not a risk score.

Use exactly these keys:
{{
  "summary": "",
  "affected_locations": [],
  "peril_type": "",
  "severity_indicators": [],
  "human_impact": "",
  "investment_relevance": "",
  "confidence_score": 0.0
}}

Catastrophe event record:
{json.dumps(event_context, indent=2)}
""".strip()


def call_ollama(prompt: str) -> str:
    """
    Call local Ollama generate endpoint.
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    data = response.json()

    return data.get("response", "")


def parse_json_safely(text: str) -> dict[str, Any]:
    """
    Parse JSON safely from local LLM output.
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError:
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
    Convert list-like fields into semicolon-separated strings for CSV.
    """

    if isinstance(value, list):
        return "; ".join(str(item) for item in value)

    if value is None:
        return ""

    return str(value)


def run_local_llm_extraction() -> pd.DataFrame:
    """
    Run local Ollama extraction on selected high-risk events.
    """

    events = load_high_risk_events()
    output_records = []

    for _, row in events.iterrows():
        print(
            f"Extracting {len(output_records) + 1}/{len(events)}: "
            f"{row.get('event_name')} | {row.get('region')}"
        )

        prompt = build_prompt(row)

        try:
            raw_output = call_ollama(prompt)
            extracted = parse_json_safely(raw_output)
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
                "llm_provider": "ollama_local",
                "llm_model": MODEL_NAME,
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

        time.sleep(0.1)

    output_df = pd.DataFrame(output_records)
    output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved local LLM extraction output to {OUTPUT_FILE}")
    print(f"Rows: {len(output_df)}")

    print("\nParse error count:")
    print(output_df["llm_parse_error"].value_counts(dropna=False))

    print("\nSample local LLM outputs:")
    sample_cols = [
        "event_name",
        "region",
        "cat_risk_score",
        "ils_exposure_zone",
        "llm_summary",
        "llm_severity_indicators",
        "llm_investment_relevance",
        "llm_confidence_score",
        "llm_parse_error",
    ]

    print(output_df[sample_cols].head(10).to_string(index=False))

    return output_df


if __name__ == "__main__":
    run_local_llm_extraction()