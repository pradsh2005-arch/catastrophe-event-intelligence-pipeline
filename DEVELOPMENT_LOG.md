# Development Log

This document records the step-by-step development process for the Catastrophe Event Intelligence Pipeline, including implementation decisions, issues encountered, fixes applied, and lessons learned.

---

## Step 0 — Project Setup

### What I did
Created the base project structure with separate folders for source code, raw data, processed data, the dashboard, and documentation.

### Files/folders created
- `src/`
- `app/`
- `data/raw/`
- `data/processed/`
- `requirements.txt`
- `.env`
- `.gitignore`
- `README.md`

### Issues encountered
The initial Windows file-creation command `type nul > filename` failed in PowerShell because `type nul` is a Command Prompt convention, not a PowerShell command.

### Fix
Used PowerShell's native file creation command instead:

```powershell
New-Item filename -ItemType File -Force
```

### Lesson
PowerShell and Command Prompt have different syntax, so setup instructions need to be Windows-terminal-specific.

---

## Step 1 — USGS Earthquake Ingestion

### What I did
Built `src/fetch_usgs.py` to call the USGS earthquake API, parse GeoJSON output, and return a structured pandas DataFrame.

### Output
- `data/raw/usgs_earthquakes.csv`

### Fields extracted
- event ID
- source
- event type
- event name
- region
- latitude
- longitude
- depth
- magnitude
- start time
- URL
- description

### Issues encountered
The first version worked, but the output schema was too narrow compared with the wider catastrophe schema planned for the full project.

### Fix
Expanded the USGS output to match the common 18-column catastrophe schema, adding placeholders for fields such as fatalities, injuries, economic damage, and severity.

### Lesson
Even clean API data should be normalised early if it will later be combined with messier sources.

---

## Step 2 — USGS Schema Standardisation

### What I did
Refactored `fetch_usgs.py` so USGS data followed the same common schema later used by GDACS and NOAA.

### Issues encountered
When saving the CSV, Python raised a `PermissionError`.

### Cause
The existing `usgs_earthquakes.csv` file was likely open in Excel or locked by another process.

### Fix
Closed the CSV/Excel file and reran the script. Also noted that locked files can be deleted manually with:

```powershell
Remove-Item data\raw\usgs_earthquakes.csv -Force
```

### Lesson
CSV files opened in Excel cannot always be overwritten by Python on Windows.

---

## Step 3 — GDACS Live Disaster Feed Ingestion

### What I did
Built `src/fetch_gdacs.py` to read the GDACS RSS feed and structure live disaster alerts into the common schema.

### Output
- `data/raw/gdacs_events.csv`

### Initial issue
Many GDACS events were classified as `Other`.

### Diagnosis
After inspecting event titles and descriptions, most `Other` events were actually forest fire notifications, e.g. "Green forest fire notification in Australia".

### Fix
Updated the event-type inference rules to classify the following as `Wildfire`:

- forest fire
- wildfire
- bushfire
- fire notification

### Additional issue
GDACS severity was initially missing because the parser looked for phrases like `green alert`, but GDACS often starts titles with `Green earthquake` or `Green forest fire notification`.

### Fix
Improved severity extraction to recognise:

- titles starting with `Green`
- titles starting with `Orange`
- titles starting with `Red`

### Lesson
Live feeds often use inconsistent wording, so parser rules should be iteratively improved after inspecting raw examples.

---

## Step 4 — NOAA Storm Events Ingestion

### What I did
Built `src/fetch_noaa.py` to download NOAA Storm Events data, decompress the `.csv.gz` file, and normalise the dataset into the common schema.

### Output
- `data/raw/noaa_storm_events_2024.csv`
- Later changed to latest available year, e.g. `data/raw/noaa_storm_events_2026.csv`

### Fields extracted
- event ID
- event type
- state/region
- start and end time
- magnitude
- fatalities
- injuries
- property damage
- event narrative
- latitude/longitude where available

### Initial design decision
Started with NOAA 2024 because it was a stable full-year historical dataset.

### Later improvement
Changed the pipeline to support `NOAA_YEAR = "latest"` so it could automatically pull the newest available NOAA Storm Events file.

### Lesson
A stable historical dataset is useful for testing, but a monitoring pipeline should also support latest-available ingestion.

---

## Step 5 — Common Schema and Source Combination

### What I did
Built `src/normalise.py` to combine USGS, GDACS, and NOAA into one dataset.

### Output
- `data/processed/combined_cat_events.csv`

### Key logic
- Validate each source has the common schema
- Align column order
- Concatenate datasets
- Remove duplicate `source + event_id` pairs

### Result
Combined dataset contained events from:

- NOAA
- USGS
- GDACS

### Lesson
Schema validation is important before combining heterogeneous public datasets.

---

## Step 6 — Regex-Based Extraction

### What I did
Built `src/regex_extract.py` to extract structured metrics from catastrophe event narratives.

### Output
- `data/processed/cat_events_with_regex.csv`

### Fields extracted
- damage amount
- wind speed
- rainfall/snowfall/precipitation inches
- earthquake magnitude
- earthquake depth
- fatalities
- injuries
- displaced people

### Issue / interpretation
Many regex columns had high missingness.

### Explanation
This is expected because regex fields are peril-specific. For example:

- earthquake depth only appears in earthquake narratives
- displaced counts mainly appear in flood/disaster-alert narratives
- damage amounts are only present when explicitly mentioned

### Lesson
High missingness in sparse feature columns is not necessarily a data-quality failure; it can reflect that fields are only relevant to certain event types.

---

## Step 7 — spaCy NLP Extraction

### What I did
Built `src/nlp_extract.py` to extract named entities and catastrophe keywords using spaCy.

### Output
- `data/processed/cat_events_with_nlp.csv`

### Fields added
- `nlp_locations`
- `nlp_organisations`
- `nlp_dates`
- `nlp_cardinal_numbers`
- `extracted_keywords`

### Issue encountered
Running spaCy over the full 70k-row NOAA-heavy dataset was too slow and initially estimated around 45 minutes.

### Fix
Limited the NLP step to a development sample using:

```python
MAX_ROWS = 10000
```

Later, with the latest NOAA year, the full final dataset was only 6,992 rows, so NLP ran across all rows in around 6 minutes.

### Lesson
NLP pipelines can become computationally expensive, so batch size and row limits should be configurable during development.

---

## Step 8 — Catastrophe Risk Scoring

### What I did
Built `src/risk_scoring.py` to assign each event a heuristic risk score and bucket.

### Output
- `data/processed/cat_events_with_risk_scores.csv`

### Risk inputs
- event type
- magnitude
- wind speed
- rainfall/snowfall/precipitation
- damage
- fatalities
- injuries
- displaced people
- GDACS alert level

### Issue encountered
Initial highest-risk events were dominated by NOAA `High Wind` rows because NOAA's `MAGNITUDE` column was being incorrectly treated as earthquake magnitude.

### Diagnosis
In NOAA storm data, `MAGNITUDE` can refer to wind, hail, or other peril-specific measurements, not earthquake magnitude.

### Fix
Updated `score_magnitude()` so it only applies to earthquake events.

### Lesson
The same column name can mean different things across data sources, so scoring logic must be source- and peril-aware.

---

## Step 9 — ILS Exposure-Zone Tagging

### What I did
Built `src/exposure_tagging.py` to map events into ILS-relevant peril-region buckets.

### Output
- `data/processed/cat_events_with_exposure_zones.csv`

### Example tags
- US Winter Storm
- US Severe Convective Storm
- US Windstorm
- US Flood
- US Heat / Drought
- Global Earthquake
- Japan Earthquake
- Australia Wildfire

### Issue encountered
When running the full in-memory pipeline later, exposure tagging failed because NLP fields were Python lists rather than CSV strings.

### Cause
The function `build_search_text()` used `pd.notna()` on list-like fields, which returned multiple boolean values and caused an ambiguity error.

### Fix
Updated `build_search_text()` to explicitly handle:

- missing values
- strings
- lists
- dictionaries
- numeric values

### Lesson
Code that works after reading from CSV may fail in an in-memory pipeline because data types can differ.

---

## Step 10 — Master Pipeline Runner

### What I did
Built `src/run_pipeline.py` to run the entire workflow with one command:

```powershell
python src\run_pipeline.py
```

### Pipeline stages
1. Fetch USGS data
2. Fetch GDACS data
3. Fetch NOAA data
4. Combine sources
5. Apply regex extraction
6. Apply spaCy NLP extraction
7. Apply risk scoring
8. Apply ILS exposure-zone tagging
9. Save final dataset

### Output
- `data/processed/final_catastrophe_event_dataset.csv`

### Final run result
The latest run produced:

- 6,992 rows
- 34 columns
- runtime of around 6 minutes 31 seconds

### Lesson
A master runner makes the project easier to reproduce and much more professional than isolated scripts.

---

## Step 11 — Streamlit Dashboard

### What I did
Built `app/dashboard.py` to visualise the final enriched dataset.

### Dashboard features
- KPI cards
- Source breakdown
- Risk bucket breakdown
- ILS exposure-zone chart
- Event-type breakdown
- Event map
- Highest-risk events table
- Raw data explorer
- Sidebar filters

### Issue / limitation
The map is sparse because many NOAA records lack latitude and longitude.

### Lesson
Dashboard design should reflect source limitations, especially missing coordinates.

---

## Step 12 — Real-World Event Validation

### What I did
Matched dashboard examples back to real-world catastrophe records.

### Example
NOAA Heavy Snow event in the Greater Lake Tahoe Area.

### Pipeline output
- event type: `Heavy Snow`
- exposure zone: `US Winter Storm`
- risk bucket: `High`
- extracted wind speed: 90 mph
- extracted snowfall/precipitation: 40.4 inches

### Validation logic
The NOAA narrative referenced the Central Sierra Snow Lab and reported 40.4 inches of snowfall, which the pipeline extracted correctly.

### Lesson
Validation examples help prove the extraction layer is not just producing abstract outputs but corresponds to real-world events.

---

## Step 13 — SQLite Database Export

### What I did
Built `src/export_to_sqlite.py` to export the final dataset into SQLite.

### Output
- `data/processed/cat_events.db`
- `data/processed/sql_query_outputs/`

### SQL objects created
- `events` table
- `exposure_zone_summary` view
- `source_summary` view
- `high_risk_events` view

### Example queries
- Events by exposure zone
- High-risk events by source
- High-risk US Winter Storm events
- Event types ranked by risk
- Top 50 events by risk score

### Validation
Created and ran `src/check_sqlite.py` to confirm:

- 6,992 rows in the `events` table
- SQL views existed
- Top high-risk events could be queried

### Lesson
Adding SQLite moved the project beyond CSV-only outputs and demonstrated database/query-language agility.

---

## Step 14 — LLM-Assisted Extraction

### Initial OpenAI API attempt

#### What I did
Built `src/llm_extract.py` to send selected high-risk event narratives to an OpenAI model and return structured JSON.

#### Issue encountered
All calls failed with:

```
insufficient_quota
```

#### Cause
ChatGPT Plus does not include OpenAI API credits. API billing is separate from the ChatGPT subscription.

#### Lesson
API usage requires separate quota/billing even if ChatGPT Plus is active.

### Local Ollama solution

#### What I did
Installed Ollama and used a local model:

```
llama3.2:3b
```

Built `src/llm_extract_ollama.py`.

#### Output
- `data/processed/llm_extracted_high_risk_events_ollama.csv`

#### Fields extracted
- LLM summary
- Affected locations
- Peril type
- Severity indicators
- Human impact
- Investment relevance
- Confidence score

#### Issue encountered
Initial local LLM outputs were too shallow, often returning vague indicators like `High`.

#### Fix
Improved the prompt to explicitly request numerical severity indicators such as:

- mph winds
- inches of snow/rain
- deaths
- injuries
- displaced people
- property damage
- power outages
- road closures
- duration

#### Result
The improved local LLM output extracted more useful indicators such as:

- 90 mph winds
- 40.4 inches snowfall
- 10–16 inches snow
- 17 fatalities
- 1,624 displaced people

#### Lesson
Local LLMs can provide free structured extraction, but prompt design matters significantly.

---

## Step 15 — Data Quality and Validation Report

### What I did
Built `src/data_quality_report.py` to create a written quality report.

### Output
- `data/processed/data_quality_report.txt`

### Checks included
- Row and column count
- Source breakdown
- Missingness by column
- Duplicate event checks
- Coordinate quality checks
- Top event types
- Risk bucket distribution
- ILS exposure-zone distribution
- Regex coverage
- NLP coverage
- Local LLM extraction quality
- Validation examples
- Limitations

### Key results
- 6,992 rows
- 34 columns
- 0 duplicate source-event ID rows
- 0 fully duplicated rows
- 0 invalid latitude rows
- 0 invalid longitude rows

### Lesson
A data quality report makes the project more rigorous and audit-friendly.

---

## Current Final Outputs

The project currently produces:

- `data/processed/final_catastrophe_event_dataset.csv`
- `data/processed/cat_events.db`
- `data/processed/sql_query_outputs/`
- `data/processed/llm_extracted_high_risk_events_ollama.csv`
- `data/processed/data_quality_report.txt`

---

## Key Engineering Decisions

### 1. Use latest NOAA data instead of fixed 2024 data
Started with 2024 for stability, then added latest-year ingestion for current monitoring.

### 2. Use a common schema across all sources
This allowed USGS, GDACS, and NOAA data to be combined despite different formats.

### 3. Use regex before NLP/LLM
Regex provided a deterministic extraction baseline before adding less deterministic NLP/LLM methods.

### 4. Use local Ollama instead of paid API calls
OpenAI API quota was unavailable, so a free local model was used instead.

### 5. Keep risk scoring heuristic
The risk score is deliberately positioned as a prioritisation metric, not a catastrophe loss model.

### 6. Add SQL after the pipeline worked
SQLite was added once the final dataset structure was stable.

---

## Known Limitations

- The risk score is a rule-based prioritisation metric, not a catastrophe loss model.
- The project does not calculate insured loss, expected loss, or portfolio impact.
- NOAA latest-year data may be partial-year data.
- Local LLM outputs should be audited before any production use.
- Many NOAA records lack coordinates, limiting map coverage.
- The project focuses on catastrophe event intelligence and extraction, not cat bond pricing or treaty modelling.