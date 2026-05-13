from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure folders exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Data source URLs
USGS_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"
GDACS_RSS_URL = "https://www.gdacs.org/xml/rss.xml"
NOAA_BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"