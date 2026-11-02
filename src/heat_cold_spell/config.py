"""Configuration settings and path management for the data processing pipeline."""

from pathlib import Path

# Project root directory: D:/codingenv/library
BASE_DIR = Path(__file__).resolve().parents[2]

# Default input CSV path
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "42180_Table.csv"

# Generated output directory
OUTPUT_DIR = BASE_DIR

# Output directories
DATA_DIR = OUTPUT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = OUTPUT_DIR / "outputs" / "reports"
# Output files
PROCESSED_DATA_PATH = PROCESSED_DIR / "processed_daily_data.csv"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.json"

# Required columns in the input dataset
REQUIRED_COLUMNS = ["INDEX", "YEAR", "MN", "DT", "MAX", "MIN"]


def create_directories() -> None:
    """Create the directories needed for processed data and reports."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)