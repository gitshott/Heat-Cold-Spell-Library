"""Configuration settings and path management for the data processing pipeline."""

from pathlib import Path

# Project root directory (the directory containing config.py)
BASE_DIR = Path(__file__).resolve().parent

# Default input CSV path, used if the user leaves the prompt blank
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "42180_Table.csv"

# All generated output will be saved under D:\codingenv\library
OUTPUT_DIR = Path(r"D:\codingenv\library")

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