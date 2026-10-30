"""Stage 1: validate raw observations and save a processed copy."""

from __future__ import annotations

import json
from pathlib import Path

from config import (
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    VALIDATION_REPORT_PATH,
    create_directories,
)
from pipeline.data import load_and_validate_daily_data


def main() -> None:
    default_path = str(RAW_DATA_PATH)
    entered_path = input(
        f"Enter the input CSV file path [{default_path}]: "
    ).strip().strip('"')

    input_path = Path(entered_path).expanduser() if entered_path else RAW_DATA_PATH

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV file not found: {input_path}")

    create_directories()
    data, report = load_and_validate_daily_data(input_path, REQUIRED_COLUMNS)

    data.to_csv(PROCESSED_DATA_PATH, index=False, date_format="%Y-%m-%d")
    VALIDATION_REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Data validation completed successfully.")
    print(f"Rows processed: {report.get('row_count', len(data))}")
    if "start_date" in report and "end_date" in report:
        print(f"Date range: {report['start_date']} to {report['end_date']}")
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")
    print(f"Validation report saved to: {VALIDATION_REPORT_PATH}")


if __name__ == "__main__":
    main()