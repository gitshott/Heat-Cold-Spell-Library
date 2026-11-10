"""Stage 1: validate raw observations and save a processed copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from heat_cold_spell.config import (
    REQUIRED_COLUMNS,
)
from heat_cold_spell.pipeline.data import load_and_validate_daily_data


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate raw observations and save a processed copy."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file containing the raw daily observations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the processed CSV file.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Output path for the validation JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    report_path = args.report.expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV file not found: {input_path}")

    data, report = load_and_validate_daily_data(input_path, REQUIRED_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Data validation completed successfully.")
    print(f"Rows processed: {report.get('row_count', len(data))}")
    if "start_date" in report and "end_date" in report:
        print(f"Date range: {report['start_date']} to {report['end_date']}")
    print(f"Processed data saved to: {output_path}")
    print(f"Validation report saved to: {report_path}")


if __name__ == "__main__":
    main()