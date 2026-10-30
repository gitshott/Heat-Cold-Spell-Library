"""Compare daily MAX and MIN prediction models on a chronological test period."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline.modeling import fit_and_predict


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = SCRIPT_DIR / "data" / "processed" / "processed_daily_data.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "model_comparison"

REQUIRED_COLUMNS = ["INDEX", "YEAR", "MN", "DT", "MAX", "MIN"]
FEATURE_COLUMNS = ["YEAR", "MONTH", "DAY"]
TARGET_COLUMNS = ["MAX", "MIN"]

TRAIN_END_DATE = pd.Timestamp("2022-05-31")
TEST_START_DATE = pd.Timestamp("2022-06-01")
TEST_END_DATE = pd.Timestamp("2024-05-31")

RANDOM_SEED = 42
LOOKBACK_DAYS = 60
DEEP_EPOCHS = 60

MODEL_NAMES = [
    "Trend + Fourier",
    "Seasonal baseline",
    "GRU",
    "LSTM",
    "DNN",
    "Random Forest",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare daily MAX and MIN prediction models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Input CSV containing INDEX, YEAR, MN, DT, MAX, and MIN.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for model comparison and prediction workbooks.",
    )
    return parser.parse_args()


def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file was not found: {file_path}")

    data = pd.read_csv(file_path)
    data.columns = data.columns.astype(str).str.strip().str.upper()

    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for column in ["YEAR", "MN", "DT", "MAX", "MIN"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["DATE"] = pd.to_datetime(
        {
            "year": data["YEAR"],
            "month": data["MN"],
            "day": data["DT"],
        },
        errors="coerce",
    )

    data[["MAX", "MIN"]] = data[["MAX", "MIN"]].replace(
        [np.inf, -np.inf], np.nan
    )
    data = data.dropna(subset=["DATE", "YEAR", "MN", "DT"]).copy()

    if data.empty:
        raise ValueError("No valid rows remain after cleaning the input.")

    data[["YEAR", "MN", "DT"]] = data[["YEAR", "MN", "DT"]].astype(int)
    data = data.sort_values("DATE").reset_index(drop=True)

    if data["DATE"].duplicated().any():
        duplicate = data.loc[data["DATE"].duplicated(), "DATE"].iloc[0]
        raise ValueError(f"Duplicate date found: {duplicate.date()}")

    for target in TARGET_COLUMNS:
        if data[target].notna().sum() == 0:
            raise ValueError(f"No valid numeric observations found in {target}.")

    return data


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.loc[data["DATE"] <= TRAIN_END_DATE].copy()
    test = data.loc[
        data["DATE"].between(TEST_START_DATE, TEST_END_DATE)
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Training or test period has no data. "
            f"Training ends {TRAIN_END_DATE.date()}; "
            f"test period is {TEST_START_DATE.date()} to {TEST_END_DATE.date()}."
        )

    return train, test


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    errors = predicted - actual
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "Bias": float(errors.mean()),
    }


def make_model_input(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """Map real dataset columns to the names expected by pipeline.modeling."""
    return pd.DataFrame({
        "DATE": data["DATE"],
        "YEAR": data["YEAR"],
        "MONTH": data["MN"],
        "DAY": data["DT"],
        # fit_and_predict currently expects its target to be named T2MWET.
        "T2MWET": data[target],
    })


def train_and_compare(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Forecast every calendar day so recursive models advance across dates
    # that have no observed row. Score only dates with an observed target.
    forecast_dates = pd.date_range(TEST_START_DATE, TEST_END_DATE, freq="D")
    evaluation = test.dropna(subset=[target]).copy()
    if evaluation.empty:
        raise ValueError(f"The test period contains no valid {target} values.")

    test_dates = pd.DatetimeIndex(evaluation["DATE"])
    actual = evaluation[target].to_numpy(dtype=float)
    model_train = make_model_input(train.dropna(subset=[target]), target)
    if model_train.empty:
        raise ValueError(f"No valid {target} values are available for training.")

    metric_rows = []
    predictions = pd.DataFrame({
        "DATE": evaluation["DATE"].to_numpy(),
        f"ACTUAL_{target}": actual,
    })

    for model_name in MODEL_NAMES:
        print(f"Training {model_name} for {target}...", flush=True)

        try:
            predicted = np.asarray(
                fit_and_predict(
                    model_name,
                    model_train,
                    pd.DatetimeIndex(forecast_dates),
                    lookback_days=LOOKBACK_DAYS,
                    random_seed=RANDOM_SEED,
                    deep_epochs=DEEP_EPOCHS,
                ),
                dtype=float,
            ).reshape(-1)

            if len(predicted) != len(forecast_dates):
                raise ValueError(
                    f"Returned {len(predicted)} predictions "
                    f"for {len(forecast_dates)} calendar dates."
                )
            if not np.isfinite(predicted).all():
                raise ValueError("Predictions contain NaN or infinite values.")

            predicted = pd.Series(predicted, index=forecast_dates).reindex(
                test_dates
            ).to_numpy(dtype=float)

        except Exception as error:
            print(
                f"Skipping {model_name} for {target}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            continue

        metric_rows.append({
            "Model": model_name,
            **calculate_metrics(actual, predicted),
        })
        predictions[model_name] = predicted

    if not metric_rows:
        raise RuntimeError(f"No model completed successfully for {target}.")

    comparison = pd.DataFrame(metric_rows)
    comparison["_ABSOLUTE_BIAS"] = comparison["Bias"].abs()
    comparison = comparison.sort_values(
        ["RMSE", "MAE", "R2", "_ABSOLUTE_BIAS"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    comparison = comparison.drop(columns="_ABSOLUTE_BIAS")

    comparison.insert(0, "Rank", np.arange(1, len(comparison) + 1))
    comparison["Selected"] = np.where(
        comparison["Rank"] == 1, "YES", "NO"
    )
    comparison = comparison[
        ["Rank", "Model", "MAE", "RMSE", "R2", "Bias", "Selected"]
    ]

    model_order = comparison["Model"].tolist()
    predictions = predictions[
        ["DATE", f"ACTUAL_{target}", *model_order]
    ]
    predictions = predictions.rename(
        columns={name: f"PREDICTED_{target}_{name}" for name in model_order}
    )

    return comparison, predictions


def format_worksheet(worksheet, date_column: int | None = None) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E78")

    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center")

    for column_cells in worksheet.columns:
        letter = column_cells[0].column_letter
        max_length = max(
            len(str(cell.value)) if cell.value is not None else 0
            for cell in column_cells
        )
        worksheet.column_dimensions[letter].width = min(
            max(max_length + 2, 12), 36
        )

        for cell in column_cells[1:]:
            if date_column == cell.column:
                cell.number_format = "yyyy-mm-dd"
            elif isinstance(cell.value, (float, np.floating)):
                cell.number_format = "0.000"

    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions


def save_workbook(
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
    target: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / f"{target.lower()}_model_comparison.xlsx"
    comparison.to_excel(comparison_path, index=False)
    workbook = load_workbook(comparison_path)
    format_worksheet(workbook.active)
    workbook.save(comparison_path)

    predictions_path = output_dir / f"{target.lower()}_model_predictions.xlsx"
    predictions.to_excel(predictions_path, index=False)
    workbook = load_workbook(predictions_path)
    format_worksheet(workbook.active, date_column=1)
    workbook.save(predictions_path)

    return comparison_path, predictions_path


def main() -> None:
    args = parse_arguments()
    data = load_data(args.input)

    print(
        f"Loaded {len(data):,} rows: "
        f"{data['DATE'].min().date()} to {data['DATE'].max().date()}",
        flush=True,
    )

    train, test = split_data(data)
    expected_test_dates = pd.date_range(TEST_START_DATE, TEST_END_DATE, freq="D")
    missing_test_dates = expected_test_dates.difference(test["DATE"])
    print(
        f"Train: {train['DATE'].min().date()} to {train['DATE'].max().date()} "
        f"| Test: {test['DATE'].min().date()} to {test['DATE'].max().date()}",
        flush=True,
    )
    if len(missing_test_dates):
        print(
            f"Test period has {len(missing_test_dates)} missing observation date(s); "
            "models will forecast across those calendar days, while metrics use "
            "dates with observed values.",
            flush=True,
        )

    for target in TARGET_COLUMNS:
        print(f"\n=== Comparing models for {target} ===", flush=True)
        comparison, predictions = train_and_compare(train, test, target)

        print(comparison.to_string(index=False))

        comparison_path, predictions_path = save_workbook(
            comparison,
            predictions,
            args.output_dir,
            target,
        )
        print(f"Saved comparison: {comparison_path}")
        print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()