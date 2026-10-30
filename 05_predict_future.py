"""Forecast future daily MAX and MIN using the models selected in Stage 4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.historical import calendar_day_365
from pipeline.modeling import (
    _annual_design,
    _recursive_sequence_prediction,
    _recursive_tabular_prediction,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "processed" / "processed_daily_data.csv"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "saved_models"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "future_predictions"
DEFAULT_FORECAST_DAYS = 1825
TARGETS = ("MAX", "MIN")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast future daily maximum and minimum temperatures."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=DEFAULT_FORECAST_DAYS,
        help="Number of daily predictions (default: 1825, about five years).",
    )
    return parser.parse_args()


def load_history(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Processed input data was not found: {path}")

    data = pd.read_csv(path)
    data.columns = data.columns.astype(str).str.strip().str.upper()
    required = {"YEAR", "MN", "DT", "MAX", "MIN"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError("Input is missing required columns: " + ", ".join(missing))

    for column in ["YEAR", "MN", "DT", "MAX", "MIN"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if "DATE" in data.columns:
        data["DATE"] = pd.to_datetime(data["DATE"], errors="coerce")
    else:
        data["DATE"] = pd.to_datetime(
            {"year": data["YEAR"], "month": data["MN"], "day": data["DT"]},
            errors="coerce",
        )

    data[["MAX", "MIN"]] = data[["MAX", "MIN"]].replace(
        [np.inf, -np.inf], np.nan
    )
    data = data.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid observation dates were found.")
    if data["DATE"].duplicated().any():
        duplicate = data.loc[data["DATE"].duplicated(), "DATE"].iloc[0]
        raise ValueError(f"Input contains duplicate dates: {duplicate.date()}")
    return data


def load_manifest(target: str, model_dir: Path, input_path: Path) -> dict:
    manifest_path = model_dir / target.lower() / "model_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Model manifest not found: {manifest_path}. "
            "Run 04_train_best_model.py first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3:
        raise ValueError(f"Unsupported model manifest version in {manifest_path}.")
    if manifest.get("target") != target:
        raise ValueError(
            f"{manifest_path} describes {manifest.get('target')!r}, not {target}."
        )

    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if input_hash != manifest.get("training_sha256"):
        raise ValueError(
            "Processed data changed after model training. "
            "Run 04_train_best_model.py again before forecasting."
        )
    return manifest


def predict_target(
    target: str,
    history: pd.DataFrame,
    dates: pd.DatetimeIndex,
    target_dir: Path,
    manifest: dict,
) -> np.ndarray:
    model_name = manifest["model"]
    target_history = history.dropna(subset=[target]).copy()
    values = target_history[target].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"No finite historical values are available for {target}.")

    model_file = manifest.get("model_file")
    support_file = manifest.get("support_file")

    if model_name == "Trend + Fourier":
        design = _annual_design(
            dates,
            pd.Timestamp(manifest["origin"]),
            harmonics=int(manifest["harmonics"]),
        )
        predicted = (
            float(manifest["intercept"])
            + design @ np.asarray(manifest["coefficients"], dtype=float)
        )

    elif model_name == "Seasonal baseline":
        climatology = np.asarray(
            manifest["climatology_by_day"], dtype=float
        )
        if climatology.shape != (365,):
            raise ValueError("Seasonal model artifact must contain 365 daily values.")
        climate_days = calendar_day_365(dates)
        predicted = climatology[climate_days - 1]

    elif model_name == "Random Forest":
        if not model_file:
            raise ValueError("Random Forest manifest is missing model_file.")
        state = joblib.load(target_dir / model_file)
        predicted = _recursive_tabular_prediction(
            state["estimator"],
            np.asarray(state["training_values"], dtype=float),
            dates,
            pd.Timestamp(state["origin"]),
        )

    elif model_name == "DNN":
        if not model_file or not support_file:
            raise ValueError("DNN manifest is missing model/scaler files.")
        import tensorflow as tf

        state = joblib.load(target_dir / support_file)
        model = tf.keras.models.load_model(
            target_dir / model_file, compile=False
        )
        predicted = _recursive_tabular_prediction(
            model,
            values,
            dates,
            pd.Timestamp(manifest["origin"]),
            state["feature_scaler"],
            state["target_scaler"],
        )

    elif model_name in {"LSTM", "GRU"}:
        if not model_file or not support_file:
            raise ValueError(f"{model_name} manifest is missing model/scaler files.")
        import tensorflow as tf

        state = joblib.load(target_dir / support_file)
        model = tf.keras.models.load_model(
            target_dir / model_file, compile=False
        )
        model_history = pd.DataFrame({
            "DATE": target_history["DATE"].to_numpy(),
            "T2MWET": values,
        })
        predicted = _recursive_sequence_prediction(
            model,
            model_history,
            dates,
            state["scaler"],
            pd.Timestamp(manifest["origin"]),
            int(manifest["lookback_days"]),
        )

    else:
        raise ValueError(f"Unsupported selected model for {target}: {model_name}")

    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if len(predicted) != len(dates):
        raise ValueError(
            f"{model_name} returned {len(predicted)} {target} predictions "
            f"for {len(dates)} forecast dates."
        )
    if not np.isfinite(predicted).all():
        raise ValueError(f"{model_name} produced non-finite {target} predictions.")
    return predicted


def save_forecast_chart(
    history: pd.DataFrame,
    forecasts: pd.DataFrame,
    output_path: Path,
    manifests: dict[str, dict],
) -> None:
    recent_start = forecasts["date"].min() - pd.DateOffset(years=2)
    recent = history[history["DATE"] >= recent_start]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, target, color in [
        (axes[0], "MAX", "#dc2626"),
        (axes[1], "MIN", "#2563eb"),
    ]:
        ax.plot(
            recent["DATE"], recent[target],
            color="#94a3b8", linewidth=0.8, label="Historical observations",
        )
        ax.plot(
            forecasts["date"], forecasts[f"predicted_{target.lower()}"],
            color=color, linewidth=1.3,
            label=f"Forecast ({manifests[target]['model']})",
        )
        ax.axvline(
            forecasts["date"].iloc[0],
            color="#475569", linestyle="--", linewidth=0.8,
        )
        ax.set_ylabel(f"{target} (�C)")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)

    axes[0].set_title("Daily maximum and minimum temperature forecasts")
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_arguments()
    if args.forecast_days <= 0:
        raise ValueError("--forecast-days must be a positive integer.")

    history = load_history(args.input)
    manifests = {
        target: load_manifest(target, args.model_dir, args.input)
        for target in TARGETS
    }

    training_ends = {
        pd.Timestamp(manifests[target]["training_end"])
        for target in TARGETS
    }
    if len(training_ends) != 1:
        raise ValueError(
            "MAX and MIN models have different training end dates. "
            "Retrain both targets together before forecasting."
        )

    training_end = training_ends.pop()
    history_end = pd.Timestamp(history["DATE"].max())
    if history_end != training_end:
        raise ValueError(
            f"Model training ends {training_end.date()}, but input data ends "
            f"{history_end.date()}. Retrain using the current processed data."
        )

    dates = pd.date_range(
        training_end + pd.Timedelta(days=1),
        periods=args.forecast_days,
        freq="D",
    )

    forecasts = pd.DataFrame({
        "date": dates,
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
    })
    for target in TARGETS:
        forecasts[f"predicted_{target.lower()}"] = predict_target(
            target,
            history,
            dates,
            args.model_dir / target.lower(),
            manifests[target],
        )

    # Flag rather than silently alter forecasts if independently fitted models
    # ever predict a daily maximum below the corresponding minimum.
    forecasts["max_below_min_flag"] = (
        forecasts["predicted_max"] < forecasts["predicted_min"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "future_max_min_predictions.csv"
    forecasts.to_csv(predictions_path, index=False, date_format="%Y-%m-%d")

    monthly = (
        forecasts.assign(year_month=forecasts["date"].dt.to_period("M").astype(str))
        .groupby(["year", "month", "year_month"], as_index=False)
        .agg(
            days=("date", "count"),
            mean_predicted_max=("predicted_max", "mean"),
            min_predicted_max=("predicted_max", "min"),
            max_predicted_max=("predicted_max", "max"),
            mean_predicted_min=("predicted_min", "mean"),
            min_predicted_min=("predicted_min", "min"),
            max_predicted_min=("predicted_min", "max"),
            max_below_min_days=("max_below_min_flag", "sum"),
        )
    )
    monthly_path = args.output_dir / "future_monthly_summary.csv"
    monthly.to_csv(monthly_path, index=False)

    annual = (
        forecasts.groupby("year", as_index=False)
        .agg(
            days=("date", "count"),
            mean_predicted_max=("predicted_max", "mean"),
            min_predicted_max=("predicted_max", "min"),
            max_predicted_max=("predicted_max", "max"),
            mean_predicted_min=("predicted_min", "mean"),
            min_predicted_min=("predicted_min", "min"),
            max_predicted_min=("predicted_min", "max"),
            max_below_min_days=("max_below_min_flag", "sum"),
        )
    )
    annual["complete_year"] = [
        int(row["days"]) == (
            366 if pd.Timestamp(int(row["year"]), 1, 1).is_leap_year else 365
        )
        for _, row in annual.iterrows()
    ]
    annual_path = args.output_dir / "future_annual_summary.csv"
    annual.to_csv(annual_path, index=False)

    chart_path = args.output_dir / "future_max_min_forecast.png"
    save_forecast_chart(history, forecasts, chart_path, manifests)

    saved = pd.read_csv(predictions_path, parse_dates=["date"])
    if not pd.DatetimeIndex(saved["date"]).equals(dates):
        raise AssertionError("Saved forecast dates do not match the expected dates.")
    for target in TARGETS:
        if not np.isfinite(saved[f"predicted_{target.lower()}"]).all():
            raise AssertionError(f"Saved {target} forecasts contain invalid values.")

    report = {
        "status": "passed",
        "targets": {
            target: {
                "model": manifests[target]["model"],
                "training_end": manifests[target]["training_end"],
                "validation_metrics": manifests[target]["validation_metrics"],
            }
            for target in TARGETS
        },
        "forecast_days": len(saved),
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": dates[-1].strftime("%Y-%m-%d"),
        "continuous_dates": True,
        "all_predictions_finite": True,
        "max_below_min_days": int(forecasts["max_below_min_flag"].sum()),
        "uncertainty_intervals": (
            "Not generated; forecast uncertainty has not been calibrated."
        ),
        "limitations": {
            target: manifests[target]["limitations"] for target in TARGETS
        },
        "outputs": {
            "predictions": str(predictions_path),
            "monthly_summary": str(monthly_path),
            "annual_summary": str(annual_path),
            "chart": str(chart_path),
        },
    }
    report_path = args.output_dir / "forecast_verification.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Predictions: {predictions_path}")
    print(f"Chart: {chart_path}")


if __name__ == "__main__":
    main()
