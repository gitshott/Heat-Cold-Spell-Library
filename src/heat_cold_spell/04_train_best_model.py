"""Refit each selected MAX/MIN candidate model on all available observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from heat_cold_spell.pipeline.historical import calendar_day_365
from heat_cold_spell.pipeline.modeling import (
    MODEL_DESCRIPTIONS,
    _annual_design,
    _sequence_features,
    _tabular_training_data,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "processed" / "processed_daily_data.csv"
)
DEFAULT_COMPARISON_DIR = PROJECT_ROOT / "outputs" / "model_comparison"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "saved_models"
TARGETS = ("MAX", "MIN")
RANDOM_SEED = 42
LOOKBACK_DAYS = 60
DEEP_EPOCHS = 60


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refit the selected MAX and MIN models on all available data."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
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


def read_selected(comparison_path: Path) -> tuple[str, pd.Series]:
    if not comparison_path.is_file():
        raise FileNotFoundError(
            f"Comparison workbook not found: {comparison_path}. "
            "Run 03_train_and_compare.py first."
        )

    comparison = pd.read_excel(comparison_path)
    comparison.columns = comparison.columns.astype(str).str.strip()
    if not {"Model", "Selected"}.issubset(comparison.columns):
        raise ValueError(f"{comparison_path.name} needs Model and Selected columns.")

    selected = comparison[
        comparison["Selected"].astype(str).str.strip().str.upper().eq("YES")
    ]
    if len(selected) != 1:
        raise ValueError(
            f"{comparison_path.name} must have exactly one selected model."
        )

    model_name = str(selected.iloc[0]["Model"])
    if model_name not in MODEL_DESCRIPTIONS:
        raise ValueError(f"Unknown selected model: {model_name}")
    return model_name, selected.iloc[0]


def fit_refit_model(
    model_name: str,
    training: pd.DataFrame,
    target: str,
    artifact_dir: Path,
) -> tuple[dict, np.ndarray]:
    """Fit the selected candidate and save its model-specific state."""
    dates = pd.DatetimeIndex(training["DATE"])
    values = training[target].to_numpy(dtype=float)
    if len(values) < LOOKBACK_DAYS + 10 and model_name in {
        "Random Forest", "DNN", "LSTM", "GRU"
    }:
        raise ValueError(f"Not enough observations to refit {model_name}.")

    fitted = np.full(len(training), np.nan, dtype=float)
    details: dict = {
        "model": model_name,
        "target": target,
        "units": "degrees Celsius",
    }

    if model_name == "Trend + Fourier":
        origin = dates[0]
        design = _annual_design(dates, origin, harmonics=6)
        model = LinearRegression().fit(design, values)
        fitted = model.predict(design)
        details.update({
            "origin": origin.strftime("%Y-%m-%d"),
            "harmonics": 6,
            "intercept": float(model.intercept_),
            "coefficients": model.coef_.tolist(),
            "feature_order": ["elapsed_years"] + [
                f"{kind}_{k}"
                for k in range(1, 7)
                for kind in ("sin", "cos")
            ],
            "trend_year_days": 365.2425,
        })

    elif model_name == "Seasonal baseline":
        climate_days = calendar_day_365(dates)
        climatology = (
            pd.DataFrame({"CLIMATE_DAY": climate_days, "VALUE": values})
            .groupby("CLIMATE_DAY")["VALUE"]
            .mean()
            .reindex(range(1, 366))
            .interpolate(limit_direction="both")
        )
        fitted = climatology.loc[climate_days].to_numpy(dtype=float)
        details["climatology_by_day"] = climatology.to_list()
        details["calendar"] = "365-day calendar used by pipeline.modeling"

    elif model_name == "Random Forest":
        features, target_values, origin = _tabular_training_data(
            pd.DataFrame({"DATE": dates, "T2MWET": values})
        )
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        model.fit(features, target_values)
        fitted[LOOKBACK_DAYS:] = model.predict(features)
        model_file = artifact_dir / "model.joblib"
        joblib.dump({
            "estimator": model,
            "training_values": values,
            "origin": origin,
        }, model_file)
        details["model_file"] = model_file.name
        details["model_format"] = "joblib"

    elif model_name in {"DNN", "LSTM", "GRU"}:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(RANDOM_SEED)
        train_frame = pd.DataFrame({"DATE": dates, "T2MWET": values})

        if model_name == "DNN":
            features, target_values, origin = _tabular_training_data(train_frame)
            feature_scaler = StandardScaler().fit(features)
            target_scaler = StandardScaler().fit(target_values.reshape(-1, 1))
            scaled_features = feature_scaler.transform(features)
            scaled_targets = target_scaler.transform(
                target_values.reshape(-1, 1)
            ).reshape(-1)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(features.shape[1],)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.10),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(
                scaled_features,
                scaled_targets,
                epochs=DEEP_EPOCHS,
                batch_size=64,
                shuffle=False,
                verbose=0,
            )
            fitted_values = target_scaler.inverse_transform(
                model.predict(scaled_features, verbose=0)
            ).reshape(-1)
            fitted[LOOKBACK_DAYS:] = fitted_values
            joblib.dump(
                {"feature_scaler": feature_scaler, "target_scaler": target_scaler},
                artifact_dir / "scalers.joblib",
            )
            details["support_file"] = "scalers.joblib"
            details["origin"] = origin.strftime("%Y-%m-%d")

        else:
            from heat_cold_spell.pipeline.modeling import _sequence_training_data

            sequences, target_values, scaler, origin = _sequence_training_data(
                train_frame, LOOKBACK_DAYS
            )
            recurrent_layer = (
                tf.keras.layers.LSTM if model_name == "LSTM"
                else tf.keras.layers.GRU
            )
            model = tf.keras.Sequential([
                tf.keras.layers.Input(
                    shape=(sequences.shape[1], sequences.shape[2])
                ),
                recurrent_layer(32, activation="tanh"),
                tf.keras.layers.Dropout(0.10),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(
                sequences,
                target_values,
                epochs=DEEP_EPOCHS,
                batch_size=64,
                shuffle=False,
                verbose=0,
            )
            scaled_fitted = model.predict(sequences, verbose=0).reshape(-1)
            fitted[LOOKBACK_DAYS:] = scaler.inverse_transform(
                scaled_fitted.reshape(-1, 1)
            ).reshape(-1)
            joblib.dump({"scaler": scaler}, artifact_dir / "scalers.joblib")
            details["support_file"] = "scalers.joblib"
            details["origin"] = origin.strftime("%Y-%m-%d")
            details["lookback_days"] = LOOKBACK_DAYS

        model_file = artifact_dir / "model.keras"
        model.save(model_file)
        details["model_file"] = model_file.name
        details["model_format"] = "keras"

    else:
        raise ValueError(f"No refit implementation exists for {model_name!r}.")

    return details, fitted


def train_target(
    data: pd.DataFrame,
    target: str,
    model_name: str,
    selected_metrics: pd.Series,
    input_path: Path,
    comparison_path: Path,
    model_dir: Path,
) -> dict:
    training = data.dropna(subset=[target]).copy()
    values = training[target].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{target} contains non-finite values.")

    target_dir = model_dir / target.lower()
    target_dir.mkdir(parents=True, exist_ok=True)
    model_details, fitted = fit_refit_model(
        model_name, training, target, target_dir
    )
    residuals = values - fitted
    end_date = pd.Timestamp(training["DATE"].iloc[-1])

    artifact = {
        "schema_version": 3,
        **model_details,
        "training_start": pd.Timestamp(training["DATE"].iloc[0]).strftime("%Y-%m-%d"),
        "training_end": end_date.strftime("%Y-%m-%d"),
        "training_rows": int(len(training)),
        "training_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "comparison_sha256": hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
        "validation_period": {"start": "2022-06-01", "end": "2024-05-31"},
        "validation_metrics": {
            key: float(selected_metrics[key])
            for key in ("MAE", "RMSE", "R2", "Bias")
            if key in selected_metrics.index and pd.notna(selected_metrics[key])
        },
        "refit_epochs": DEEP_EPOCHS if model_name in {"DNN", "LSTM", "GRU"} else None,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "tensorflow_version": (
            __import__("tensorflow").__version__
            if model_name in {"DNN", "LSTM", "GRU"} else None
        ),
        "training_rmse_c_not_validation": float(
            np.sqrt(np.nanmean(residuals ** 2))
        ),
        "limitations": (
            "Training-fit RMSE is not validation performance. Model selection "
            "metrics come from the chronological holdout. Forecast uncertainty "
            "is not calibrated by this refit."
        ),
    }
    (target_dir / "model_manifest.json").write_text(
        json.dumps(artifact, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    diagnostics = pd.DataFrame({
        "DATE": training["DATE"].to_numpy(),
        f"ACTUAL_{target}": values,
        f"FITTED_{target}": fitted,
        "RESIDUAL_C": residuals,
    })
    diagnostics.to_csv(
        target_dir / "training_diagnostics.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    return artifact


def write_model_rationale(
    selections: dict[str, tuple[str, pd.Series]],
    model_dir: Path,
) -> None:
    lines = [
        "# Model selection and refit",
        "",
        "## Purpose and evaluation method",
        "",
        "The comparison is a practical, deliberately varied shortlist, not a claim "
        "that these are the only valid forecasting methods. Six candidates are "
        "evaluated on the same chronological test period (2022-06-01 through "
        "2024-05-31). A time-ordered split is used because a random split could "
        "let information from later dates leak into training.",
        "",
        "The ranking uses RMSE first, then MAE, R2, and absolute bias. RMSE is "
        "primary because it penalizes occasional large errors; MAE shows typical "
        "absolute error and is less sensitive to extremes. R2 describes explained "
        "variation, while bias reveals systematic over- or under-prediction. "
        "MAX and MIN are ranked separately because their seasonal patterns and "
        "error profiles can differ.",
        "",
        "## What each candidate does and why it is included",
        "",
        "### Seasonal baseline",
        "",
        "For each calendar day, predicts the historical mean for that day of the "
        "year. It is a simple, transparent benchmark for recurring seasonality. "
        "A more complex model should beat this baseline on dates it has not seen.",
        "",
        "### Trend + Fourier",
        "",
        "Fits a linear long-term trend together with sine and cosine terms for "
        "annual cycles. It represents smooth seasonality and gradual change with "
        "few parameters, making it a useful interpretable comparator and a "
        "reasonable long-horizon baseline.",
        "",
        "### Random Forest",
        "",
        "Fits an ensemble of decision trees to calendar, lag, and rolling-history "
        "features. It can learn nonlinear thresholds and feature interactions "
        "without requiring a linear relationship. It tests whether those effects "
        "improve predictions over the smooth seasonal models.",
        "",
        "### DNN",
        "",
        "Uses a feed-forward neural network on engineered calendar and recent-history "
        "features. It tests nonlinear combinations of those inputs, while sharing "
        "the structured feature setup with the tree model.",
        "",
        "### LSTM",
        "",
        "Uses a recurrent network over a rolling sequence of the previous 60 daily "
        "observations and seasonal/time features. It tests whether explicitly "
        "learning sequence patterns helps beyond fixed lag and rolling summaries.",
        "",
        "### GRU",
        "",
        "Uses a gated recurrent sequence model, like an LSTM, with a different "
        "gating structure and typically fewer parameters. Comparing both helps "
        "check whether sequence learning adds value and whether the result depends "
        "on the recurrent architecture.",
        "",
        "## Why this shortlist, rather than every available model?",
        "",
        "These six cover distinct useful baselines and model families: calendar "
        "climatology, smooth trend/seasonality, nonlinear trees, dense neural "
        "networks, and two recurrent sequence architectures. This gives a readable "
        "comparison with manageable training time and dependencies. Adding many "
        "closely related candidates increases compute and can make the apparent "
        "winner look better simply because more alternatives were tried on the "
        "same holdout.",
        "",
        "Methods such as SARIMA/state-space models, gradient-boosted trees, support "
        "vector regression, and decomposable trend models are not ruled out. They "
        "were left outside this first comparison to keep the experiment focused; "
        "they could be added as candidates and evaluated under the same date split "
        "if there is a reason to test their assumptions or feature strengths. No "
        "candidate is declared best in advance.",
        "",
        "## Selected models and holdout metrics",
        "",
        "| Target | Selected model | MAE | RMSE | R2 | Bias |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for target, (name, row) in selections.items():
        def metric(key: str) -> str:
            value = row.get(key, np.nan)
            return f"{float(value):.4f}" if pd.notna(value) else "n/a"
        lines.append(
            f"| {target} | {name} | {metric('MAE')} | {metric('RMSE')} | "
            f"{metric('R2')} | {metric('Bias')} |"
        )
    lines.extend([
        "",
        "## Refit and limits",
        "",
        "After selection, the winning model for each target is refit using all "
        "valid observations available in the processed input. The holdout metrics "
        "remain the evidence for choosing the model; training-fit error is not "
        "validation performance. Results apply to the selected test period and "
        "data pipeline. A different time period or additional candidate methods "
        "may produce a different winner. Long-range forecasts are estimates and "
        "do not include calibrated uncertainty intervals.",
        "",
    ])
    (model_dir / "model_selection_explanation.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )



def main() -> None:
    args = parse_arguments()
    data = load_data(args.input)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    selections = {}

    for target in TARGETS:
        comparison_path = (
            args.comparison_dir / f"{target.lower()}_model_comparison.xlsx"
        )
        model_name, metrics = read_selected(comparison_path)
        selections[target] = (model_name, metrics)
        artifact = train_target(
            data, target, model_name, metrics, args.input,
            comparison_path, args.model_dir,
        )
        print(
            f"{target}: refit {model_name} using {artifact['training_rows']:,} "
            f"observations through {artifact['training_end']}; artifact: "
            f"{args.model_dir / target.lower() / 'model_manifest.json'}",
            flush=True,
        )

    write_model_rationale(selections, args.model_dir)
    print(f"Explanation: {args.model_dir / 'model_selection_explanation.md'}")


if __name__ == "__main__":
    main()
