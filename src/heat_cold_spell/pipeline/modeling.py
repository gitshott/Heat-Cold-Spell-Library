"""Candidate temperature models and time-ordered evaluation utilities."""

from __future__ import annotations

import os
from collections.abc import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from heat_cold_spell.pipeline.historical import (
    SEASON_BY_MONTH,
    calendar_day_365,
    qualifying_runs,
)

MODEL_DESCRIPTIONS = {
    "Seasonal baseline": "Historical mean for the same calendar day",
    "Trend + Fourier": "Linear trend with six annual harmonic pairs",
    "Random Forest": "Tree ensemble using calendar, lag, and rolling features",
    "DNN": "Dense neural network using calendar, lag, and rolling features",
    "LSTM": "Recurrent network using the previous 60 daily observations",
    "GRU": "GRU recurrent network using the previous 60 daily observations",
}
MODEL_ORDER = list(MODEL_DESCRIPTIONS)
LAGS = [1, 2, 3, 7, 14, 30, 60]


def _annual_design(
    dates: Sequence[pd.Timestamp] | pd.Series | pd.DatetimeIndex,
    origin: pd.Timestamp,
    harmonics: int = 6,
) -> np.ndarray:
    index = pd.DatetimeIndex(dates)
    elapsed_years = (index - origin).days.to_numpy(dtype=float) / 365.2425
    climate_days = calendar_day_365(index)
    phase = 2.0 * np.pi * (climate_days - 1) / 365.0
    columns = [elapsed_years]
    for harmonic in range(1, harmonics + 1):
        columns.extend([np.sin(harmonic * phase), np.cos(harmonic * phase)])
    return np.column_stack(columns)


def _seasonal_baseline(train: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    working = train.copy()
    working["CLIMATE_DAY"] = calendar_day_365(working["DATE"])
    climatology = working.groupby("CLIMATE_DAY")["T2MWET"].mean().reindex(range(1, 366))
    climatology = climatology.interpolate(limit_direction="both")
    return climatology.loc[calendar_day_365(dates)].to_numpy(dtype=float)


def _trend_fourier(train: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    origin = train["DATE"].iloc[0]
    model = LinearRegression()
    model.fit(_annual_design(train["DATE"], origin), train["T2MWET"])
    return model.predict(_annual_design(dates, origin))


def _tabular_feature_row(
    values: list[float] | np.ndarray,
    date: pd.Timestamp,
    origin: pd.Timestamp,
) -> list[float]:
    history = np.asarray(values, dtype=float)
    climate_day = int(calendar_day_365(pd.DatetimeIndex([date]))[0])
    phase = 2.0 * np.pi * (climate_day - 1) / 365.0
    elapsed_years = (date - origin).days / 365.2425
    features = [elapsed_years]
    for harmonic in range(1, 4):
        features.extend([np.sin(harmonic * phase), np.cos(harmonic * phase)])
    features.extend(float(history[-lag]) for lag in LAGS)
    for window in [7, 30, 60]:
        recent = history[-window:]
        features.extend([float(recent.mean()), float(recent.std(ddof=0))])
    return features


def _tabular_training_data(
    train: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    values = train["T2MWET"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(train["DATE"])
    origin = dates[0]
    features = [
        _tabular_feature_row(values[:position], dates[position], origin)
        for position in range(max(LAGS), len(values))
    ]
    target = values[max(LAGS):]
    return np.asarray(features, dtype=np.float32), target.astype(np.float32), origin


def _recursive_tabular_prediction(
    model,
    train_values: np.ndarray,
    dates: pd.DatetimeIndex,
    origin: pd.Timestamp,
    feature_scaler: StandardScaler | None = None,
    target_scaler: StandardScaler | None = None,
) -> np.ndarray:
    history = list(np.asarray(train_values, dtype=float))
    predictions: list[float] = []
    for date in dates:
        row = np.asarray([_tabular_feature_row(history, date, origin)], dtype=np.float32)
        if feature_scaler is not None:
            row = feature_scaler.transform(row)
        scaled_prediction = np.asarray(
            model(row, training=False).numpy()
            if hasattr(model, "layers")
            else model.predict(row)
        ).reshape(-1)[0]
        prediction = (
            float(target_scaler.inverse_transform([[scaled_prediction]])[0, 0])
            if target_scaler is not None
            else float(scaled_prediction)
        )
        history.append(prediction)
        predictions.append(prediction)
    return np.asarray(predictions)


def _random_forest(train: pd.DataFrame, dates: pd.DatetimeIndex, random_seed: int) -> np.ndarray:
    features, target, origin = _tabular_training_data(train)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=random_seed,
        n_jobs=1,
    )
    model.fit(features, target)
    return _recursive_tabular_prediction(
        model,
        train["T2MWET"].to_numpy(),
        dates,
        origin,
    )


def _import_tensorflow():
    import tensorflow as tf

    return tf


def _dnn(
    train: pd.DataFrame,
    dates: pd.DatetimeIndex,
    random_seed: int,
    maximum_epochs: int,
) -> np.ndarray:
    tf = _import_tensorflow()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(random_seed)
    features, target, origin = _tabular_training_data(train)
    feature_scaler = StandardScaler().fit(features)
    target_scaler = StandardScaler().fit(target.reshape(-1, 1))
    scaled_features = feature_scaler.transform(features)
    scaled_target = target_scaler.transform(target.reshape(-1, 1)).reshape(-1)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(features.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )
    ]
    model.fit(
        scaled_features,
        scaled_target,
        validation_split=0.10,
        epochs=maximum_epochs,
        batch_size=64,
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
    )
    return _recursive_tabular_prediction(
        model,
        train["T2MWET"].to_numpy(),
        dates,
        origin,
        feature_scaler,
        target_scaler,
    )


def _sequence_features(
    dates: pd.DatetimeIndex,
    scaled_values: np.ndarray,
    origin: pd.Timestamp,
) -> np.ndarray:
    climate_days = calendar_day_365(dates)
    phase = 2.0 * np.pi * (climate_days - 1) / 365.0
    elapsed_years = (dates - origin).days.to_numpy(dtype=float) / 365.2425
    return np.column_stack(
        [
            scaled_values,
            np.sin(phase),
            np.cos(phase),
            np.sin(2.0 * phase),
            np.cos(2.0 * phase),
            elapsed_years,
        ]
    ).astype(np.float32)


def _sequence_training_data(
    train: pd.DataFrame,
    lookback_days: int,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, pd.Timestamp]:
    values = train["T2MWET"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(train["DATE"])
    scaler = StandardScaler().fit(values.reshape(-1, 1))
    scaled_values = scaler.transform(values.reshape(-1, 1)).reshape(-1)
    features = _sequence_features(dates, scaled_values, dates[0])
    sequences = np.asarray(
        [features[position - lookback_days:position] for position in range(lookback_days, len(features))],
        dtype=np.float32,
    )
    targets = scaled_values[lookback_days:].astype(np.float32)
    return sequences, targets, scaler, dates[0]


def _recursive_sequence_prediction(
    model,
    train: pd.DataFrame,
    dates: pd.DatetimeIndex,
    scaler: StandardScaler,
    origin: pd.Timestamp,
    lookback_days: int,
) -> np.ndarray:
    history_values = list(train["T2MWET"].to_numpy(dtype=float))
    history_dates = list(pd.DatetimeIndex(train["DATE"]))
    predictions: list[float] = []
    for date in dates:
        recent_values = np.asarray(history_values[-lookback_days:])
        recent_dates = pd.DatetimeIndex(history_dates[-lookback_days:])
        scaled = scaler.transform(recent_values.reshape(-1, 1)).reshape(-1)
        sequence = _sequence_features(recent_dates, scaled, origin)[None, :, :]
        scaled_prediction = float(
            np.asarray(model(sequence, training=False).numpy()).reshape(-1)[0]
        )
        prediction = float(scaler.inverse_transform([[scaled_prediction]])[0, 0])
        history_values.append(prediction)
        history_dates.append(date)
        predictions.append(prediction)
    return np.asarray(predictions)


def _recurrent_model(
    model_name: str,
    train: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lookback_days: int,
    random_seed: int,
    maximum_epochs: int,
) -> np.ndarray:
    tf = _import_tensorflow()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(random_seed)
    features, target, scaler, origin = _sequence_training_data(train, lookback_days)
    recurrent_layer = tf.keras.layers.LSTM if model_name == "LSTM" else tf.keras.layers.GRU
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(features.shape[1], features.shape[2])),
            recurrent_layer(32, activation="tanh"),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )
    ]
    model.fit(
        features,
        target,
        validation_split=0.10,
        epochs=maximum_epochs,
        batch_size=64,
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
    )
    return _recursive_sequence_prediction(
        model,
        train,
        dates,
        scaler,
        origin,
        lookback_days,
    )


def fit_and_predict(
    model_name: str,
    train: pd.DataFrame,
    prediction_dates: pd.DatetimeIndex,
    lookback_days: int,
    random_seed: int,
    deep_epochs: int,
) -> np.ndarray:
    """Fit one named candidate and forecast the complete requested horizon."""
    if model_name == "Seasonal baseline":
        return _seasonal_baseline(train, prediction_dates)
    if model_name == "Trend + Fourier":
        return _trend_fourier(train, prediction_dates)
    if model_name == "Random Forest":
        return _random_forest(train, prediction_dates, random_seed)
    if model_name == "DNN":
        return _dnn(train, prediction_dates, random_seed, deep_epochs)
    if model_name in {"LSTM", "GRU"}:
        return _recurrent_model(
            model_name,
            train,
            prediction_dates,
            lookback_days,
            random_seed,
            deep_epochs,
        )
    raise ValueError(f"Unknown model: {model_name}")


def attach_prediction_labels(
    test: pd.DataFrame,
    predicted: np.ndarray,
    minimum_spell_days: int,
) -> pd.DataFrame:
    """Create one auditable prediction table with actual and predicted labels."""
    result = test[
        [
            "DATE",
            "YEAR",
            "MONTH",
            "SEASON",
            "T2MWET",
            "COLD_THRESHOLD_P10",
            "HEAT_THRESHOLD_P90",
            "WARM_EXCEEDANCE",
            "COLD_EXCEEDANCE",
            "HEAT_SPELL",
            "COLD_SPELL",
        ]
    ].copy()
    result = result.rename(columns={"T2MWET": "ACTUAL_T2MWET"})
    result["PREDICTED_T2MWET"] = np.asarray(predicted, dtype=float)
    result["PREDICTED_WARM_EXCEEDANCE"] = result["PREDICTED_T2MWET"] > result["HEAT_THRESHOLD_P90"]
    result["PREDICTED_COLD_EXCEEDANCE"] = result["PREDICTED_T2MWET"] < result["COLD_THRESHOLD_P10"]
    result["PREDICTED_HEAT_SPELL"], _ = qualifying_runs(
        result["PREDICTED_WARM_EXCEEDANCE"].to_numpy(), minimum_spell_days
    )
    result["PREDICTED_COLD_SPELL"], _ = qualifying_runs(
        result["PREDICTED_COLD_EXCEEDANCE"].to_numpy(), minimum_spell_days
    )
    return result


def _classification_metrics(actual: pd.Series, predicted: pd.Series) -> tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual.astype(bool),
        predicted.astype(bool),
        average="binary",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1)


def calculate_prediction_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Calculate temperature, exceedance, and spell-day metrics."""
    actual = predictions["ACTUAL_T2MWET"].to_numpy(dtype=float)
    predicted = predictions["PREDICTED_T2MWET"].to_numpy(dtype=float)
    errors = predicted - actual
    warm_precision, warm_recall, warm_f1 = _classification_metrics(
        predictions["WARM_EXCEEDANCE"], predictions["PREDICTED_WARM_EXCEEDANCE"]
    )
    cold_precision, cold_recall, cold_f1 = _classification_metrics(
        predictions["COLD_EXCEEDANCE"], predictions["PREDICTED_COLD_EXCEEDANCE"]
    )
    heat_spell_precision, heat_spell_recall, heat_spell_f1 = _classification_metrics(
        predictions["HEAT_SPELL"], predictions["PREDICTED_HEAT_SPELL"]
    )
    cold_spell_precision, cold_spell_recall, cold_spell_f1 = _classification_metrics(
        predictions["COLD_SPELL"], predictions["PREDICTED_COLD_SPELL"]
    )
    return {
        "N_DAYS": int(len(predictions)),
        "MAE_C": float(mean_absolute_error(actual, predicted)),
        "RMSE_C": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "BIAS_C": float(errors.mean()),
        "WARM_PRECISION": warm_precision,
        "WARM_RECALL": warm_recall,
        "WARM_F1": warm_f1,
        "COLD_PRECISION": cold_precision,
        "COLD_RECALL": cold_recall,
        "COLD_F1": cold_f1,
        "HEAT_SPELL_PRECISION": heat_spell_precision,
        "HEAT_SPELL_RECALL": heat_spell_recall,
        "HEAT_SPELL_F1": heat_spell_f1,
        "COLD_SPELL_PRECISION": cold_spell_precision,
        "COLD_SPELL_RECALL": cold_spell_recall,
        "COLD_SPELL_F1": cold_spell_f1,
    }


def grouped_rmse(predictions: pd.DataFrame, group_column: str) -> dict[object, float]:
    """Return RMSE for every value of a grouping column."""
    output: dict[object, float] = {}
    for key, group in predictions.groupby(group_column, observed=True, sort=True):
        output[key] = float(
            np.sqrt(mean_squared_error(group["ACTUAL_T2MWET"], group["PREDICTED_T2MWET"]))
        )
    return output
