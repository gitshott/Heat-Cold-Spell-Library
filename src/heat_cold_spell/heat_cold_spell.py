"""Simple reusable API for daily maximum/minimum temperature analysis.

Public workflow: prepare_data -> analyze_history -> compare_models ->
train_models -> predict_future -> predict_spells.
The numbered scripts remain available as command-line entry points.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

_LIBRARY_DIR = Path(__file__).resolve().parent
_SCRIPT_CACHE: dict[str, ModuleType] = {}


def _path(value: str | Path, label: str) -> Path:
    result = Path(value).expanduser()
    if not result.exists():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def _load_workflow_script(filename: str, cache_key: str) -> ModuleType:
    """Load an existing workflow script without running its CLI main()."""
    if cache_key in _SCRIPT_CACHE:
        return _SCRIPT_CACHE[cache_key]
    script_path = _LIBRARY_DIR / filename
    if not script_path.is_file():
        raise FileNotFoundError(f"Required workflow file is missing: {script_path}")
    spec = importlib.util.spec_from_file_location(f"_heat_cold_{cache_key}", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load workflow file: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _SCRIPT_CACHE[cache_key] = module
    return module


def prepare_data(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Clean a raw daily MAX/MIN CSV and save prepared data plus a report.

    Input must contain INDEX, YEAR, MN, DT, MAX, and MIN, matching Stage 1.
    Returns paths and the validation summary.
    """
    source = _path(input_path, "Input file")
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if report_path is None:
        report_destination = destination.with_name(
            f"{destination.stem}_validation.json"
        )
    else:
        report_destination = Path(report_path).expanduser()
    report_destination.parent.mkdir(parents=True, exist_ok=True)

    from heat_cold_spell.config import REQUIRED_COLUMNS
    from heat_cold_spell.pipeline.data import load_and_validate_daily_data

    data, report = load_and_validate_daily_data(source, REQUIRED_COLUMNS)
    data.to_csv(destination, index=False, date_format="%Y-%m-%d")
    report_destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"data_path": destination, "report_path": report_destination, "report": report}


def analyze_history(
    input_path: str | Path,
    output_dir: str | Path,
    location: str,
) -> dict[str, Path]:
    """Classify historical MAX/MIN days, find spells, summarize, and chart."""
    source = _path(input_path, "Input file")
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    workflow = _load_workflow_script("02_historical_analysis.py", "historical")

    old_output_dir = workflow.OUTPUT_DIR
    try:
        workflow.OUTPUT_DIR = destination
        raw = workflow.load_data(source)
        daily = workflow.classify_daily_data(raw, location)
        spells = workflow.detect_spells(daily)
        annual = workflow.create_annual_summary(daily, spells)
        daily_path = destination / "daily_classifications.csv"
        spells_path = destination / "spell_events.csv"
        annual_path = destination / "annual_summaries.csv"
        daily.to_csv(daily_path, index=False, date_format="%Y-%m-%d")
        spells.to_csv(spells_path, index=False)
        annual.to_csv(annual_path, index=False)
        workflow.create_monthly_charts(daily)
        yearly_chart = workflow.create_yearly_spell_chart(annual)
    finally:
        workflow.OUTPUT_DIR = old_output_dir

    return {
        "daily_classifications": daily_path,
        "spell_events": spells_path,
        "annual_summary": annual_path,
        "yearly_spell_chart": yearly_chart,
        "output_dir": destination,
    }


def compare_models(
    input_path: str | Path,
    output_dir: str | Path,
) -> dict[str, dict[str, Path]]:
    """Compare candidate forecasters separately for daily MAX and MIN."""
    source = _path(input_path, "Input file")
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    workflow = _load_workflow_script("03_train_and_compare.py", "comparison")
    data = workflow.load_data(source)
    train, test = workflow.split_data(data)
    outputs: dict[str, dict[str, Path]] = {}
    for target in workflow.TARGET_COLUMNS:
        comparison, predictions = workflow.train_and_compare(train, test, target)
        comparison_path, predictions_path = workflow.save_workbook(
            comparison, predictions, destination, target
        )
        outputs[target] = {
            "comparison": comparison_path,
            "predictions": predictions_path,
        }
    return outputs


def train_models(
    input_path: str | Path,
    comparison_dir: str | Path,
    model_dir: str | Path,
) -> dict[str, Path]:
    """Refit the selected MAX/MIN models on all prepared observations."""
    source = _path(input_path, "Input file")
    comparisons = _path(comparison_dir, "Comparison directory")
    destination = Path(model_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    workflow = _load_workflow_script("04_train_best_model.py", "refit")
    data = workflow.load_data(source)
    selections: dict[str, tuple[str, pd.Series]] = {}
    manifests: dict[str, Path] = {}
    for target in workflow.TARGETS:
        comparison_path = comparisons / f"{target.lower()}_model_comparison.xlsx"
        model_name, metrics = workflow.read_selected(comparison_path)
        selections[target] = (model_name, metrics)
        artifact = workflow.train_target(
            data, target, model_name, metrics, source, comparison_path, destination
        )
        manifests[target] = destination / target.lower() / "model_manifest.json"
    workflow.write_model_rationale(selections, destination)
    return manifests


def predict_future(
    input_path: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    days: int = 1825,
) -> dict[str, Any]:
    """Forecast daily MAX/MIN with the selected saved models."""
    source = _path(input_path, "Input file")
    models = _path(model_dir, "Model directory")
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    if days <= 0:
        raise ValueError("days must be a positive integer")

    workflow = _load_workflow_script("05_predict_future.py", "forecast")
    old_argv = sys.argv
    try:
        sys.argv = [
            str(_LIBRARY_DIR / "05_predict_future.py"),
            "--input", str(source),
            "--model-dir", str(models),
            "--output-dir", str(destination),
            "--forecast-days", str(days),
        ]
        workflow.main()
    finally:
        sys.argv = old_argv

    report_path = destination / "forecast_verification.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {"report_path": report_path, "report": report, **report["outputs"]}


def _read_daily_extremes(path: Path, forecast: bool = False) -> pd.DataFrame:
    """Read observed or forecast MAX/MIN values into one standard schema."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls", ".xlsm"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("Input must be CSV or Excel")
    frame.columns = frame.columns.astype(str).str.strip().str.upper()
    max_column = "PREDICTED_MAX" if "PREDICTED_MAX" in frame else "MAX"
    min_column = "PREDICTED_MIN" if "PREDICTED_MIN" in frame else "MIN"
    if max_column not in frame or min_column not in frame:
        raise ValueError("Input must contain MAX/MIN or PREDICTED_MAX/PREDICTED_MIN")
    if "DATE" in frame:
        dates = pd.to_datetime(frame["DATE"], errors="coerce")
    elif {"YEAR", "MONTH", "DAY"}.issubset(frame.columns):
        dates = pd.to_datetime(
            {"year": frame["YEAR"], "month": frame["MONTH"], "day": frame["DAY"]},
            errors="coerce",
        )
    elif {"YEAR", "MN", "DT"}.issubset(frame.columns):
        dates = pd.to_datetime(
            {"year": frame["YEAR"], "month": frame["MN"], "day": frame["DT"]},
            errors="coerce",
        )
    else:
        raise ValueError("Input needs DATE or YEAR plus MONTH/DAY (or MN/DT)")
    output = pd.DataFrame({
        "DATE": dates,
        "MAX": pd.to_numeric(frame[max_column], errors="coerce"),
        "MIN": pd.to_numeric(frame[min_column], errors="coerce"),
    }).dropna(subset=["DATE"])
    output["YEAR"] = output["DATE"].dt.year
    output["MN"] = output["DATE"].dt.month
    output["DT"] = output["DATE"].dt.day
    output = output.sort_values("DATE").reset_index(drop=True)
    if output["DATE"].duplicated().any():
        raise ValueError("Input contains duplicate dates")
    return output


def _apply_historical_reference(
    target: pd.DataFrame,
    reference: pd.DataFrame,
    location: str,
) -> pd.DataFrame:
    """Apply latest available 15 observed years of calendar-day normals."""
    spell_workflow = _load_workflow_script(
        "06_pastspells.py",
        "spell_rules",
    )
    HEAT_MIN_TMAX_C = spell_workflow.HEAT_MIN_TMAX_C

    location = location.strip().lower()
    if location not in HEAT_MIN_TMAX_C:
        raise ValueError("location must be plains, coastal, or hilly")
    reference = reference[reference["DATE"] < target["DATE"].min()].copy()
    if reference.empty:
        raise ValueError("Historical reference must precede the forecast dates")
    latest_year = int(reference["YEAR"].max())
    first_year = max(int(reference["YEAR"].min()), latest_year - 14)
    reference = reference[reference["YEAR"].between(first_year, latest_year)].copy()
    reference["CALENDAR_DAY"] = reference["DATE"].dt.strftime("%m-%d")
    normals = reference.groupby("CALENDAR_DAY").agg(
        NORMAL_MAX=("MAX", "mean"), NORMAL_MIN=("MIN", "mean")
    )
    result = target.copy()
    result["CALENDAR_DAY"] = result["DATE"].dt.strftime("%m-%d")
    result = result.join(normals, on="CALENDAR_DAY")
    # If the reference has no leap-day rows, interpolate the Feb 29 normal.
    leap = result["CALENDAR_DAY"].eq("02-29") & result["NORMAL_MAX"].isna()
    if leap.any():
        neighbors = normals.reindex(["02-28", "03-01"]).mean()
        result.loc[leap, "NORMAL_MAX"] = neighbors.get("NORMAL_MAX", np.nan)
        result.loc[leap, "NORMAL_MIN"] = neighbors.get("NORMAL_MIN", np.nan)
    result["MAX_DEPARTURE"] = result["MAX"] - result["NORMAL_MAX"]
    result["MIN_DEPARTURE"] = result["MIN"] - result["NORMAL_MIN"]
    max_valid = result["MAX"].notna() & result["NORMAL_MAX"].notna()
    min_valid = result["MIN"].notna() & result["NORMAL_MIN"].notna()
    heat = max_valid & (
        ((result["MAX"] >= HEAT_MIN_TMAX_C[location]) & (result["MAX_DEPARTURE"] >= 4.5))
        | (result["MAX"] >= 45.0)
    )
    cold = min_valid & (result["MIN"] < 10.0) & (result["MIN_DEPARTURE"] <= -3.0)
    result["HEAT_CLASSIFICATION"] = "Normal"
    result["COLD_CLASSIFICATION"] = "Normal"
    result.loc[~max_valid, "HEAT_CLASSIFICATION"] = "Missing"
    result.loc[~min_valid, "COLD_CLASSIFICATION"] = "Missing"
    result.loc[heat, "HEAT_CLASSIFICATION"] = "Heat Wave Day"
    result.loc[cold, "COLD_CLASSIFICATION"] = "Cold Wave Day"
    result["CLASSIFICATION"] = "Normal"
    result.loc[cold, "CLASSIFICATION"] = "Cold Wave Day"
    result.loc[heat, "CLASSIFICATION"] = "Heat Wave Day"
    result.loc[~max_valid & ~min_valid, "CLASSIFICATION"] = "Missing"
    return result


def predict_spells(
    input_path: str | Path,
    output_dir: str | Path,
    location: str,
    historical_path: str | Path | None = None,
) -> dict[str, Path]:
    """Find heat/cold spells in observed or predicted daily MAX/MIN values.

    For a forecast input, pass historical_path to calculate normals from the
    latest 15 observed years before the forecast begins. Without historical_path,
    normals follow the existing retrospective 15-year-window method.
    """
    source = _path(input_path, "Input file")
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        input_columns = set(pd.read_csv(source, nrows=0).columns.str.strip().str.upper())
    elif suffix in {".xlsx", ".xls", ".xlsm"}:
        input_columns = set(pd.read_excel(source, nrows=0).columns.astype(str).str.strip().str.upper())
    else:
        raise ValueError("Input must be CSV or Excel")
    is_forecast = {"PREDICTED_MAX", "PREDICTED_MIN"}.issubset(input_columns)
    if is_forecast and historical_path is None:
        raise ValueError("Pass historical_path so forecast spells use observed-data normals")

    target = _read_daily_extremes(source)
    workflow = _load_workflow_script("06_pastspells.py", "spell_rules")
    if historical_path is None:
        daily = workflow.classify_daily_data(target, location)
    else:
        reference_path = _path(historical_path, "Historical reference file")
        reference = _read_daily_extremes(reference_path)
        daily = _apply_historical_reference(target, reference, location)
    events = workflow.detect_spells(daily)
    daily_path = destination / "daily_spell_classification.csv"
    events_path = destination / "spell_events.csv"
    daily.to_csv(daily_path, index=False, date_format="%Y-%m-%d")
    events.to_csv(events_path, index=False)
    return {"daily_classifications": daily_path, "spell_events": events_path}


__all__ = [
    "prepare_data",
    "analyze_history",
    "compare_models",
    "train_models",
    "predict_future",
    "predict_spells",
]
