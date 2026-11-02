# Detailed usage guide

This guide expands the examples in the root [README](../README.md). It documents the current source tree and uses the public functions exported by the `heat_cold_spell` package.

## 1. Prepare a raw dataset

Create a CSV with at least `INDEX`, `YEAR`, `MN`, `DT`, `MAX`, and `MIN`. The temperature fields are interpreted as degrees Celsius.

```python
from pathlib import Path
from heat_cold_spell import prepare_data

result = prepare_data(
    Path("data/daily_observations.csv"),
    Path("work/prepared_daily_data.csv"),
    report_path=Path("work/prepared_validation.json"),
)

print(result["data_path"])
print(result["report"])
```

The validator converts the date parts to a `DATE` column, sorts by date, drops rows with invalid dates or missing `YEAR`, `MAX`, or `MIN`, and writes a JSON summary. It does not infer units or repair bad temperature values.

## 2. Analyze historical spells

```python
from heat_cold_spell import analyze_history

history = analyze_history(
    "work/prepared_daily_data.csv",
    "work/historical",
    location="plains",  # plains, coastal, or hilly
)

for name, path in history.items():
    print(name, path)
```

The returned paths include daily classifications, spell events, an annual summary, and the yearly spell chart. The workflow also writes three monthly heatmaps in the output directory.

## 3. Compare models

```python
from heat_cold_spell import compare_models

comparison = compare_models(
    "work/prepared_daily_data.csv",
    "work/model_comparison",
)

print(comparison["MAX"]["comparison"])
print(comparison["MIN"]["predictions"])
```

The implementation uses a fixed chronological split: training through 2022-05-31 and testing from 2022-06-01 through 2024-05-31. Candidate models are evaluated independently for MAX and MIN. Ranking prioritizes RMSE, then MAE, R², and absolute bias.

## 4. Refit the selected models

```python
from heat_cold_spell import train_models

manifests = train_models(
    "work/prepared_daily_data.csv",
    "work/model_comparison",
    "work/models",
)

print(manifests)
```

This reads the `Selected` column in both comparison workbooks. Exactly one row must be marked `YES` in each workbook. The input bytes and comparison workbooks are hashed into the manifests so later forecasting can reject stale artifacts.

The selected deep models (`DNN`, `LSTM`, or `GRU`) require TensorFlow. Random Forest uses joblib; the linear and seasonal models are represented directly in JSON manifests.

## 5. Forecast future daily values

```python
from heat_cold_spell import predict_future

forecast = predict_future(
    "work/prepared_daily_data.csv",
    "work/models",
    "work/forecast",
    days=365,
)

print(forecast["report_path"])
print(forecast["report"]["start_date"])
print(forecast["predictions"])
```

The forecast starts on the day after the common model training end date. The implementation verifies continuous dates and finite predictions, then records any independently predicted MAX < MIN rows in `max_below_min_flag`; it does not alter those values.

## 6. Classify forecast spells using observed normals

```python
from heat_cold_spell import predict_spells

spell_outputs = predict_spells(
    "work/forecast/future_max_min_predictions.csv",
    "work/forecast_spells",
    location="plains",
    historical_path="work/prepared_daily_data.csv",
)

print(spell_outputs)
```

For forecast inputs, `historical_path` is required. The facade filters the historical reference to records before the first forecast date, uses the latest available 15 observed years, and averages calendar-day MAX/MIN normals. If February 29 is missing, it uses the mean of February 28 and March 1 when available.

## 7. Inspect outputs

Typical paths are:

```text
work/
├── prepared_daily_data.csv
├── prepared_validation.json
├── historical/
│   ├── daily_classifications.csv
│   ├── spell_events.csv
│   ├── annual_summaries.csv
│   └── *.png
├── model_comparison/
│   ├── max_model_comparison.xlsx
│   ├── max_model_predictions.xlsx
│   ├── min_model_comparison.xlsx
│   └── min_model_predictions.xlsx
├── models/
│   ├── model_selection_explanation.md
│   ├── max/model_manifest.json
│   └── min/model_manifest.json
└── forecast/
    ├── future_max_min_predictions.csv
    ├── future_monthly_summary.csv
    ├── future_annual_summary.csv
    ├── future_max_min_forecast.png
    └── forecast_verification.json
```

The precise model artifact files depend on the selected model. Do not treat generated outputs as a license grant for the input data or trained artifacts.

## 8. Direct script usage

The public facade is recommended for Python callers. From the repository root, first make the `src` directory importable.

### PowerShell

```powershell
$env:PYTHONPATH = "src"
```

### Linux/macOS

```bash
export PYTHONPATH=src
```

The numbered modules remain available for the existing workflow:

```bash
python -m heat_cold_spell.03_train_and_compare --help
python -m heat_cold_spell.04_train_best_model --help
python -m heat_cold_spell.05_predict_future --help
python -m heat_cold_spell.06_pastspells --help
```

`01_prepare_data.py` and `02_historical_analysis.py` prompt interactively for their input path and, for historical analysis, location. They use the paths defined in `src/heat_cold_spell/config.py` and `src/heat_cold_spell/02_historical_analysis.py`; review those constants before using them outside the original environment.

