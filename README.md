# Heat & Cold Spell Analysis and Forecasting Library

A Python library and workflow pipeline for analyzing historical daily extreme temperatures (maximum and minimum), training predictive machine learning models, and identifying heat waves and cold waves under standardized meteorological criteria.

---

## Technical Overview & Architecture

The pipeline processes daily maximum ($T_{\max}$) and minimum ($T_{\min}$) temperatures through a six-stage workflow:

```
+-----------------------------------------------------------------------------------+
|                            1. Data Preparation & Validation                        |
|   Inputs raw daily MAX/MIN data -> Cleans, standardizes dates, and validates schema   |
+-----------------------------------------------------------------------------------+
                                          |
          +-------------------------------+-------------------------------+
          |                                                               |
          v                                                               v
+------------------------------------+                         +------------------------------------+
|  2. Retrospective Historical       |                         |  3. Model Training & Comparison    |
|     Analysis & Spell Detection     |                         |     Evaluates candidate forecasters|
|  - Computes 15-year normals        |                         |     on chronological holdout split |
|  - Identifies heat/cold spells     |                         +------------------------------------+
|  - Generates annual summaries      |                                            |
+------------------------------------+                                            v
                                                               +------------------------------------+
                                                               |  4. Model Refitting & Manifest     |
                                                               |     Refits winning candidate models|
                                                               |     on all available data          |
                                                               +------------------------------------+
                                                                                  |
                                                                                  v
                                                               +------------------------------------+
                                                               |  5. Multi-Year Future Prediction   |
                                                               |     Generates forward predictions   |
                                                               |     (e.g., 5-year horizon)         |
                                                               +------------------------------------+
                                                                                  |
                                                                                  v
                                                               +------------------------------------+
                                                               |  6. Forecast Spell Detection       |
                                                               |     Applies spell rules to future  |
                                                               |     forecasts using prior normals  |
                                                               +------------------------------------+
```

---

## Theoretical & Methodological Foundations

### 1. Climatological Normals & Departures
Daily temperature anomalies are measured relative to **calendar-day climatological normals** calculated across a 15-year baseline window:

* **Historical Analysis**: Uses a 15-year centered moving window ($\pm 7$ years around the given year) to account for multi-decadal trends. A trailing 15-year window is applied to the final 7 historical years.
* **Forecast Spell Detection**: Uses a trailing 15-year window of observed data immediately preceding the forecast period to avoid future data leakage.

Temperature departures (anomalies) are defined as:

$$\Delta T_{\max} = T_{\max} - N_{\max, d}, \quad \Delta T_{\min} = T_{\min} - N_{\min, d}$$

where $N_{\max, d}$ and $N_{\min, d}$ represent the 15-year mean normal temperatures for calendar day $d \in [1, 365]$.

---

### 2. Meteorological Wave & Spell Criteria

#### Daily Classifications

* **Heat Wave Day**: Triggered if either:
  1. **Regional Threshold + Departure**: $T_{\max} \ge T_{\text{regional}}$ **AND** $\Delta T_{\max} \ge +4.5^\circ\text{C}$
     * *Plains*: $T_{\text{regional}} = 40.0^\circ\text{C}$
     * *Coastal*: $T_{\text{regional}} = 37.0^\circ\text{C}$
     * *Hilly*: $T_{\text{regional}} = 30.0^\circ\text{C}$
  2. **Absolute Extreme Threshold**: $T_{\max} \ge 45.0^\circ\text{C}$ (regardless of departure).
* **Cold Wave Day**: Triggered if:
  * $T_{\min} < 10.0^\circ\text{C}$ **AND** $\Delta T_{\min} \le -3.0^\circ\text{C}$

#### Spell Event Aggregation
Consecutive qualifying extreme days are grouped into formal spell events based on duration triggers:

* **Heat Wave Event**: $\ge 5$ consecutive Heat Wave Days **OR** $\ge 2$ consecutive days with $T_{\max} \ge 45.0^\circ\text{C}$.
* **Cold Wave Event**: $\ge 5$ consecutive Cold Wave Days.

---

### 3. Machine Learning Forecasting Pipeline

#### Candidate Model Families
The pipeline evaluates six model architectures separately for $T_{\max}$ and $T_{\min}$:

1. **Seasonal Baseline**: Non-parametric daily climatology mean benchmark.
2. **Trend + Fourier**: Linear trend combined with 6 harmonic sine/cosine annual cycles.
3. **Random Forest**: Non-linear ensemble trees on calendar and rolling lag features.
4. **DNN**: Deep Feed-Forward Neural Network on engineered temporal features.
5. **LSTM**: Long Short-Term Memory recurrent neural network over 60-day sequence lookbacks.
6. **GRU**: Gated Recurrent Unit neural network over 60-day sequence lookbacks.

#### Chronological Holdout Evaluation
Models are evaluated on a chronological test period (`2022-06-01` to `2024-05-31`) to prevent temporal data leakage. Selection prioritizes models using the following metric hierarchy:

$$\text{Rank Criteria: } \text{RMSE} \longrightarrow \text{MAE} \longrightarrow R^2 \longrightarrow |\text{Bias}|$$

---

## Installation & Setup

### Requirements
* Python 3.9+
* Required packages: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `joblib`, `openpyxl`, `matplotlib`

### Directory Structure Requirements
```text
.
├── src/
│   └── heat_cold_spell/
│       ├── __init__.py
│       ├── heat_cold_spell.py        # Primary high-level API wrapper
│       ├── config.py
│       ├── 01_prepare_data.py        # CLI Stage 1: Data preparation
│       ├── 02_historical_analysis.py # CLI Stage 2: Historical analysis
│       ├── 03_train_and_compare.py   # CLI Stage 3: Model comparison
│       ├── 04_train_best_model.py    # CLI Stage 4: Model refitting
│       ├── 05_predict_future.py      # CLI Stage 5: Future prediction
│       ├── 06_pastspells.py          # CLI Stage 6: Spell detection
│       └── pipeline/                 # Internal processing modules
│           ├── __init__.py
│           ├── data.py
│           ├── historical.py
│           └── modeling.py
```

The examples below assume that the `src` directory is available on `PYTHONPATH`.

---

## Python API Usage

The library provides a high-level Functional API via the `heat_cold_spell` package:

```python
import heat_cold_spell as hcs

# 1. Prepare and validate raw observations
data_info = hcs.prepare_data(
    input_path="raw_daily_data.csv",
    output_path="processed_data.csv"
)

# 2. Analyze historical records & generate spell reports + heatmaps
hist_results = hcs.analyze_history(
    input_path="processed_data.csv",
    output_dir="./outputs/historical",
    location="plains"  # Options: 'plains', 'coastal', 'hilly'
)

# 3. Train and compare model candidates on chronological holdout
comparison = hcs.compare_models(
    input_path="processed_data.csv",
    output_dir="./outputs/comparison"
)

# 4. Refit the winning candidate models on all data
models = hcs.train_models(
    input_path="processed_data.csv",
    comparison_dir="./outputs/comparison",
    model_dir="./saved_models"
)

# 5. Forecast future daily MAX/MIN temperatures (e.g., 5 years / 1825 days)
forecasts = hcs.predict_future(
    input_path="processed_data.csv",
    model_dir="./saved_models",
    output_dir="./outputs/forecasts",
    days=1825
)

# 6. Detect heat and cold spells in future forecasts
forecast_spells = hcs.predict_spells(
    input_path="./outputs/forecasts/future_max_min_predictions.csv",
    output_dir="./outputs/forecast_spells",
    location="plains",
    historical_path="processed_data.csv"
)
```

---

## Command-Line Interface (CLI)

Each pipeline stage can also be executed as a module from the repository root.
First make the `src` directory importable.

### PowerShell

```powershell
$env:PYTHONPATH = "src"
```

### Linux/macOS

```bash
export PYTHONPATH=src
```

Then run the pipeline stages:

```bash
# Stage 1: Validate and prepare daily CSV/Excel data
python -m heat_cold_spell.01_prepare_data

# Stage 2: Analyze historical trends, heatmaps, and spell events
python -m heat_cold_spell.02_historical_analysis

# Stage 3: Train and evaluate candidate forecasters
python -m heat_cold_spell.03_train_and_compare \
  --input data/processed/processed_daily_data.csv \
  --output-dir outputs/comparison

# Stage 4: Refit the top-ranked models on the full dataset
python -m heat_cold_spell.04_train_best_model \
  --input data/processed/processed_daily_data.csv \
  --comparison-dir outputs/comparison \
  --model-dir saved_models

# Stage 5: Generate multi-year future predictions
python -m heat_cold_spell.05_predict_future \
  --input data/processed/processed_daily_data.csv \
  --model-dir saved_models \
  --output-dir outputs/forecasts \
  --forecast-days 1825

# Stage 6: Detect heat and cold spells
python -m heat_cold_spell.06_pastspells
```

---

## Output Artefacts

| Module / Script | Generated Artefacts |
| :--- | :--- |
| `01_prepare_data.py` | `processed_daily_data.csv`, `validation_report.json` |
| `02_historical_analysis.py` | `daily_classifications.csv`, `spell_events.csv`, `annual_summaries.csv`, `monthly_*_heatmap.png`, `yearly_spell_occurrences.png` |
| `03_train_and_compare.py` | `{max,min}_model_comparison.xlsx`, `{max,min}_model_predictions.xlsx` |
| `04_train_best_model.py` | `{max,min}/model_manifest.json`, serialized model weights (`model.keras` or `model.joblib`), `model_selection_explanation.md` |
| `05_predict_future.py` | `future_max_min_predictions.csv`, `future_monthly_summary.csv`, `future_annual_summary.csv`, `forecast_verification.json`, `future_max_min_forecast.png` |
| `06_pastspells.py` | Daily spell classification and spell-event output files |
| `predict_spells` | `daily_spell_classification.csv`, `spell_events.csv` |
