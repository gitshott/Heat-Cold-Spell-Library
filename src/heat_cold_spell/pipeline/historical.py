"""Historical T2MWET patterns, thresholds, trends, and spell detection."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


SEASON_BY_MONTH = {
    1: "Winter",
    2: "Winter",
    3: "Pre-monsoon",
    4: "Pre-monsoon",
    5: "Pre-monsoon",
    6: "Monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Post-monsoon",
    11: "Post-monsoon",
    12: "Winter",
}
SEASON_ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]


def calendar_day_365(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Map dates to a stable 1..365 calendar, with February 29 sharing day 59."""
    index = pd.DatetimeIndex(dates)
    result = index.dayofyear.to_numpy(dtype=int)
    february_29 = (index.month == 2) & (index.day == 29)
    after_february = index.is_leap_year & (index.month > 2)
    result[february_29] = 59
    result[after_february] -= 1
    return result


def calculate_calendar_thresholds(
    data: pd.DataFrame,
    reference_start_year: int,
    reference_end_year: int,
    cold_percentile: float,
    heat_percentile: float,
    window_half_width_days: int,
) -> pd.DataFrame:
    """Calculate smoothed calendar-day thresholds from a fixed reference period."""
    reference = data[data["YEAR"].between(reference_start_year, reference_end_year)].copy()
    if reference["YEAR"].nunique() < 10:
        raise ValueError("At least ten reference years are required for stable thresholds")

    reference["CLIMATE_DAY"] = calendar_day_365(reference["DATE"])
    reference_days = reference["CLIMATE_DAY"].to_numpy()
    reference_values = reference["T2MWET"].to_numpy(dtype=float)
    representative_dates = pd.date_range("2001-01-01", "2001-12-31", freq="D")

    rows: list[dict[str, float | int]] = []
    for climate_day in range(1, 366):
        direct_distance = np.abs(reference_days - climate_day)
        circular_distance = np.minimum(direct_distance, 365 - direct_distance)
        sample = reference_values[circular_distance <= window_half_width_days]
        representative = representative_dates[climate_day - 1]
        rows.append(
            {
                "CLIMATE_DAY": climate_day,
                "MONTH": int(representative.month),
                "DAY": int(representative.day),
                "COLD_THRESHOLD_P10": float(np.quantile(sample, cold_percentile)),
                "MEDIAN_T2MWET": float(np.quantile(sample, 0.50)),
                "HEAT_THRESHOLD_P90": float(np.quantile(sample, heat_percentile)),
                "REFERENCE_SAMPLE_COUNT": int(len(sample)),
            }
        )
    return pd.DataFrame(rows)


def qualifying_runs(condition: np.ndarray, minimum_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Mark every day in qualifying consecutive runs and assign event numbers."""
    condition = np.asarray(condition, dtype=bool)
    membership = np.zeros(len(condition), dtype=bool)
    event_ids = np.zeros(len(condition), dtype=int)
    event_number = 0
    start = 0

    while start < len(condition):
        if not condition[start]:
            start += 1
            continue
        end = start + 1
        while end < len(condition) and condition[end]:
            end += 1
        if end - start >= minimum_days:
            event_number += 1
            membership[start:end] = True
            event_ids[start:end] = event_number
        start = end

    return membership, event_ids


def label_historical_spells(
    data: pd.DataFrame,
    thresholds: pd.DataFrame,
    minimum_spell_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach thresholds to daily data and describe all qualifying spell events."""
    daily = data.copy().sort_values("DATE").reset_index(drop=True)
    daily["DAY_OF_YEAR"] = daily["DATE"].dt.dayofyear
    daily["CLIMATE_DAY"] = calendar_day_365(daily["DATE"])
    daily["SEASON"] = daily["MONTH"].map(SEASON_BY_MONTH)
    daily = daily.merge(
        thresholds[["CLIMATE_DAY", "COLD_THRESHOLD_P10", "HEAT_THRESHOLD_P90"]],
        on="CLIMATE_DAY",
        how="left",
        validate="many_to_one",
    )

    daily["WARM_EXCEEDANCE"] = daily["T2MWET"] > daily["HEAT_THRESHOLD_P90"]
    daily["COLD_EXCEEDANCE"] = daily["T2MWET"] < daily["COLD_THRESHOLD_P10"]
    daily["HEAT_THRESHOLD_DIFFERENCE"] = daily["T2MWET"] - daily["HEAT_THRESHOLD_P90"]
    daily["COLD_THRESHOLD_DIFFERENCE"] = daily["COLD_THRESHOLD_P10"] - daily["T2MWET"]
    daily["HEAT_SPELL"], daily["HEAT_EVENT_ID"] = qualifying_runs(
        daily["WARM_EXCEEDANCE"].to_numpy(), minimum_spell_days
    )
    daily["COLD_SPELL"], daily["COLD_EVENT_ID"] = qualifying_runs(
        daily["COLD_EXCEEDANCE"].to_numpy(), minimum_spell_days
    )

    event_rows: list[dict[str, object]] = []
    definitions = [
        (
            "heat",
            "HEAT_EVENT_ID",
            "HEAT_THRESHOLD_P90",
            "HEAT_THRESHOLD_DIFFERENCE",
        ),
        (
            "cold",
            "COLD_EVENT_ID",
            "COLD_THRESHOLD_P10",
            "COLD_THRESHOLD_DIFFERENCE",
        ),
    ]
    for spell_type, event_column, threshold_column, difference_column in definitions:
        qualifying = daily[daily[event_column] > 0]
        for event_id, group in qualifying.groupby(event_column, sort=True):
            start_date = group["DATE"].iloc[0]
            end_date = group["DATE"].iloc[-1]
            event_rows.append(
                {
                    "SPELL_TYPE": spell_type,
                    "EVENT_ID": int(event_id),
                    "START_DATE": start_date,
                    "END_DATE": end_date,
                    "START_YEAR": int(start_date.year),
                    "START_MONTH": int(start_date.month),
                    "START_SEASON": SEASON_BY_MONTH[int(start_date.month)],
                    "DURATION_DAYS": int(len(group)),
                    "MEAN_T2MWET": float(group["T2MWET"].mean()),
                    "MIN_T2MWET": float(group["T2MWET"].min()),
                    "MAX_T2MWET": float(group["T2MWET"].max()),
                    "MEAN_THRESHOLD": float(group[threshold_column].mean()),
                    "MEAN_THRESHOLD_EXCEEDANCE": float(group[difference_column].mean()),
                    "MAX_THRESHOLD_EXCEEDANCE": float(group[difference_column].max()),
                }
            )

    event_columns = [
        "SPELL_TYPE",
        "EVENT_ID",
        "START_DATE",
        "END_DATE",
        "START_YEAR",
        "START_MONTH",
        "START_SEASON",
        "DURATION_DAYS",
        "MEAN_T2MWET",
        "MIN_T2MWET",
        "MAX_T2MWET",
        "MEAN_THRESHOLD",
        "MEAN_THRESHOLD_EXCEEDANCE",
        "MAX_THRESHOLD_EXCEEDANCE",
    ]
    events = pd.DataFrame(event_rows, columns=event_columns)
    return daily, events


def _group_summary(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = data.groupby(group_columns, observed=True, sort=True)
    summary = grouped.agg(
        OBSERVATION_COUNT=("T2MWET", "size"),
        MEAN_T2MWET=("T2MWET", "mean"),
        MEDIAN_T2MWET=("T2MWET", "median"),
        MIN_T2MWET=("T2MWET", "min"),
        MAX_T2MWET=("T2MWET", "max"),
        STD_T2MWET=("T2MWET", "std"),
        WARM_EXCEEDANCE_DAYS=("WARM_EXCEEDANCE", "sum"),
        COLD_EXCEEDANCE_DAYS=("COLD_EXCEEDANCE", "sum"),
        HEAT_SPELL_DAYS=("HEAT_SPELL", "sum"),
        COLD_SPELL_DAYS=("COLD_SPELL", "sum"),
    )
    quantiles = grouped["T2MWET"].quantile([0.10, 0.90]).unstack(-1)
    summary["P10_T2MWET"] = quantiles[0.10]
    summary["P90_T2MWET"] = quantiles[0.90]
    return summary.reset_index()


def _attach_event_counts(
    summary: pd.DataFrame,
    events: pd.DataFrame,
    summary_keys: list[str],
    event_key_map: dict[str, str],
) -> pd.DataFrame:
    if events.empty:
        summary["HEAT_EVENTS"] = 0
        summary["COLD_EVENTS"] = 0
        return summary

    event_copy = events.copy()
    event_group_keys = []
    for summary_key, event_key in event_key_map.items():
        event_copy[summary_key] = event_copy[event_key]
        event_group_keys.append(summary_key)
    counts = (
        event_copy.groupby(event_group_keys + ["SPELL_TYPE"], observed=True)
        .size()
        .unstack("SPELL_TYPE", fill_value=0)
        .reset_index()
        .rename(columns={"heat": "HEAT_EVENTS", "cold": "COLD_EVENTS"})
    )
    result = summary.merge(counts, on=summary_keys, how="left")
    for column in ["HEAT_EVENTS", "COLD_EVENTS"]:
        if column not in result:
            result[column] = 0
        result[column] = result[column].fillna(0).astype(int)
    return result


def build_historical_summaries(
    daily: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create monthly, year-month, seasonal, and annual summary tables."""
    monthly = _group_summary(daily, ["MONTH"])
    monthly.insert(1, "MONTH_NAME", pd.to_datetime(monthly["MONTH"], format="%m").dt.month_name())
    monthly = _attach_event_counts(
        monthly,
        events,
        ["MONTH"],
        {"MONTH": "START_MONTH"},
    )

    year_month = _group_summary(daily, ["YEAR", "MONTH"])
    year_month.insert(
        2,
        "MONTH_NAME",
        pd.to_datetime(year_month["MONTH"], format="%m").dt.month_name(),
    )
    year_month = _attach_event_counts(
        year_month,
        events,
        ["YEAR", "MONTH"],
        {"YEAR": "START_YEAR", "MONTH": "START_MONTH"},
    )

    seasonal_data = daily.copy()
    seasonal_data["SEASON"] = pd.Categorical(
        seasonal_data["SEASON"], categories=SEASON_ORDER, ordered=True
    )
    seasonal = _group_summary(seasonal_data, ["SEASON"])
    seasonal = _attach_event_counts(
        seasonal,
        events,
        ["SEASON"],
        {"SEASON": "START_SEASON"},
    )

    annual = _group_summary(daily, ["YEAR"])
    expected_days = annual["YEAR"].map(
        lambda year: int(pd.Timestamp(year=int(year), month=12, day=31).dayofyear)
    )
    observed_bounds = daily.groupby("YEAR")["DATE"].agg(["min", "max"]).reset_index()
    annual = annual.merge(observed_bounds, on="YEAR", how="left")
    annual.insert(2, "EXPECTED_DAYS", expected_days)
    annual.insert(
        3,
        "COMPLETE_YEAR",
        (annual["OBSERVATION_COUNT"] == annual["EXPECTED_DAYS"])
        & (annual["min"].dt.month == 1)
        & (annual["min"].dt.day == 1)
        & (annual["max"].dt.month == 12)
        & (annual["max"].dt.day == 31),
    )
    annual = annual.drop(columns=["min", "max"])
    annual = _attach_event_counts(
        annual,
        events,
        ["YEAR"],
        {"YEAR": "START_YEAR"},
    )

    if events.empty:
        annual["LONGEST_HEAT_SPELL"] = 0
        annual["LONGEST_COLD_SPELL"] = 0
    else:
        longest = (
            events.groupby(["START_YEAR", "SPELL_TYPE"])["DURATION_DAYS"]
            .max()
            .unstack("SPELL_TYPE", fill_value=0)
            .reset_index()
            .rename(
                columns={
                    "START_YEAR": "YEAR",
                    "heat": "LONGEST_HEAT_SPELL",
                    "cold": "LONGEST_COLD_SPELL",
                }
            )
        )
        annual = annual.merge(longest, on="YEAR", how="left")
        for column in ["LONGEST_HEAT_SPELL", "LONGEST_COLD_SPELL"]:
            if column not in annual:
                annual[column] = 0
            annual[column] = annual[column].fillna(0).astype(int)

    return monthly, year_month, seasonal, annual


def calculate_historical_trends(annual: pd.DataFrame) -> pd.DataFrame:
    """Calculate linear trends using complete years only."""
    complete = annual[annual["COMPLETE_YEAR"]].copy()
    metric_columns = [
        "MEAN_T2MWET",
        "WARM_EXCEEDANCE_DAYS",
        "COLD_EXCEEDANCE_DAYS",
        "HEAT_SPELL_DAYS",
        "COLD_SPELL_DAYS",
        "HEAT_EVENTS",
        "COLD_EVENTS",
    ]
    rows: list[dict[str, float | int | str]] = []
    years = complete["YEAR"].to_numpy(dtype=float)
    for metric in metric_columns:
        result = linregress(years, complete[metric].to_numpy(dtype=float))
        rows.append(
            {
                "METRIC": metric,
                "N_COMPLETE_YEARS": int(len(complete)),
                "START_YEAR": int(complete["YEAR"].min()),
                "END_YEAR": int(complete["YEAR"].max()),
                "SLOPE_PER_YEAR": float(result.slope),
                "SLOPE_PER_DECADE": float(result.slope * 10.0),
                "INTERCEPT": float(result.intercept),
                "R2": float(result.rvalue**2),
                "P_VALUE": float(result.pvalue),
                "STANDARD_ERROR": float(result.stderr),
            }
        )
    return pd.DataFrame(rows)


def create_historical_figures(
    daily: pd.DataFrame,
    thresholds: pd.DataFrame,
    monthly: pd.DataFrame,
    seasonal: pd.DataFrame,
    annual: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Create the six Stage 2 historical-analysis figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    heat_color = "#D94841"
    cold_color = "#277DA1"
    neutral_color = "#34495E"
    created: list[Path] = []

    figure, axis = plt.subplots(figsize=(15, 6))
    axis.plot(daily["DATE"], daily["T2MWET"], color="#9AA5B1", linewidth=0.35, alpha=0.7, label="Daily T2MWET")
    rolling = daily.set_index("DATE")["T2MWET"].rolling(30, center=True).mean()
    axis.plot(rolling.index, rolling, color=neutral_color, linewidth=1.1, label="30-day mean")
    axis.set(title="Historical daily wet-bulb temperature", xlabel="Date", ylabel="T2MWET (°C)")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    path = output_dir / "historical_temperature_series.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    created.append(path)

    figure, axis = plt.subplots(figsize=(11, 6))
    x = monthly["MONTH"].to_numpy()
    axis.plot(x, monthly["MEAN_T2MWET"], marker="o", color=neutral_color, label="Mean")
    axis.fill_between(x, monthly["P10_T2MWET"], monthly["P90_T2MWET"], color="#F4A261", alpha=0.25, label="P10–P90")
    axis.set_xticks(x, monthly["MONTH_NAME"], rotation=30)
    axis.set(title="Historical monthly T2MWET pattern", xlabel="Month", ylabel="T2MWET (°C)")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    path = output_dir / "monthly_temperature_patterns.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    created.append(path)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.bar(seasonal["SEASON"].astype(str), seasonal["MEAN_T2MWET"], color=[cold_color, "#F4A261", "#2A9D8F", "#8E6C8A"])
    axis.errorbar(
        seasonal["SEASON"].astype(str),
        seasonal["MEAN_T2MWET"],
        yerr=seasonal["STD_T2MWET"],
        fmt="none",
        ecolor="#222222",
        capsize=4,
    )
    axis.set(title="Historical seasonal T2MWET mean and variability", xlabel="Season", ylabel="T2MWET (°C)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    path = output_dir / "seasonal_temperature_patterns.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    created.append(path)

    figure, axis = plt.subplots(figsize=(13, 6))
    x = thresholds["CLIMATE_DAY"].to_numpy()
    axis.plot(x, thresholds["MEDIAN_T2MWET"], color=neutral_color, label="Median")
    axis.plot(x, thresholds["HEAT_THRESHOLD_P90"], color=heat_color, label="Heat threshold P90")
    axis.plot(x, thresholds["COLD_THRESHOLD_P10"], color=cold_color, label="Cold threshold P10")
    axis.fill_between(x, thresholds["COLD_THRESHOLD_P10"], thresholds["HEAT_THRESHOLD_P90"], color="#B8C0C8", alpha=0.2)
    axis.set(title="Fixed calendar-day T2MWET thresholds (2006–2020)", xlabel="Climate day", ylabel="T2MWET (°C)")
    axis.legend(frameon=False, ncol=3)
    axis.grid(alpha=0.2)
    path = output_dir / "calendar_day_thresholds.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    created.append(path)

    figure, axis = plt.subplots(figsize=(15, 6))
    rolling = daily.set_index("DATE")["T2MWET"].rolling(15, center=True).mean()
    axis.plot(rolling.index, rolling, color="#A5ADB5", linewidth=0.8, label="15-day mean")
    heat_days = daily[daily["HEAT_SPELL"]]
    cold_days = daily[daily["COLD_SPELL"]]
    axis.scatter(heat_days["DATE"], heat_days["T2MWET"], s=8, color=heat_color, label="Heat-spell days")
    axis.scatter(cold_days["DATE"], cold_days["T2MWET"], s=8, color=cold_color, label="Cold-spell days")
    axis.set(title="Historical five-day relative heat and cold spells", xlabel="Date", ylabel="T2MWET (°C)")
    axis.legend(frameon=False, ncol=3)
    axis.grid(alpha=0.2)
    path = output_dir / "historical_spell_timeline.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    created.append(path)

    complete = annual[annual["COMPLETE_YEAR"]]
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(complete["YEAR"], complete["MEAN_T2MWET"], marker="o", color=neutral_color, label="Annual mean")
    trend = linregress(complete["YEAR"], complete["MEAN_T2MWET"])
    axes[0].plot(complete["YEAR"], trend.intercept + trend.slope * complete["YEAR"], linestyle="--", color=heat_color, label=f"Trend {trend.slope * 10:+.2f} °C/decade")
    axes[0].set(ylabel="Mean T2MWET (°C)", title="Annual mean and historical spell-day totals")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].bar(complete["YEAR"] - 0.2, complete["HEAT_SPELL_DAYS"], width=0.4, color=heat_color, label="Heat-spell days")
    axes[1].bar(complete["YEAR"] + 0.2, complete["COLD_SPELL_DAYS"], width=0.4, color=cold_color, label="Cold-spell days")
    axes[1].set(xlabel="Year", ylabel="Days", xticks=complete["YEAR"])
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    path = output_dir / "annual_temperature_and_spell_trends.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    created.append(path)

    return created


def verify_historical_analysis(
    daily: pd.DataFrame,
    thresholds: pd.DataFrame,
    events: pd.DataFrame,
    annual: pd.DataFrame,
    minimum_spell_days: int,
) -> dict[str, object]:
    """Fail loudly if Stage 2 results violate key consistency rules."""
    if len(thresholds) != 365 or set(thresholds["CLIMATE_DAY"]) != set(range(1, 366)):
        raise AssertionError("Calendar threshold table must contain exactly climate days 1..365")
    if thresholds[["COLD_THRESHOLD_P10", "HEAT_THRESHOLD_P90"]].isna().any().any():
        raise AssertionError("Calendar threshold table contains missing thresholds")
    if (thresholds["COLD_THRESHOLD_P10"] >= thresholds["HEAT_THRESHOLD_P90"]).any():
        raise AssertionError("At least one cold threshold is not below its heat threshold")
    if daily[["COLD_THRESHOLD_P10", "HEAT_THRESHOLD_P90"]].isna().any().any():
        raise AssertionError("At least one historical date has a missing threshold")
    if not daily["DATE"].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise AssertionError("Historical daily dates are not continuous")
    if not events.empty:
        if (events["DURATION_DAYS"] < minimum_spell_days).any():
            raise AssertionError("An event shorter than the minimum spell length was exported")
        date_durations = (events["END_DATE"] - events["START_DATE"]).dt.days + 1
        if not date_durations.eq(events["DURATION_DAYS"]).all():
            raise AssertionError("At least one event duration disagrees with its dates")

    daily_heat_days = int(daily["HEAT_SPELL"].sum())
    daily_cold_days = int(daily["COLD_SPELL"].sum())
    if int(annual["HEAT_SPELL_DAYS"].sum()) != daily_heat_days:
        raise AssertionError("Annual heat-spell days do not reconcile to daily labels")
    if int(annual["COLD_SPELL_DAYS"].sum()) != daily_cold_days:
        raise AssertionError("Annual cold-spell days do not reconcile to daily labels")

    return {
        "status": "passed",
        "threshold_rows": int(len(thresholds)),
        "daily_rows": int(len(daily)),
        "heat_events": int((events["SPELL_TYPE"] == "heat").sum()),
        "cold_events": int((events["SPELL_TYPE"] == "cold").sum()),
        "heat_spell_days": daily_heat_days,
        "cold_spell_days": daily_cold_days,
        "minimum_spell_days": int(minimum_spell_days),
        "missing_daily_thresholds": 0,
        "daily_dates_continuous": True,
        "annual_spell_days_reconciled": True,
    }
