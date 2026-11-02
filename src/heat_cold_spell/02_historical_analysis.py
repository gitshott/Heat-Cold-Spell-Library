"""Historical heat and cold spell analysis for YEAR, MN, DT, MAX, MIN data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "historical_analysis"

MINIMUM_SPELL_DAYS = 5
EXTREME_HEAT_TMAX_C = 45.0
HEAT_DEPARTURE_C = 4.5
COLD_LIMIT_C = 10.0
COLD_DEPARTURE_C = -3.0

# Minimum actual MAX for the regional heat-wave departure rule.
HEAT_MIN_TMAX_C = {
    "plains": 40.0,
    "coastal": 37.0,
    "hilly": 30.0,
}

def load_data(path: Path) -> pd.DataFrame:
    """Load CSV or Excel data, validate columns, and combine duplicate dates."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(
            "Input must be CSV or Excel (.csv, .xlsx, .xls, .xlsm)."
        )

    df.columns = [str(column).strip().upper() for column in df.columns]

    required = ["YEAR", "MN", "DT", "MAX", "MIN"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["DATE"] = pd.to_datetime(
        {
            "year": df["YEAR"],
            "month": df["MN"],
            "day": df["DT"],
        },
        errors="coerce",
    )

    invalid_dates = int(df["DATE"].isna().sum())
    df = df.dropna(subset=["DATE"]).copy()
    if df.empty:
        raise ValueError("No valid dates could be created from YEAR, MN, and DT.")

    duplicate_count = int(df["DATE"].duplicated().sum())

    # Average duplicate records for each date.
    df = (
        df.groupby("DATE", as_index=False)
        .agg(MAX=("MAX", "mean"), MIN=("MIN", "mean"))
    )

    df["YEAR"] = df["DATE"].dt.year
    df["MN"] = df["DATE"].dt.month
    df["DT"] = df["DATE"].dt.day
    df = df.sort_values("DATE").reset_index(drop=True)

    if df[["MAX", "MIN"]].isna().all().all():
        raise ValueError("No numeric values were found in MAX or MIN.")

    if invalid_dates:
        print(f"Skipped {invalid_dates} row(s) with invalid dates.")
    if duplicate_count:
        print(f"Averaged duplicate records on {duplicate_count} date(s).")

    return df



def calculate_normals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate calendar-day normals using centered 15-year windows."""
    first_year = int(df["YEAR"].min())
    last_year = int(df["YEAR"].max())
    years = sorted(df["YEAR"].unique())

    # Use a trailing 15-year baseline for the final seven years.
    trailing_start = max(first_year, last_year - 14)
    trailing_years = set(range(trailing_start, last_year + 1))

    normal_rows = []

    for target_year in years:
        if target_year > last_year - 7:
            window_years = trailing_years
        else:
            start_year = max(first_year, int(target_year) - 7)
            end_year = min(last_year, int(target_year) + 7)
            window_years = set(range(start_year, end_year + 1))

        window = df[df["YEAR"].isin(window_years)].copy()
        window["CALENDAR_DAY"] = window["DATE"].dt.strftime("%m-%d")

        normals = (
            window.groupby("CALENDAR_DAY", as_index=False)
            .agg(
                NORMAL_MAX=("MAX", "mean"),
                NORMAL_MIN=("MIN", "mean"),
                MAX_NORMAL_OBSERVATIONS=("MAX", "count"),
                MIN_NORMAL_OBSERVATIONS=("MIN", "count"),
            )
        )
        normals["YEAR"] = int(target_year)
        normal_rows.append(normals)

    if not normal_rows:
        raise ValueError("Could not calculate normals from the input data.")

    return pd.concat(normal_rows, ignore_index=True)



def classify_daily_data(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """Add calendar-day normals, departures, and daily classifications."""
    location = location.strip().lower()
    if location not in HEAT_MIN_TMAX_C:
        raise ValueError(
            "Location must be one of: "
            + ", ".join(HEAT_MIN_TMAX_C)
        )

    result = df.copy()
    result["CALENDAR_DAY"] = result["DATE"].dt.strftime("%m-%d")

    normals = calculate_normals(result)
    result = result.merge(
        normals,
        on=["YEAR", "CALENDAR_DAY"],
        how="left",
        validate="many_to_one",
    )

    result["MAX_DEPARTURE"] = result["MAX"] - result["NORMAL_MAX"]
    result["MIN_DEPARTURE"] = result["MIN"] - result["NORMAL_MIN"]

    max_valid = result["MAX"].notna() & result["NORMAL_MAX"].notna()
    min_valid = result["MIN"].notna() & result["NORMAL_MIN"].notna()

    regional_threshold = HEAT_MIN_TMAX_C[location]

    # Heat day: (regional threshold and >= 4.5 C above normal) OR MAX >= 45 C.
    heat_day = max_valid & (
        (
            (result["MAX"] >= regional_threshold)
            & (result["MAX_DEPARTURE"] >= HEAT_DEPARTURE_C)
        )
        | (result["MAX"] >= EXTREME_HEAT_TMAX_C)
    )

    # Cold day: MIN < 10 C and at least 3 C below normal.
    cold_day = (
        min_valid
        & (result["MIN"] < COLD_LIMIT_C)
        & (result["MIN_DEPARTURE"] <= COLD_DEPARTURE_C)
    )

    result["HEAT_CLASSIFICATION"] = "Normal"
    result["COLD_CLASSIFICATION"] = "Normal"
    result.loc[~max_valid, "HEAT_CLASSIFICATION"] = "Missing"
    result.loc[~min_valid, "COLD_CLASSIFICATION"] = "Missing"
    result.loc[heat_day, "HEAT_CLASSIFICATION"] = "Heat Wave Day"
    result.loc[cold_day, "COLD_CLASSIFICATION"] = "Cold Wave Day"

    result["CLASSIFICATION"] = "Normal"
    result.loc[~max_valid & ~min_valid, "CLASSIFICATION"] = "Missing"
    result.loc[cold_day, "CLASSIFICATION"] = "Cold Wave Day"
    result.loc[heat_day, "CLASSIFICATION"] = "Heat Wave Day"

    return result.sort_values("DATE").reset_index(drop=True)



def add_event(
    events: list[dict],
    year: int,
    spell: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    trigger: str,
) -> None:
    events.append({
        "Year": int(year),
        "Spell": spell,
        "From Date": start.strftime("%Y-%m-%d"),
        "To Date": end.strftime("%Y-%m-%d"),
        "Duration Days": (end - start).days + 1,
        "Trigger": trigger,
    })



def qualifying_runs(
    group: pd.DataFrame,
    mask: pd.Series,
):
    """Yield qualifying consecutive date ranges; gaps break runs."""
    run_start = None
    previous_date = None

    for date, qualifies in zip(group["DATE"], mask):
        consecutive = (
            previous_date is not None
            and (date - previous_date).days == 1
        )

        if run_start is not None and (not qualifies or not consecutive):
            yield run_start, previous_date
            run_start = None

        if qualifies and run_start is None:
            run_start = date

        previous_date = date

    if run_start is not None:
        yield run_start, previous_date



def detect_spells(df: pd.DataFrame) -> pd.DataFrame:
    """Detect heat and cold spells without duplicate overlapping heat events."""
    events = []

    for year, group in df.groupby("YEAR", sort=True):
        group = group.sort_values("DATE").reset_index(drop=True)

        # Heat classification already includes MAX >= 45 C. For each
        # continuous heat run, record it once if either spell rule applies.
        heat_mask = group["HEAT_CLASSIFICATION"].eq("Heat Wave Day")
        extreme_mask = group["MAX"].notna() & (
            group["MAX"] >= EXTREME_HEAT_TMAX_C
        )

        for start, end in qualifying_runs(group, heat_mask):
            run = group[
                group["DATE"].between(start, end)
            ]
            duration = (end - start).days + 1
            has_two_consecutive_extreme_days = any(
                (right - left).days == 1
                for left, right in zip(
                    run.loc[extreme_mask, "DATE"],
                    run.loc[extreme_mask, "DATE"].iloc[1:],
                )
            )

            triggers = []
            if duration >= MINIMUM_SPELL_DAYS:
                triggers.append("5 or more consecutive heat-wave days")
            if has_two_consecutive_extreme_days:
                triggers.append(
                    "2 or more consecutive days with MAX >= 45 C"
                )

            if triggers:
                add_event(
                    events,
                    year,
                    "Heat Wave",
                    start,
                    end,
                    "; ".join(triggers),
                )

        # Cold spells: five or more consecutive cold-wave days.
        cold_mask = group["COLD_CLASSIFICATION"].eq("Cold Wave Day")
        for start, end in qualifying_runs(group, cold_mask):
            duration = (end - start).days + 1
            if duration >= MINIMUM_SPELL_DAYS:
                add_event(
                    events,
                    year,
                    "Cold Wave",
                    start,
                    end,
                    "5 or more consecutive cold-wave days",
                )

    columns = [
        "Year",
        "Spell",
        "From Date",
        "To Date",
        "Duration Days",
        "Trigger",
    ]
    result = pd.DataFrame(events, columns=columns)

    if not result.empty:
        result = result.sort_values(
            ["Year", "From Date", "Spell"]
        ).reset_index(drop=True)

    return result



def create_annual_summary(
    daily: pd.DataFrame,
    spells: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for year, group in daily.groupby("YEAR", sort=True):
        year_spells = spells[spells["Year"] == year]

        heat_spells = year_spells[year_spells["Spell"] == "Heat Wave"]
        cold_spells = year_spells[year_spells["Spell"] == "Cold Wave"]

        heat_durations = heat_spells["Duration Days"]
        cold_durations = cold_spells["Duration Days"]

        rows.append({
            "YEAR": int(year),
            "OBSERVATION_DAYS": int(len(group)),
            "MEAN_MAX": (
                round(group["MAX"].mean(), 2)
                if group["MAX"].notna().any() else np.nan
            ),
            "MEAN_MIN": (
                round(group["MIN"].mean(), 2)
                if group["MIN"].notna().any() else np.nan
            ),
            "HEAT_WAVE_DAYS": int(
                group["HEAT_CLASSIFICATION"].eq("Heat Wave Day").sum()
            ),
            "COLD_WAVE_DAYS": int(
                group["COLD_CLASSIFICATION"].eq("Cold Wave Day").sum()
            ),
            "HEAT_SPELLS": int(len(heat_spells)),
            "COLD_SPELLS": int(len(cold_spells)),
            "LONGEST_HEAT_SPELL_DAYS": (
                int(heat_durations.max()) if not heat_durations.empty else 0
            ),
            "LONGEST_COLD_SPELL_DAYS": (
                int(cold_durations.max()) if not cold_durations.empty else 0
            ),
        })

    return pd.DataFrame(rows)



def save_heatmap(
    monthly: pd.DataFrame,
    value_column: str,
    title: str,
    colorbar_label: str,
    output_path: Path,
    cmap: str,
    value_format: str,
) -> None:
    """Save a month-by-year heatmap as a PNG."""
    if monthly.empty:
        print(f"Skipped empty chart: {title}")
        return

    pivot = (
        monthly.pivot(index="YEAR", columns="MN", values=value_column)
        .reindex(columns=range(1, 13))
        .sort_index()
    )

    values = pivot.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)

    fig_height = max(5, min(14, 0.35 * len(pivot.index) + 2))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    image = ax.imshow(
        masked_values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        origin="lower",
    )

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_xticks(range(12))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int))
    ax.tick_params(axis="y", labelsize=8)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    # Add values to cells when the chart is small enough to remain readable.
    if len(pivot.index) <= 25:
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if np.isfinite(value):
                    ax.text(
                        column_index,
                        row_index,
                        format(value, value_format),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def create_monthly_charts(daily: pd.DataFrame) -> None:
    """Create monthly MAX, heat-day, and cold-day heatmaps."""
    monthly = (
        daily.assign(
            HEAT_WAVE_DAY=daily["HEAT_CLASSIFICATION"].eq("Heat Wave Day"),
            COLD_WAVE_DAY=daily["COLD_CLASSIFICATION"].eq("Cold Wave Day"),
        )
        .groupby(["YEAR", "MN"], as_index=False)
        .agg(
            MONTHLY_MEAN_MAX=("MAX", "mean"),
            HEAT_WAVE_DAYS=("HEAT_WAVE_DAY", "sum"),
            COLD_WAVE_DAYS=("COLD_WAVE_DAY", "sum"),
        )
    )

    save_heatmap(
        monthly,
        "MONTHLY_MEAN_MAX",
        "Monthly Mean Daily Maximum Temperature",
        "Mean MAX temperature (°C)",
        OUTPUT_DIR / "monthly_mean_max_heatmap.png",
        "YlOrRd",
        ".1f",
    )
    save_heatmap(
        monthly,
        "HEAT_WAVE_DAYS",
        "Monthly Heat-Wave Days",
        "Heat-wave days",
        OUTPUT_DIR / "monthly_heat_wave_days_heatmap.png",
        "YlOrRd",
        ".0f",
    )
    save_heatmap(
        monthly,
        "COLD_WAVE_DAYS",
        "Monthly Cold-Wave Days",
        "Cold-wave days",
        OUTPUT_DIR / "monthly_cold_wave_days_heatmap.png",
        "Blues",
        ".0f",
    )



def create_yearly_spell_chart(annual: pd.DataFrame) -> Path:
    """Save a grouped bar chart of heat- and cold-spell counts by year."""
    chart_path = OUTPUT_DIR / "yearly_spell_occurrences.png"
    fig_width = max(10, min(20, len(annual) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    positions = np.arange(len(annual))
    bar_width = 0.4
    ax.bar(
        positions - bar_width / 2,
        annual["HEAT_SPELLS"],
        bar_width,
        label="Heat waves",
        color="#dc2626",
    )
    ax.bar(
        positions + bar_width / 2,
        annual["COLD_SPELLS"],
        bar_width,
        label="Cold waves",
        color="#2563eb",
    )
    ax.set_title("Yearly Occurrence of Heat and Cold Spells")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of spells")
    ax.set_xticks(positions, annual["YEAR"].astype(int), rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return chart_path



def main() -> None:
    entered_path = input(
        "Enter input CSV/Excel file path: "
    ).strip().strip('"').strip("'")

    if not entered_path:
        raise ValueError("An input file path is required.")

    location = input(
        "Station location (plains, coastal, or hilly): "
    ).strip().lower()

    if location not in HEAT_MIN_TMAX_C:
        raise ValueError(
            "Location must be one of: "
            + ", ".join(HEAT_MIN_TMAX_C)
        )

    source = Path(entered_path).expanduser()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_data(source)
    daily = classify_daily_data(raw, location)
    spells = detect_spells(daily)
    annual = create_annual_summary(daily, spells)

    # Save the three requested data files.
    daily_path = OUTPUT_DIR / "daily_classifications.csv"
    spells_path = OUTPUT_DIR / "spell_events.csv"
    annual_path = OUTPUT_DIR / "annual_summaries.csv"

    daily.to_csv(daily_path, index=False, date_format="%Y-%m-%d")
    spells.to_csv(spells_path, index=False)
    annual.to_csv(annual_path, index=False)

    # Save three month-by-year heatmaps.
    create_monthly_charts(daily)
    yearly_spell_chart_path = create_yearly_spell_chart(annual)

    print("\nHistorical analysis completed.")
    print(f"Station location: {location}")
    print(f"Daily classifications: {daily_path}")
    print(f"Spell events: {spells_path}")
    print(f"Annual summaries: {annual_path}")
    print(f"Heatmaps saved in: {OUTPUT_DIR}")
    print(f"Yearly spell occurrence chart: {yearly_spell_chart_path}")

    if spells.empty:
        print("No qualifying spells detected.")



if __name__ == "__main__":
    main()
