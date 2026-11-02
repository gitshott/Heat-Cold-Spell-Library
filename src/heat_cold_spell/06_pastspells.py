from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED = ("YEAR", "MN", "DT", "MAX", "MIN")
MINIMUM_SPELL_DAYS = 5

# Minimum actual MAX temperature for the heat-wave departure rule.
HEAT_MIN_TMAX_C = {
    "plains": 40.0,
    "coastal": 37.0,
    "hilly": 30.0,
}

def load_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Input must be CSV or Excel (.csv, .xlsx, .xls, .xlsm).")

    df.columns = [str(column).strip().upper() for column in df.columns]
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for column in REQUIRED:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["DATE"] = pd.to_datetime(
        {"year": df["YEAR"], "month": df["MN"], "day": df["DT"]},
        errors="coerce",
    )

    invalid_dates = int(df["DATE"].isna().sum())
    df = df.dropna(subset=["DATE"])
    if df.empty:
        raise ValueError("No valid dates could be created from YEAR, MN, and DT.")

    duplicate_count = int(df["DATE"].duplicated().sum())

    # Average duplicate records for the same date.
    df = df.groupby("DATE", as_index=False).agg({"MAX": "mean", "MIN": "mean"})

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
    """Calculate year-specific calendar-day normals using 15-year windows."""
    first_year = int(df["YEAR"].min())
    last_year = int(df["YEAR"].max())
    years = sorted(df["YEAR"].astype(int).unique())

    # The final seven years use the same trailing 15-year baseline.
    trailing_start = max(first_year, last_year - 14)
    trailing_years = set(range(trailing_start, last_year + 1))
    normal_rows = []

    for target_year in years:
        if target_year > last_year - 7:
            window_years = trailing_years
        else:
            # Center the window where possible. At the start of the record,
            # use the available years through target_year + 7.
            start_year = max(first_year, target_year - 7)
            end_year = min(last_year, target_year + 7)
            window_years = set(range(start_year, end_year + 1))

        window = df[df["YEAR"].isin(window_years)].copy()
        window["CALENDAR_DAY"] = window["DATE"].dt.strftime("%m-%d")

        normals = window.groupby("CALENDAR_DAY", as_index=False).agg(
            NORMAL_MAX=("MAX", "mean"),
            NORMAL_MIN=("MIN", "mean"),
            MAX_NORMAL_OBSERVATIONS=("MAX", "count"),
            MIN_NORMAL_OBSERVATIONS=("MIN", "count"),
        )
        normals["YEAR"] = target_year
        normal_rows.append(normals)

    if not normal_rows:
        raise ValueError("Could not calculate normals from the input data.")

    return pd.concat(normal_rows, ignore_index=True)


def classify_daily_data(
    df: pd.DataFrame,
    location: str,
) -> pd.DataFrame:
    location = location.strip().lower()
    if location not in HEAT_MIN_TMAX_C:
        valid_locations = ", ".join(HEAT_MIN_TMAX_C)
        raise ValueError(f"Location must be one of: {valid_locations}")

    minimum_tmax = HEAT_MIN_TMAX_C[location]

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

    # Heat-wave day:
    # - MAX must meet the location's minimum temperature and be at least
    #   4.5 C above NORMAL_MAX; OR
    # - MAX is at least 45 C.
    #
    # The separate spell rule below requires MAX >= 45 C on two consecutive
    # days to declare a spell through that trigger.
    heat_day = max_valid & (
        (
            (result["MAX"] >= minimum_tmax)
            & (result["MAX_DEPARTURE"] >= 4.5)
        )
        | (result["MAX"] >= 45.0)
    )

    # Cold-wave day:
    # actual MIN must be below 10 C and at least 3 C below NORMAL_MIN.
    cold_day = (
        min_valid
        & (result["MIN"] < 10.0)
        & (result["MIN_DEPARTURE"] <= -3.0)
    )

    result["HEAT_CLASSIFICATION"] = "Normal"
    result["COLD_CLASSIFICATION"] = "Normal"

    result.loc[~max_valid, "HEAT_CLASSIFICATION"] = "Missing"
    result.loc[~min_valid, "COLD_CLASSIFICATION"] = "Missing"

    result.loc[heat_day, "HEAT_CLASSIFICATION"] = "Heat Wave Day"
    result.loc[cold_day, "COLD_CLASSIFICATION"] = "Cold Wave Day"

    result["CLASSIFICATION"] = "Normal"
    result.loc[cold_day, "CLASSIFICATION"] = "Cold Wave Day"
    result.loc[heat_day, "CLASSIFICATION"] = "Heat Wave Day"
    result.loc[~max_valid & ~min_valid, "CLASSIFICATION"] = "Missing"

    return result.sort_values("DATE").reset_index(drop=True)


def qualifying_runs(group: pd.DataFrame, mask: pd.Series):
    """Yield consecutive qualifying date ranges within one year."""
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


def add_event(events, year, spell, start, end, trigger):
    events.append(
        {
            "Year": int(year),
            "Spell": spell,
            "From Date": start.strftime("%Y-%m-%d"),
            "To Date": end.strftime("%Y-%m-%d"),
            "Duration Days": (end - start).days + 1,
            "Trigger": trigger,
        }
    )


def detect_spells(df: pd.DataFrame) -> pd.DataFrame:
    events = []

    for year, group in df.groupby("YEAR", sort=True):
        group = group.sort_values("DATE").reset_index(drop=True)

        # Regular heat spells: at least five consecutive heat-wave days.
        heat_mask = group["HEAT_CLASSIFICATION"].eq("Heat Wave Day")
        for start, end in qualifying_runs(group, heat_mask):
            if (end - start).days + 1 >= MINIMUM_SPELL_DAYS:
                add_event(
                    events,
                    year,
                    "Heat Wave",
                    start,
                    end,
                    "5 or more consecutive heat-wave days",
                )

        # Additional heat trigger: two consecutive days with MAX >= 45 C.
        extreme_heat_mask = group["MAX"].notna() & (group["MAX"] >= 45.0)
        for start, end in qualifying_runs(group, extreme_heat_mask):
            if (end - start).days + 1 >= 2:
                add_event(
                    events,
                    year,
                    "Heat Wave",
                    start,
                    end,
                    "2 or more consecutive days with MAX >= 45 C",
                )

        # Cold spells: at least five consecutive cold-wave days.
        cold_mask = group["COLD_CLASSIFICATION"].eq("Cold Wave Day")
        for start, end in qualifying_runs(group, cold_mask):
            if (end - start).days + 1 >= MINIMUM_SPELL_DAYS:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect heat and cold spells from daily MAX/MIN data."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        help="Input CSV or Excel file with YEAR, MN, DT, MAX, and MIN columns",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output folder; defaults to the folder containing the input",
    )
    parser.add_argument(
        "--location",
        choices=sorted(HEAT_MIN_TMAX_C),
        help="Station location: plains, coastal, or hilly",
    )
    args = parser.parse_args()

    entered_path = args.file_path
    if not entered_path:
        entered_path = input("Enter input CSV/Excel file path: ").strip().strip('"').strip("'")

    location = args.location
    if not location:
        location = input(
            "Station location (plains, coastal, or hilly): "
        ).strip().lower()

    if location not in HEAT_MIN_TMAX_C:
        valid_locations = ", ".join(HEAT_MIN_TMAX_C)
        raise ValueError(f"Location must be one of: {valid_locations}")

    source = Path(entered_path).expanduser()
    daily = classify_daily_data(load_data(source), location)
    spells = detect_spells(daily)

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else source.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_path = output_dir / f"{source.stem}_daily_classification.csv"
    spells_path = output_dir / f"{source.stem}_spell_predictions.csv"

    daily.to_csv(daily_path, index=False, date_format="%Y-%m-%d")
    spells.to_csv(spells_path, index=False)

    print("Spell detection complete.")
    print(f"Station location: {location}")
    print(f"Daily classifications: {daily_path}")
    print(f"Spell predictions: {spells_path}")

    if spells.empty:
        print("No qualifying spells detected.")
    else:
        print(spells.to_string(index=False))


if __name__ == "__main__":
    main()