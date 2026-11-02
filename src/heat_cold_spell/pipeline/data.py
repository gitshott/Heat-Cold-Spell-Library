import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any

def load_and_validate_daily_data(file_path: Path, required_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(file_path)
    
    # Check for missing required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw data: {', '.join(missing_cols)}")
    
    # Cast numeric columns
    for col in ["YEAR", "MN", "DT", "MAX", "MIN"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Construct standard DATE column from YEAR, MN (Month), and DT (Day)
    df["DATE"] = pd.to_datetime(
        {
            "year": df["YEAR"],
            "month": df["MN"],
            "day": df["DT"]
        },
        errors="coerce"
    )
    
    # Drop records with invalid dates or key measurements
    df = df.dropna(subset=["DATE", "YEAR", "MAX", "MIN"]).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    df = df.sort_values("DATE").reset_index(drop=True)
    
    # Generate validation report metrics
    report = {
        "row_count": len(df),
        "start_date": df["DATE"].min().strftime("%Y-%m-%d"),
        "end_date": df["DATE"].max().strftime("%Y-%m-%d"),
        "missing_max_count": int(df["MAX"].isna().sum()),
        "missing_min_count": int(df["MIN"].isna().sum()),
        "columns": list(df.columns)
    }
    
    return df, report