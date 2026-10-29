"""
Missing Data Analysis Toolkit
Core analysis and imputation helpers built with NumPy and pandas.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and return a DataFrame."""
    return pd.read_csv(path)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing count and percentage for every column."""
    total = len(df)
    report = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [
            round(float(df[c].isna().mean() * 100), 2) if total else 0.0
            for c in df.columns
        ],
        "unique_non_missing": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    return report.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)


def row_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values per row without exposing row contents."""
    counts = df.isna().sum(axis=1)
    return pd.DataFrame({
        "row_index": df.index,
        "missing_count": counts.to_numpy(),
        "missing_percent": (counts / max(df.shape[1], 1) * 100).round(2).to_numpy(),
        "has_missing": counts.gt(0).to_numpy(),
    })


def missingness_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Count repeated patterns such as 'age missing, income present'."""
    if df.empty or df.shape[1] == 0:
        return pd.DataFrame(columns=["pattern", "rows", "percent"])
    masks = df.isna()
    labels = masks.apply(
        lambda row: ", ".join(masks.columns[row.to_numpy()]) if row.any() else "No missing values",
        axis=1,
    )
    out = labels.value_counts(dropna=False).rename_axis("pattern").reset_index(name="rows")
    out["percent"] = (out["rows"] / len(df) * 100).round(2)
    return out


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for numeric columns, including missing counts."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T
    summary["missing_count"] = numeric.isna().sum()
    summary["missing_percent"] = (numeric.isna().mean() * 100).round(2)
    return summary.reset_index(names="column")


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Most common non-missing category and category count."""
    rows = []
    for col in df.select_dtypes(exclude="number").columns:
        non_null = df[col].dropna()
        mode = non_null.mode()
        rows.append({
            "column": col,
            "non_missing_count": int(non_null.size),
            "unique_categories": int(non_null.nunique()),
            "most_frequent_value": mode.iloc[0] if not mode.empty else None,
            "most_frequent_count": int(non_null.value_counts().iloc[0]) if not non_null.empty else 0,
            "missing_count": int(df[col].isna().sum()),
        })
    return pd.DataFrame(rows)


def _mode_or_nan(series: pd.Series) -> Any:
    modes = series.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else np.nan


def impute_missing(
    df: pd.DataFrame,
    strategy: str = "auto",
    constant: Any = "Unknown",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return a copy with missing values imputed.
    Strategies: auto, mean, median, mode, constant, ffill, bfill.
    'auto' uses median for numeric columns and mode for non-numeric columns.
    """
    valid = {"auto", "mean", "median", "mode", "constant", "ffill", "bfill"}
    if strategy not in valid:
        raise ValueError(f"strategy must be one of {sorted(valid)}")
    result = df.copy()
    selected = columns if columns is not None else list(result.columns)
    unknown = sorted(set(selected) - set(result.columns))
    if unknown:
        raise KeyError(f"Unknown columns: {unknown}")

    for col in selected:
        s = result[col]
        if not s.isna().any():
            continue
        chosen = strategy
        if chosen == "auto":
            chosen = "median" if pd.api.types.is_numeric_dtype(s) else "mode"
        if chosen == "mean":
            if not pd.api.types.is_numeric_dtype(s):
                raise TypeError(f"mean strategy requires numeric column: {col}")
            fill_value = s.mean()
            result[col] = s.fillna(fill_value)
        elif chosen == "median":
            if not pd.api.types.is_numeric_dtype(s):
                raise TypeError(f"median strategy requires numeric column: {col}")
            result[col] = s.fillna(s.median())
        elif chosen == "mode":
            result[col] = s.fillna(_mode_or_nan(s))
        elif chosen == "constant":
            result[col] = s.fillna(constant)
        elif chosen == "ffill":
            result[col] = s.ffill()
        elif chosen == "bfill":
            result[col] = s.bfill()
    return result


def compare_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """Compare remaining missing values after common imputation strategies."""
    strategies = ["drop_rows", "auto", "mean", "median", "mode", "constant", "ffill", "bfill"]
    rows = []
    for strategy in strategies:
        try:
            if strategy == "drop_rows":
                candidate = df.dropna()
            else:
                # Mean/median apply only to numeric columns; mode only to categorical columns.
                candidate = df.copy()
                numeric = candidate.select_dtypes(include="number").columns.tolist()
                categorical = candidate.select_dtypes(exclude="number").columns.tolist()
                if strategy in {"mean", "median"}:
                    candidate = impute_missing(candidate, strategy, columns=numeric)
                    candidate = impute_missing(candidate, "mode", columns=categorical)
                elif strategy == "mode":
                    candidate = impute_missing(candidate, "mode")
                else:
                    candidate = impute_missing(candidate, strategy, constant="Unknown")
            rows.append({
                "strategy": strategy,
                "rows_after": len(candidate),
                "cells_missing_after": int(candidate.isna().sum().sum()),
                "missing_cells_filled": int(df.isna().sum().sum() - candidate.isna().sum().sum()),
                "rows_removed": int(len(df) - len(candidate)),
            })
        except (TypeError, ValueError) as exc:
            rows.append({"strategy": strategy, "rows_after": len(df), "cells_missing_after": int(df.isna().sum().sum()),
                         "missing_cells_filled": 0, "rows_removed": 0, "error": str(exc)})
    return pd.DataFrame(rows)


def generate_report(df: pd.DataFrame, output_dir: str | Path) -> dict[str, str]:
    """Write key analysis tables and a JSON summary to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    tables = {
        "missingness_report.csv": missingness_report(df),
        "row_missingness.csv": row_missingness(df),
        "missingness_patterns.csv": missingness_patterns(df),
        "numeric_summary.csv": numeric_summary(df),
        "categorical_summary.csv": categorical_summary(df),
        "strategy_comparison.csv": compare_strategies(df),
    }
    for filename, table in tables.items():
        path = out / filename
        table.to_csv(path, index=False)
        paths[filename] = str(path)
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_cells": int(df.size),
        "missing_cells": int(df.isna().sum().sum()),
        "overall_missing_percent": round(float(df.isna().sum().sum() / max(df.size, 1) * 100), 2),
        "columns_with_missing_values": int(df.isna().any().sum()),
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["summary.json"] = str(summary_path)
    return paths


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze and handle missing data in a CSV file.")
    parser.add_argument("--input", default="data/customer_data.csv", help="Input CSV path")
    parser.add_argument("--output", default="reports", help="Output directory")
    parser.add_argument("--strategy", default="auto",
                        choices=["auto", "mean", "median", "mode", "constant", "ffill", "bfill"],
                        help="Imputation strategy for the cleaned CSV")
    parser.add_argument("--constant", default="Unknown", help="Constant value for constant strategy")
    args = parser.parse_args()

    data = load_dataset(args.input)
    paths = generate_report(data, args.output)
    cleaned = impute_missing(data, strategy=args.strategy, constant=args.constant)
    cleaned_path = Path(args.output) / f"cleaned_{args.strategy}.csv"
    cleaned.to_csv(cleaned_path, index=False)
    print(f"Loaded dataset: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"Missing cells: {int(data.isna().sum().sum())} ({data.isna().sum().sum()/max(data.size,1)*100:.2f}%)")
    print("\nMissingness by column:")
    print(missingness_report(data).to_string(index=False))
    print(f"\nReports saved to: {Path(args.output).resolve()}")
    print(f"Cleaned dataset saved to: {cleaned_path.resolve()}")
    print(f"Strategy: {args.strategy}")
    print(f"Report files: {', '.join(paths.keys())}")


if __name__ == "__main__":
    main()
