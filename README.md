# Missing Data Analysis Toolkit

An end-to-end beginner-friendly data-analysis project built with **Python, NumPy, and pandas**. It profiles missing data, summarizes numeric/categorical columns, discovers recurring missingness patterns, compares cleaning strategies, and exports cleaned data plus reports.

## Features

- Load a CSV dataset with pandas
- Missing-value counts and percentages per column
- Row-level missingness summary
- Repeated missingness-pattern counts
- Numeric and categorical summaries
- Compare row deletion, mean, median, mode, automatic imputation, constant fill, forward fill, and backward fill
- Export reports as CSV and a JSON dataset summary
- Save a cleaned dataset using a selected strategy
- Reproducible sample customer dataset with intentionally missing values

## Project structure

```text
missing_data_analysis_toolkit/
├── data/
│   └── customer_data.csv
├── reports/                  # generated reports are saved here
├── src/
│   └── missing_data_toolkit.py
├── requirements.txt
└── README.md
```

## 1. Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Run the complete workflow

Run from the project root:

```bash
python src/missing_data_toolkit.py
```

Choose a strategy:

```bash
python src/missing_data_toolkit.py --strategy median
python src/missing_data_toolkit.py --strategy mode
python src/missing_data_toolkit.py --strategy constant --constant "Not provided"
```

Use your own dataset:

```bash
python src/missing_data_toolkit.py --input data/your_file.csv --output reports --strategy auto
```

## 3. Output files

The `reports/` folder receives:

- `missingness_report.csv` — missing count and percentage per column
- `row_missingness.csv` — missing count/percentage per row
- `missingness_patterns.csv` — common combinations of missing columns
- `numeric_summary.csv` — descriptive statistics for numeric columns
- `categorical_summary.csv` — category counts and most frequent category
- `strategy_comparison.csv` — comparison of cleaning strategies
- `summary.json` — high-level dataset summary
- `cleaned_auto.csv` (or the selected strategy) — cleaned dataset

## 4. How the strategies work

| Strategy | Best suited for | Main caution |
|---|---|---|
| Mean | Numeric data with limited skew/outliers | Sensitive to outliers |
| Median | Numeric data with skew/outliers | Can reduce natural variation |
| Mode | Categorical data; sometimes discrete numeric data | Dominant category can be overrepresented |
| Constant | Explicit “Unknown”/“Not provided” category | Choose a meaningful sentinel |
| Forward/backward fill | Ordered/time-series data | Sort rows correctly first; avoid filling across unrelated groups |
| Drop rows | Very small amount of missing data | Can discard many useful observations |
| Auto | Quick baseline: median for numeric, mode for non-numeric | A starting point, not a universally correct choice |

**Important:** missingness may be informative. Before imputing, investigate why values are absent. Mean/median/mode imputation is a baseline and may distort statistical relationships. Do not use forward/backward fill unless row order has real meaning.

## 5. Suggested project workflow

1. Understand the columns and data types.
2. Quantify missing values globally, by column, and by row.
3. Inspect missingness patterns and compare them with domain knowledge.
4. Select a strategy based on each column's meaning and data type.
5. Compare how many cells are filled and rows removed.
6. Export the cleaned dataset and all reports.
7. Validate that the output has the expected schema and missing-value count.

## 6. Extend it for a portfolio

- Add unit tests with `pytest`
- Add a visual missingness heatmap with `matplotlib` (optional; not required by the core toolkit)
- Add group-wise imputation, e.g. median income by city
- Add a train/test-safe imputation workflow for machine-learning projects
- Add a command to generate a Markdown or HTML report
- Add logging and a configuration file

## Limitations

This toolkit reports patterns but does **not** prove whether data is MCAR, MAR, or MNAR. Those are assumptions that require context and statistical investigation. The sample dataset is synthetic and intended for practice.
