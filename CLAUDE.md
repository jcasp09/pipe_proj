# PrepPipe — project context for Claude

## What this is

A course project (Big Data Analytics, master's in ML): a web app that helps users perform
**data cleaning, EDA and correlation analysis** on their own tabular datasets, then gives them
a **leakage-safe preprocessing pipeline** — as downloads and through an API — ready for modeling.

Practical problem it addresses: data preparation is slow and error-prone, and mistakes
(data leakage, inconsistent preprocessing between training and production) silently hurt models.

The tool is general (any single-table tabular dataset), demonstrated on several domains.

## About the developer

I'm learning as I build. My strongest background is ML (pandas, scikit-learn); I have less
experience structuring larger apps. When you write code:

- Work in small steps and explain the design choices briefly.
- Keep core logic readable; prefer clear code over clever code.
- Add or update tests with each feature.

## Core design rules (do not break these)

1. **The user uploads the full labeled dataset; the app does the train/test split.**
   Optionally the user may upload a predefined train and test file instead.
2. **The test set is never used for EDA or fitting.** After the split it is only ever transformed.
3. **Stateful steps are fit on the training split only** (imputation values, scaling, percentile
   caps, encodings). The API only _transforms_; it never refits.
4. **The recipe stores strategies, never learned values.**
   Yes: `"impute": "median"`, `"clip": {"upper_percentile": 99}`.
   No: `"impute_value": 5400`, `"clip_upper": 4980`.
5. **Two pipelines are exported from one recipe:**
   - _unfitted_ — for the user's cross-validation (refits inside each fold, no leakage)
   - _fitted_ on the full training split — for test evaluation, inference and the API
6. **Row-dropping (dedup, removing rows) is training-data preparation only.**
   `transform` must return exactly one output row per input row; outliers are capped, not dropped.
7. **The model is not part of the pipeline.** We output everything right before the model.
8. **Columns are referenced by name, never by index.** The target is chosen by name and is optional.
9. Stateless fixes (sentinel codes like -999 / 96 / 98 / "?" → missing, impossible values →
   missing, dropping ID columns) are fixed rules and may be decided from EDA.
10. Classification targets: stratified split by default. Regression: no stratify.
    If a date column is marked, split by time instead of randomly.

## User flow (recommended steps)

Stage 1 — load & set up (full dataset, nothing is fit):

1. Upload CSV or Parquet
2. Confirm header row (headers are not guaranteed in CSVs; detect and let the user confirm)
3. Confirm inferred column types (numeric / categorical / datetime / ID / text)
4. Choose target (or none) → detect task type, show class distribution / imbalance warning
5. Confirm excluded columns (ID-like, constant, near-empty)

Stage 2 — split: 6. Split settings (test size default 20%, seed, stratify, optional time-based split) 7. Split once; lock away the test set

Stage 3 — analyze (training split only): 8. Profile: stats, percentiles (1/5/25/50/75/95/99), skewness, missingness, outliers,
sentinel values → ranked issue list 9. Correlation: Pearson/Spearman (numeric), Cramér's V (categorical), correlation ratio
(mixed); target correlation; multicollinearity and leakage warnings

Stage 4 — clean: 10. Stateless fixes (confirm sentinel → missing, etc.) 11. Per-column strategies with smart defaults and before/after previews 12. Compile recipe → unfitted pipeline; fit a copy on the training split 13. Transform the test set 14. (Optional) impact check: cross-validated baseline model, raw vs. cleaned

Stage 5 — export:

- `recipe.json` (includes scikit-learn version, split settings, recipe version ID)
- unfitted pipeline `.joblib` + stratified CV code snippet
- fitted pipeline `.joblib`
- `train_clean` / `test_clean` (Parquet or CSV)
- API: `POST /pipelines/{id}/transform`, `GET /pipelines/{id}/recipe`, `GET /pipelines/{id}/schema`
- (Optional) "refit on all data" for deployment only

## Tech stack

- Python 3.11+, pandas (core), scikit-learn (`Pipeline`, `ColumnTransformer`,
  `set_output(transform="pandas")` so outputs keep feature names)
- Polars or DuckDB for fast loading/profiling of large files (add once the core works)
- Streamlit for the UI, FastAPI for the API, Plotly for charts
- pytest for tests

## Planned repo layout

```
preppipe/
  CLAUDE.md
  pyproject.toml
  src/preppipe/
    io.py            # loading, delimiter/header detection
    schema.py        # column type inference
    split.py         # train/test split logic
    profile.py       # stats, outliers, sentinel detection, issue list
    correlation.py   # mixed-type correlation, leakage/multicollinearity checks
    recipe.py        # recipe data model + JSON (de)serialization
    transformers.py  # custom sklearn transformers (e.g., SentinelReplacer, PercentileClipper)
    compiler.py      # recipe -> unfitted sklearn Pipeline
  app/streamlit_app.py
  api/main.py
  tests/
  notebooks/         # scratch exploration only; imports from src/preppipe
  data/              # gitignored
```

Core logic lives in `src/preppipe` and must not import Streamlit or FastAPI,
so the UI, API, and tests all share it.

## Demo / test datasets

| Dataset                                  | Domain         | Notes                                                                       |
| ---------------------------------------- | -------------- | --------------------------------------------------------------------------- |
| Give Me Some Credit (Kaggle competition) | Finance        | ~150k rows; missing income; 96/98 sentinel codes; age 0; ~7% positive class |
| UCI Air Quality                          | Environment    | missing coded as -200; semicolon delimiter; date/time columns               |
| UCI Diabetes 130-US Hospitals            | Healthcare     | "?" as missing; many categoricals; ID columns                               |
| NYC TLC Taxi Trips                       | Transportation | millions of rows, Parquet; negative fares, zero distances                   |
| Synthetic                                | Generated      | known injected issues, for measuring detection accuracy                     |

Never commit data files.

## Build phases

1. **MVP:** upload → target → split → profile (numeric + categorical) → per-column
   imputation/outlier/scaling/encoding choices → recipe + unfitted/fitted pipeline + clean data downloads
2. Correlation tab, header/type confirmation, sentinel detection, datetime handling
3. FastAPI transform endpoint with input schema validation; large-data path (Polars/DuckDB,
   sampling for interactive EDA); performance benchmarks
4. Impact check (raw vs. cleaned CV score), refit-on-all-data, polish

## Out of scope

Images, raw free text, multi-table joins, model training/tuning (beyond the optional impact
check), resampling for imbalance (mention only: apply to training folds during modeling).
