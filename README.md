# PrepPipe

Data cleaning, EDA and correlation analysis for tabular datasets, exported as a
leakage-safe scikit-learn preprocessing pipeline (unfitted for CV, fitted for inference/API).

See `CLAUDE.md` for design rules and roadmap.

## Setup

```bash
uv sync --all-extras          # core + Streamlit app + FastAPI + dev tools
uv run pytest                 # tests
uv run streamlit run app/streamlit_app.py
uv run uvicorn api.main:app --reload
```
