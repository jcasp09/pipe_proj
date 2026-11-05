"""FastAPI service. Run with: uv run uvicorn api.main:app --reload"""

from fastapi import FastAPI

import preppipe

app = FastAPI(title="PrepPipe API", version=preppipe.__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
