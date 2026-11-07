"""Shared fixtures. pytest loads this file automatically; any test can use these
fixtures just by naming them as a function argument."""

from pathlib import Path

import pytest


@pytest.fixture
def write_csv(tmp_path):
    """Return a function that writes text to a CSV file in a temp dir and returns its path.

    `tmp_path` is a built-in pytest fixture: a fresh empty directory per test,
    so tests never touch real data or each other's files.
    """

    def _write(text: str, name: str = "data.csv") -> Path:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return _write
