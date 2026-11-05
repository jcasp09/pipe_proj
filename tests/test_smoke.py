import importlib

import pytest

CORE_MODULES = [
    "io", "schema", "split", "profile", "correlation", "recipe", "transformers", "compiler",
]


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_module_imports(name):
    importlib.import_module(f"preppipe.{name}")


def test_core_does_not_import_ui_or_api():
    """Core must stay usable without Streamlit/FastAPI (CLAUDE.md rule)."""
    import subprocess
    import sys

    code = (
        "import sys, preppipe\n"
        + "".join(f"import preppipe.{m}\n" for m in CORE_MODULES)
        + "bad = [m for m in ('streamlit', 'fastapi') if m in sys.modules]\n"
        + "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
