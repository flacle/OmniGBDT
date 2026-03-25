from __future__ import annotations

import os
from pathlib import Path
import tomllib


DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent

with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
    _PYPROJECT = tomllib.load(pyproject_file)


project = "OmniGBDT"
copyright = "2020-2026, original GBDT-MO authors and fork maintainers"
author = "Original authors: Zhendong Zhang and Cheolkon Jung"
release = _PYPROJECT["project"]["version"]
version = release

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "includehidden": False,
}
html_static_path = ["_static"]
html_css_files = ["extra.css"]
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
