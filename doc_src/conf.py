# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

import datetime
from pathlib import Path

project = "gemseo-benchmark"
copyright = f"{datetime.datetime.now().year}, IRT Saint Exupéry"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

sphinx_gallery_conf = {
    "examples_dirs": "_examples",  # path to your example scripts
    "gallery_dirs": "examples",  # path to where to save gallery generated output
    "capture_repr": ("_repr_html_", "__repr__"),
    "default_thumb_file": str(Path(__file__).parent / "_static/icon.png"),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]

intersphinx_mapping = {
    "gemseo": ("https://gemseo.readthedocs.io/en/stable/", None),
}
tls_verify = False
