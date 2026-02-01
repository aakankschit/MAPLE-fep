# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import pathlib
import sys

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# Add the src directory to the Python path
docs_dir = pathlib.Path(__file__).parent.resolve()
project_root = docs_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MAPLE-fep"
copyright = "2025, MAPLE Contributors"
author = "Aakankschit Nandkeolyar"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

# Graphviz settings
graphviz_output_format = "svg"
graphviz_dot_args = ["-Gdpi=150"]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
    "member-order": "bysource",
}

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Type hints settings
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True

# MyST parser settings
myst_enable_extensions = [
    "deflist",
    "tasklist",
    "colon_fence",
    "html_admonition",
    "html_image",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# HTML context
html_context = {
    "display_github": True,
    "github_user": "aakankschit",
    "github_repo": "MAPLE-fep",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Custom sidebar
html_sidebars = {
    "**": [
        "versions.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}

# -- Options for intersphinx extension --------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pyro": ("https://docs.pyro.ai/", None),
}

# -- Options for LaTeX output -----------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "fncychap": "",
    "printindex": "",
}

latex_documents = [
    ("index", "MAPLE.tex", "MAPLE Documentation", "MAPLE Contributors", "manual"),
]

# -- Options for manual page output -----------------------------------------
man_pages = [("index", "maple", "MAPLE Documentation", [author], 1)]

# -- Options for Texinfo output ---------------------------------------------
texinfo_documents = [
    (
        "index",
        "MAPLE",
        "MAPLE Documentation",
        author,
        "MAPLE",
        "Maximum A Posteriori Learning of Energies for FEP analysis.",
        "Miscellaneous",
    ),
]

# -- Options for EPUB output ------------------------------------------------
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
