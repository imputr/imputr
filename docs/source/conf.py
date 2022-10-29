# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../..')) 

# -- Project information

project = 'Imputr'
copyright = '2022'
author = 'Rauf Akdemir'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'autoapi.extension',
    # 'sphinxcontrib.fulltoc'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# # -- Options for HTML output

html_theme = 'furo'

html_logo = "../_static/imputr-logo-horizontal.svg"

autoapi_dirs = ['../../imputr']

html_theme_options = {
    # 'navigation_depth': 4
}

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Configs for napoleon
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_attr_annotations = True