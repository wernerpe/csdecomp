# Configuration file for Sphinx documentation builder

project = 'CSDecomp'
copyright = '2025'
author = 'Peter Werner, Richard Cheng, Tom Stewart'

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'breathe',  # Add breathe
    'exhale'    # Add exhale
]

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns to exclude
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The theme to use for HTML documentation
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
}

# Output options
autodoc_member_order = 'bysource'
# add_module_names = False

# Breathe configuration
breathe_projects = {
    "CSDecomp": "../doxyoutput/xml"  # Path is relative to source directory
}
breathe_default_project = "CSDecomp"

# Setup exhale
exhale_args = {
    "containmentFolder":     "./api",
    "rootFileName":          "library_root.rst",
    "rootFileTitle":         "Library API",
    "doxygenStripFromPath":  "..",
    "createTreeView":        True,
}