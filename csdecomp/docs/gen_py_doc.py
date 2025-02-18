from sphinx.cmd.build import main as sphinx_build
from pathlib import Path

# This is still experimental and does work 100%

def build_sphinx_docs():
    docs_dir = Path("")
    
    # Build the documentation
    sphinx_build_args = [
        '-b', 'html',  # Build HTML
        str(docs_dir / "source"),  # Source directory
        str(docs_dir / "build" / "html")  # Output directory
    ]
    return sphinx_build(sphinx_build_args)

if __name__ == "__main__":
    build_sphinx_docs()