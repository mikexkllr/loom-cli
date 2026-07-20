"""PyInstaller entry point.

Frozen builds have no console-script shim (the one `[project.scripts]`
generates), so this just imports and runs the Typer app directly.
"""

from loom.cli.main import app

if __name__ == "__main__":
    app()
