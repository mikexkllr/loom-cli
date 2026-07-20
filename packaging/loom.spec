# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone `loom` binary.

Build locally:
    uv sync --group build
    uv run pyinstaller packaging/loom.spec --noconfirm --clean
    ./dist/loom --help

CI builds this per-OS/arch in .github/workflows/release.yml.

The langchain/langgraph/deepagents ecosystem leans on dynamic imports and
importlib.metadata version checks that PyInstaller's static analysis can't
see, so packages in COLLECT_PACKAGES get their data/binaries/submodules
explicitly collected, and packages in METADATA_ONLY get their dist-info
copied in (enough for `importlib.metadata.version(...)` checks to pass).
If a frozen binary raises ModuleNotFoundError or PackageNotFoundError for
something not listed here, add it to the matching list and rebuild — these
lists are the whole point of the spec.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

# SPECPATH is injected into the spec's exec namespace by PyInstaller itself
# (this file has no __file__ — it's exec'd, not imported).
REPO_ROOT = Path(SPECPATH).resolve().parent

# Import (module) names, not PyPI distribution names — collect_all resolves
# by import path. langgraph-checkpoint / langgraph-checkpoint-sqlite install
# into the shared langgraph/ tree, so collecting "langgraph" covers them too.
COLLECT_PACKAGES = [
    "langchain",
    "langchain_core",
    "langchain_ollama",
    "langchain_anthropic",
    "langchain_openai",
    "langchain_google_genai",
    "langchain_mcp_adapters",
    "langgraph",
    "langsmith",
    "deepagents",
    "tiktoken",
    "mcp",
    "anthropic",
    "openai",
    "google.genai",
]

# Distribution names — these only need importlib.metadata to see a version,
# no dynamic submodule/data loading.
METADATA_ONLY = [
    "pydantic",
    "pydantic-core",
    "httpx",
    "httpcore",
    "typer",
    "rich",
    "prompt_toolkit",
    "pyyaml",
]

datas = [
    (str(REPO_ROOT / "loom" / "config"), "loom/config"),
    (str(REPO_ROOT / "loom" / "skills"), "loom/skills"),
]
binaries = []
hiddenimports = []

def _is_test_noise(name: str) -> bool:
    """Some packages (google.genai) bundle their own test suite as regular
    submodules/data — collect_all pulls it in too. It's dead weight in a
    frozen binary, so drop anything under a `.../tests/...` path."""
    return ".tests." in f".{name}." or "/tests/" in name.replace("\\", "/")


for pkg in COLLECT_PACKAGES:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    except Exception:
        continue
    datas += [d for d in pkg_datas if not _is_test_noise(d[1])]
    binaries += pkg_binaries
    hiddenimports += [h for h in pkg_hidden if not _is_test_noise(h)]

for dist in METADATA_ONLY:
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

a = Analysis(
    [str(REPO_ROOT / "packaging" / "entry_point.py")],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="loom",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
