#!/bin/sh
# Install Loom from the latest GitHub release — no Python/uv required.
#
#   curl -LsSf https://raw.githubusercontent.com/mikexkllr/loom-cli/main/scripts/install.sh | sh
#
# Re-running this script re-downloads and reinstalls the latest build (the
# same thing `loom update` does from inside the app, minus the checksum diff
# — this always overwrites). Override the install directory with
# LOOM_INSTALL_DIR; it defaults to ~/.local/bin.
#
# Windows: use install.ps1 instead (curl|sh doesn't apply there).
set -eu

REPO="mikexkllr/loom-cli"
RELEASE_BASE="https://github.com/${REPO}/releases/latest/download"
INSTALL_DIR="${LOOM_INSTALL_DIR:-$HOME/.local/bin}"

err() { echo "error: $*" >&2; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || err "'$1' is required but not found"
}

fetch() {
    # fetch <url> <output-path>
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$1" -O "$2"
    else
        err "need curl or wget to download Loom"
    fi
}

detect_asset() {
    os="$(uname -s)"
    machine="$(uname -m)"

    case "$os" in
        Darwin) plat="macos" ;;
        Linux) plat="linux" ;;
        MINGW*|MSYS*|CYGWIN*)
            err "on Windows, run install.ps1 instead: irm https://raw.githubusercontent.com/${REPO}/main/scripts/install.ps1 | iex"
            ;;
        *) err "unsupported OS: $os" ;;
    esac

    case "$machine" in
        x86_64|amd64) arch="x64" ;;
        arm64|aarch64) arch="arm64" ;;
        *) err "unsupported architecture: $machine" ;;
    esac

    echo "loom-${plat}-${arch}"
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        err "need sha256sum or shasum to verify the download"
    fi
}

main() {
    need_cmd uname
    need_cmd mkdir
    need_cmd chmod

    asset="$(detect_asset)"
    workdir="$(mktemp -d)"
    trap 'rm -rf "$workdir"' EXIT

    echo "downloading ${asset} …"
    fetch "${RELEASE_BASE}/${asset}" "${workdir}/loom"
    fetch "${RELEASE_BASE}/checksums.txt" "${workdir}/checksums.txt"

    expected="$(awk -v a="$asset" '$2 == a { print $1 }' "${workdir}/checksums.txt")"
    [ -n "$expected" ] || err "no checksum for ${asset} in checksums.txt — release may be incomplete"

    actual="$(sha256_of "${workdir}/loom")"
    [ "$actual" = "$expected" ] || err "checksum mismatch (expected ${expected}, got ${actual}) — download corrupted or tampered, aborting"

    mkdir -p "$INSTALL_DIR"
    chmod +x "${workdir}/loom"
    mv "${workdir}/loom" "${INSTALL_DIR}/loom"

    echo "✓ installed loom to ${INSTALL_DIR}/loom"

    case ":$PATH:" in
        *":${INSTALL_DIR}:"*) ;;
        *)
            echo ""
            echo "${INSTALL_DIR} isn't on your PATH yet. Add this to your shell profile"
            echo "(~/.bashrc, ~/.zshrc, …) and open a new shell:"
            echo ""
            echo "    export PATH=\"${INSTALL_DIR}:\$PATH\""
            ;;
    esac

    echo ""
    echo "Run 'loom' to get started. Re-run this script anytime to reinstall the"
    echo "latest build, or use 'loom update' once it's installed."
}

main "$@"
