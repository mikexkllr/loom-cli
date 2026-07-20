# Install Loom from the latest GitHub release — no Python/uv required.
#
#   irm https://raw.githubusercontent.com/mikexkllr/loom-cli/main/scripts/install.ps1 | iex
#
# Re-running this script re-downloads and reinstalls the latest build (the
# same thing `loom update` does from inside the app, minus the checksum diff
# — this always overwrites). Override the install directory by setting
# $env:LOOM_INSTALL_DIR before running.

$ErrorActionPreference = "Stop"

$Repo = "mikexkllr/loom-cli"
$ReleaseBase = "https://github.com/$Repo/releases/latest/download"
$Asset = "loom-windows-x64.exe"
$InstallDir = if ($env:LOOM_INSTALL_DIR) { $env:LOOM_INSTALL_DIR } else { "$env:LOCALAPPDATA\loom\bin" }

function Get-Sha256($Path) {
    (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLower()
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$WorkDir = New-Item -ItemType Directory -Force -Path (Join-Path $env:TEMP "loom-install-$([guid]::NewGuid())")

try {
    Write-Host "downloading $Asset …"
    $BinaryPath = Join-Path $WorkDir "loom.exe"
    $ChecksumsPath = Join-Path $WorkDir "checksums.txt"
    Invoke-WebRequest -Uri "$ReleaseBase/$Asset" -OutFile $BinaryPath -UseBasicParsing
    Invoke-WebRequest -Uri "$ReleaseBase/checksums.txt" -OutFile $ChecksumsPath -UseBasicParsing

    $ChecksumLine = Select-String -Path $ChecksumsPath -Pattern ([regex]::Escape($Asset)) | Select-Object -First 1
    if (-not $ChecksumLine) {
        throw "no checksum for $Asset in checksums.txt — release may be incomplete"
    }
    $Expected = ($ChecksumLine.Line -split '\s+')[0].ToLower()
    $Actual = Get-Sha256 $BinaryPath
    if ($Actual -ne $Expected) {
        throw "checksum mismatch (expected $Expected, got $Actual) — download corrupted or tampered, aborting"
    }

    $Target = Join-Path $InstallDir "loom.exe"
    Move-Item -Force -Path $BinaryPath -Destination $Target
    Write-Host "installed loom to $Target"
}
finally {
    Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
}

$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not (";$UserPath;".ToLower().Contains(";$InstallDir;".ToLower()))) {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    Write-Host ""
    Write-Host "Added $InstallDir to your user PATH. Open a new terminal for it to take effect."
} else {
    Write-Host ""
    Write-Host "$InstallDir is already on your PATH."
}

Write-Host ""
Write-Host "Run 'loom' to get started. Re-run this script anytime to reinstall the"
Write-Host "latest build, or use 'loom update' once it's installed."
