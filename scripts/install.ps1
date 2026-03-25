# Trevec installer for Windows — https://trevec.dev
# Usage: irm dl.trevec.dev/install.ps1 | iex
# Pinned: $env:TREVEC_VERSION="v1.1.0"; irm dl.trevec.dev/install.ps1 | iex

$ErrorActionPreference = "Stop"

$BaseUrl = if ($env:TREVEC_BASE_URL) { $env:TREVEC_BASE_URL } else { "https://dl.trevec.dev" }
$InstallDir = if ($env:TREVEC_INSTALL) { $env:TREVEC_INSTALL } else { "$env:USERPROFILE\.trevec" }
$BinDir = "$InstallDir\bin"
$Target = "x86_64-pc-windows-msvc"

function Write-Info($msg) { Write-Host $msg -ForegroundColor White }
function Write-Success($msg) { Write-Host $msg -ForegroundColor Green }
function Write-Err($msg) { Write-Host "error: $msg" -ForegroundColor Red; exit 1 }

# Resolve version
function Get-TrevecVersion {
    if ($env:TREVEC_VERSION) {
        return $env:TREVEC_VERSION
    }
    try {
        $version = (Invoke-WebRequest -Uri "$BaseUrl/version.txt" -UseBasicParsing).Content.Trim()
        return $version
    } catch {
        Write-Err "Failed to fetch latest version from $BaseUrl/version.txt"
    }
}

# Main install
Write-Info "Installing trevec..."
Write-Host ""

$Version = Get-TrevecVersion
Write-Info "  Version:  $Version"
Write-Info "  Platform: windows/x64"
Write-Host ""

$Archive = "trevec-$Target.zip"
$Url = "$BaseUrl/releases/$Version/$Archive"
$ChecksumUrl = "$Url.sha256"

# Create temp directory
$TmpDir = Join-Path ([System.IO.Path]::GetTempPath()) "trevec-install-$(Get-Random)"
New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null

try {
    # Download archive and checksum
    Write-Info "Downloading $Archive..."
    Invoke-WebRequest -Uri $Url -OutFile "$TmpDir\$Archive" -UseBasicParsing
    Invoke-WebRequest -Uri $ChecksumUrl -OutFile "$TmpDir\$Archive.sha256" -UseBasicParsing

    # Verify checksum
    $ExpectedHash = (Get-Content "$TmpDir\$Archive.sha256" -Raw).Trim().Split(" ")[0].ToLower()
    $ActualHash = (Get-FileHash "$TmpDir\$Archive" -Algorithm SHA256).Hash.ToLower()

    if ($ActualHash -ne $ExpectedHash) {
        Write-Err "Checksum verification failed.`n  Expected: $ExpectedHash`n  Got:      $ActualHash"
    }

    # Extract
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    Expand-Archive -Path "$TmpDir\$Archive" -DestinationPath $BinDir -Force

    # Update PATH
    $CurrentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($CurrentPath -notlike "*$BinDir*") {
        [Environment]::SetEnvironmentVariable("PATH", "$BinDir;$CurrentPath", "User")
        Write-Info "Added $BinDir to user PATH"
    }

    Write-Host ""
    Write-Success "trevec $Version installed successfully!"
    Write-Host ""
    Write-Info "Run 'trevec --help' to get started."
    Write-Info "You may need to restart your terminal for PATH changes to take effect."
} finally {
    # Cleanup temp directory
    Remove-Item -Path $TmpDir -Recurse -Force -ErrorAction SilentlyContinue
}
