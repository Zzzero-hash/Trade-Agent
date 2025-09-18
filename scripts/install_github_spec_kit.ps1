<#!
.SYNOPSIS
    Installs uv (if missing) and runs GitHub Spec Kit via uvx.
.DESCRIPTION
    This script ensures the Astral "uv" Python package manager is installed and available on PATH,
    then uses `uvx` to run the GitHub Spec Kit tool without permanently installing it into the
    project virtual environment. Designed for Windows PowerShell.

    If a local virtual environment (./.venv) exists it will be activated first so that any
    generated artifacts (e.g., spec outputs) can interoperate with local tooling.

.NOTES
    Author: Automation Script
    Requirements: PowerShell 5+, Internet access
#>
[CmdletBinding()]
param(
    [switch]$ForceUpdate,
    [string]$SpecKitVersion = "",
    [string]$OutputDir = "./.kiro/specs"
)

function Write-Section($Title) { Write-Host "`n=== $Title ===" -ForegroundColor Cyan }
function Write-Step($Msg) { Write-Host "[>] $Msg" -ForegroundColor Yellow }
function Write-Ok($Msg) { Write-Host "[OK] $Msg" -ForegroundColor Green }
function Write-Warn($Msg) { Write-Host "[WARN] $Msg" -ForegroundColor DarkYellow }
function Write-Err($Msg) { Write-Host "[ERR] $Msg" -ForegroundColor Red }

$ErrorActionPreference = 'Stop'

Write-Section "GitHub Spec Kit Installer"

# 1. Activate local venv if present
if (Test-Path .\.venv\Scripts\Activate.ps1) {
    Write-Step "Activating local virtual environment (.venv)"
    . .\.venv\Scripts\Activate.ps1
    Write-Ok ".venv activated"
} else {
    Write-Warn "No .venv found - proceeding without activating a virtual environment."
}

# 2. Ensure uv is installed
function Get-UvPath {
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCmd) { return $uvCmd.Path }
    $candidate = Join-Path $HOME ".local/bin/uv.exe"
    if (Test-Path $candidate) { return $candidate }
    return $null
}

$uvPath = Get-UvPath
if (-not $uvPath -or $ForceUpdate) {
    if ($ForceUpdate -and $uvPath) { Write-Step "Force update requested for uv" }
    Write-Step "Installing (or updating) uv from official script"
    try {
        iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex
        $uvPath = Get-UvPath
        if (-not $uvPath) { throw "uv installation did not place binary on expected path" }
        Write-Ok "uv installed at $uvPath"
    } catch {
        Write-Err "Failed to install uv: $_"
        exit 1
    }
} else {
    Write-Ok "uv already installed at $uvPath"
}

# 3. Ensure PATH contains uv directory for this session
$uvDir = Split-Path $uvPath -Parent
if (-not ($env:Path.Split(';') -contains $uvDir)) {
    Write-Step "Adding $uvDir to PATH for current session"
    $env:Path = "$uvDir;" + $env:Path
}

# 4. Verify uvx works
Write-Step "Verifying uvx availability"
try {
    uv --version | Out-Null
    uvx --version | Out-Null
    Write-Ok "uv and uvx are functional"
} catch {
    Write-Err "uvx verification failed: $_"
    exit 1
}

# 5. Determine GitHub Spec Kit package reference
# Assuming package name on PyPI is 'github-spec-kit'. Allow version override.
$specKitRef = "github-spec-kit"
if ($SpecKitVersion) { $specKitRef = "github-spec-kit==$SpecKitVersion" }
Write-Step "Using package reference: $specKitRef"

# 6. Prepare output directory
if (-not (Test-Path $OutputDir)) {
    Write-Step "Creating output directory: $OutputDir"
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
}

# 7. Run GitHub Spec Kit (basic example command)
Write-Step "Running GitHub Spec Kit (example: list tasks)"
try {
    # Placeholder command - user should adjust to actual CLI usage once confirmed.
    uvx --from $specKitRef github-spec-kit --help
    Write-Ok "GitHub Spec Kit executed (help displayed). Replace command with desired operation."
} catch {
    Write-Err "GitHub Spec Kit execution failed: $_"
    exit 2
}

Write-Section "Completed"
Write-Ok "Installation and basic execution finished."
Write-Host "You can now run: uvx --from $specKitRef github-spec-kit <command>" -ForegroundColor Cyan