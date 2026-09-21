# MUYAH-CODE installer for Windows.
#
#   irm https://raw.githubusercontent.com/MUYAHGaious/muyah-code/main/install.ps1 | iex
#
# Installs with uv or pipx when you have one (each keeps MUYAH-CODE in its own environment), otherwise with
# pip for your user. Then it makes sure `muyah` works from any folder: the launcher's folder is added to
# your user PATH, and to this terminal too, so you can type `muyah` right away.

$ErrorActionPreference = "Stop"
$Source = "git+https://github.com/MUYAHGaious/muyah-code"

function Has($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

function Add-ToSessionPath($dir) {
    if ($dir -and (Test-Path $dir) -and -not (($env:Path -split ";") -contains $dir)) {
        $env:Path = "$env:Path;$dir"
    }
}

function Find-Python {
    foreach ($candidate in @("py", "python", "python3")) {
        if (Has $candidate) {
            & $candidate -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        }
    }
    return $null
}

Write-Host "Installing MUYAH-CODE..." -ForegroundColor Cyan

if (Has "uv") {
    uv tool install --force $Source
    uv tool update-shell
    Add-ToSessionPath (uv tool dir --bin)
}
elseif (Has "pipx") {
    pipx install --force $Source
    pipx ensurepath
    Add-ToSessionPath (Join-Path $env:USERPROFILE ".local\bin")
}
else {
    $python = Find-Python
    if (-not $python) {
        Write-Host "MUYAH-CODE needs Python 3.10 or newer: https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "(tick 'Add python.exe to PATH' in the installer), then run this again."
        return
    }
    & $python -m pip install --user --upgrade --quiet $Source
    if ($LASTEXITCODE -ne 0) { throw "pip could not install MUYAH-CODE (see the messages above)." }
    & $python -m muyah_code path
    Add-ToSessionPath (& $python -c "from muyah_code.pathfix import find_launcher_dir; print(find_launcher_dir() or '')")
}

if (Has "muyah") {
    Write-Host ""
    Write-Host "Done. Type 'muyah' in any folder to start (new terminals find it too)." -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "Installed. Open a new terminal and type 'muyah'. If it is still not found, run: python -m muyah_code path" -ForegroundColor Yellow
}
