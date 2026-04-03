<# 
  Local runner for AI_IMAGE_VERIFIER (Windows).
  Usage (from repo root in PowerShell):
    .\run_local.ps1 setup                    # venv + pip install
    .\run_local.ps1 smoke                    # tiny synthetic data + 1-epoch train + eval + explain
    .\run_local.ps1 demo                     # Streamlit app (needs a checkpoint under models/)
    .\run_local.ps1 train -- --model cnn --epochs 2
    .\run_local.ps1 eval -- --model cnn --checkpoint models/best_cnn.pth --split val
    .\run_local.ps1 explain -- --model cnn --checkpoint models/best_cnn.pth --split test
    .\run_local.ps1 explore
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'smoke', 'demo', 'train', 'eval', 'explain', 'explore')]
    [string]$Command = 'setup',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Get-PythonExe {
    $venvPy = Join-Path $RepoRoot 'venv\Scripts\python.exe'
    if (Test-Path $venvPy) { return $venvPy }
    return (Get-Command python -ErrorAction Stop).Source
}

function Ensure-Venv {
    $venvDir = Join-Path $RepoRoot 'venv'
    if (-not (Test-Path $venvDir)) {
        Write-Host 'Creating virtual environment: venv' -ForegroundColor Cyan
        & python -m venv $venvDir
    }
    $py = Join-Path $venvDir 'Scripts\python.exe'
    if (-not (Test-Path $py)) {
        throw "venv python not found at $py"
    }
    Write-Host 'Installing / updating dependencies...' -ForegroundColor Cyan
    & $py -m pip install --upgrade pip
    & $py -m pip install -r (Join-Path $RepoRoot 'requirements.txt')
    return $py
}

function Invoke-SmokeData {
    param([string]$PythonExe)
    $trainReal = Join-Path $RepoRoot 'data\raw\train\REAL'
    $first = Get-ChildItem -Path $trainReal -Filter '*.png' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($first) {
        Write-Host 'data/raw/train already has images; skipping synthetic seed.' -ForegroundColor DarkGray
        return
    }
    Write-Host 'Seeding minimal train/REAL and train/FAKE (smoke test only)...' -ForegroundColor Yellow
    $script = @'
from pathlib import Path
from PIL import Image
import numpy as np
for cls, color in [('REAL', (200, 180, 160)), ('FAKE', (80, 120, 200))]:
    d = Path('data/raw/train') / cls
    d.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        arr = np.full((64, 64, 3), color, dtype=np.uint8)
        arr += np.random.randint(0, 15, arr.shape, dtype=np.uint8)
        Image.fromarray(arr).save(d / f'sample_{i:03d}.png')
print('OK')
'@
    & $PythonExe -c $script
}

function Ensure-SmokeTest {
    param([string]$PythonExe)
    $testReal = Join-Path $RepoRoot 'data\raw\test\REAL'
    if (-not (Test-Path $testReal)) { New-Item -ItemType Directory -Force -Path $testReal | Out-Null }
    if (-not (Test-Path (Join-Path $RepoRoot 'data\raw\test\FAKE'))) {
        New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot 'data\raw\test\FAKE') | Out-Null
    }
    $tr = Join-Path $RepoRoot 'data\raw\train\REAL\sample_000.png'
    $tf = Join-Path $RepoRoot 'data\raw\train\FAKE\sample_000.png'
    if ((Test-Path $tr) -and -not (Test-Path (Join-Path $testReal 'sample_000.png'))) {
        Copy-Item $tr (Join-Path $testReal 'sample_000.png')
    }
    if ((Test-Path $tf) -and -not (Test-Path (Join-Path $RepoRoot 'data\raw\test\FAKE\sample_000.png'))) {
        Copy-Item $tf (Join-Path $RepoRoot 'data\raw\test\FAKE\sample_000.png')
    }
}

switch ($Command) {
    'setup' {
        $py = Ensure-Venv
        Write-Host "Done. Python: $py" -ForegroundColor Green
        Write-Host 'Next: place CIFAKE under data/raw, then: .\run_local.ps1 train -- --model efficientnet --epochs 20' -ForegroundColor Gray
    }
    'smoke' {
        $py = Ensure-Venv
        Invoke-SmokeData -PythonExe $py
        Ensure-SmokeTest -PythonExe $py
        Write-Host 'Training (1 epoch, cnn)...' -ForegroundColor Cyan
        & $py (Join-Path $RepoRoot 'src\train.py') '--model', 'cnn', '--epochs', '1', '--batch_size', '4', '--num_workers', '0', '--val_ratio', '0.25', '--lr', '1e-3'
        Write-Host 'Evaluate (val)...' -ForegroundColor Cyan
        & $py (Join-Path $RepoRoot 'src\evaluate.py') '--model', 'cnn', '--checkpoint', (Join-Path $RepoRoot 'models\best_cnn.pth'), '--split', 'val', '--data_root', 'data/raw', '--batch_size', '4', '--num_workers', '0'
        Write-Host 'Explain (test)...' -ForegroundColor Cyan
        & $py (Join-Path $RepoRoot 'src\explain.py') '--model', 'cnn', '--checkpoint', (Join-Path $RepoRoot 'models\best_cnn.pth'), '--split', 'test', '--num_samples', '2', '--data_root', 'data/raw', '--out_dir', 'outputs/explain_smoke'
        Write-Host 'Smoke finished. Check models/best_cnn.pth and outputs/' -ForegroundColor Green
    }
    'demo' {
        $py = Get-PythonExe
        if ($py -notmatch 'venv') {
            $py = Ensure-Venv
        }
        Write-Host 'Starting Streamlit at http://localhost:8501 (Ctrl+C to stop)' -ForegroundColor Cyan
        & $py -m streamlit run (Join-Path $RepoRoot 'app.py')
    }
    'train' {
        $py = Ensure-Venv
        & $py (Join-Path $RepoRoot 'src\train.py') @RemainingArgs
    }
    'eval' {
        $py = Ensure-Venv
        & $py (Join-Path $RepoRoot 'src\evaluate.py') @RemainingArgs
    }
    'explain' {
        $py = Ensure-Venv
        & $py (Join-Path $RepoRoot 'src\explain.py') @RemainingArgs
    }
    'explore' {
        $py = Ensure-Venv
        & $py (Join-Path $RepoRoot 'src\explore.py') @RemainingArgs
    }
}
