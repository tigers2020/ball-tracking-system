# Script to activate the virtual environment
if (Test-Path -Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment for Tennis Ball Tracker..." -ForegroundColor Green
    & ".\.venv\Scripts\Activate.ps1"
} else {
    Write-Host "Virtual environment not found at .venv\Scripts\Activate.ps1" -ForegroundColor Red
    Write-Host "Creating a new virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if (Test-Path -Path ".\.venv\Scripts\Activate.ps1") {
        Write-Host "Virtual environment created successfully, activating..." -ForegroundColor Green
        & ".\.venv\Scripts\Activate.ps1"
        Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Blue
        pip install -r requirements.txt
    } else {
        Write-Host "Failed to create virtual environment. Please check your Python installation." -ForegroundColor Red
    }
} 