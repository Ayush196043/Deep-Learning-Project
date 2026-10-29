# ==============================================================================
# One-Click Full-Stack Launcher: Potato Leaf Disease Detection
# Starts FastAPI Backend (Port 8000) & React Frontend (Port 3000)
# ==============================================================================

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  🥔 POTATO LEAF DISEASE DETECTION — FULL-STACK LAUNCHER  " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""

$RootPath = (Get-Location).Path
$VenvPython = Join-Path $RootPath ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment not found at $VenvPython" -ForegroundColor Red
    Write-Host "Please create it using: python -m venv .venv; pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# 1. Start FastAPI Backend in a separate window
Write-Host "[1/3] Starting FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Cyan
$BackendCmd = "cd '$RootPath'; & '$VenvPython' -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "$BackendCmd" -WindowStyle Normal

# Wait 3 seconds for server initialization
Write-Host "Waiting for backend to load CNN weights into memory..." -ForegroundColor Yellow
Start-Sleep -Seconds 4

# 2. Start React Frontend in a separate window
Write-Host "[2/3] Starting React Web Frontend on http://localhost:3000..." -ForegroundColor Cyan
$FrontendDir = Join-Path $RootPath "frontend"
$FrontendCmd = "cd '$FrontendDir'; npm start"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "$FrontendCmd" -WindowStyle Normal

# 3. Ready notification
Write-Host "[3/3] Services launched!" -ForegroundColor Green
Write-Host "----------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "Backend API Docs:    http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "Backend Health:      http://127.0.0.1:8000/health" -ForegroundColor Yellow
Write-Host "React Web Frontend:  http://localhost:3000" -ForegroundColor Cyan
Write-Host "----------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "Your browser will automatically open React on http://localhost:3000" -ForegroundColor Green
Write-Host "Specimen leaf images are located in: data\raw\" -ForegroundColor Cyan
Write-Host "Press Ctrl+C in either terminal window to stop that service." -ForegroundColor DarkGray
