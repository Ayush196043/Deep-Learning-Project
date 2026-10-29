# ==============================================================================
# Google Cloud Run Deployment Script (PowerShell)
# Potato Leaf Disease Detection - Production ML Service
# ==============================================================================

[CmdletBinding()]
param (
    [string]$ProjectId = "potato-leaf-disease-ai",
    [string]$Region = "asia-south1",
    [string]$RepoName = "potato-app-repo",
    [string]$ServiceName = "potato-disease-api",
    [string]$ImageTag = "v1.0.0"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " [GCP CLOUD RUN DEPLOYMENT] Potato Leaf Disease Classifier " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Project ID:    $ProjectId" -ForegroundColor Yellow
Write-Host "Region:        $Region" -ForegroundColor Yellow
Write-Host "Service Name:  $ServiceName" -ForegroundColor Yellow
Write-Host "Artifact Tag:  $ImageTag" -ForegroundColor Yellow
Write-Host ""

# 1. Check if gcloud CLI is available
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "[ERROR] Google Cloud SDK ('gcloud') is not installed or not in PATH. Please install from https://cloud.google.com/sdk"
    exit 1
}

# 2. Configure active project
Write-Host "[1/5] Setting active project..." -ForegroundColor Cyan
gcloud config set project $ProjectId

# 3. Enable required APIs
Write-Host "[2/5] Enabling GCP APIs..." -ForegroundColor Cyan
gcloud services enable `
    run.googleapis.com `
    cloudbuild.googleapis.com `
    artifactregistry.googleapis.com `
    storage.googleapis.com `
    logging.googleapis.com

# 4. Check or create Artifact Registry repository
Write-Host "[3/5] Verifying Artifact Registry repository..." -ForegroundColor Cyan
$repoCheck = gcloud artifacts repositories describe $RepoName --location=$Region 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating Docker repository '$RepoName' in $Region..." -ForegroundColor Yellow
    gcloud artifacts repositories create $RepoName `
        --repository-format=docker `
        --location=$Region `
        --description="Docker repository for Potato Leaf Disease FastAPI backend"
}

# 5. Build and submit container image via Cloud Build
$FullImageUrl = "$Region-docker.pkg.dev/$ProjectId/$RepoName/${ServiceName}:$ImageTag"
Write-Host "[4/5] Building container image via Google Cloud Build..." -ForegroundColor Cyan
Write-Host "Target Image: $FullImageUrl" -ForegroundColor DarkCyan
gcloud builds submit --tag $FullImageUrl .

# 6. Deploy to Google Cloud Run with cost-conscious settings
Write-Host "[5/5] Deploying container to Cloud Run (Scale-to-Zero)..." -ForegroundColor Cyan
gcloud run deploy $ServiceName `
    --image $FullImageUrl `
    --platform managed `
    --region $Region `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --min-instances 0 `
    --max-instances 5 `
    --concurrency 80 `
    --timeout 60s `
    --set-env-vars="MODEL_PATH=/app/models/saved_models/baseline_best.keras,PORT=8080"

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " [SUCCESS] Deployment completed successfully!" -ForegroundColor Green
Write-Host " Retrieve your service URL above and test with:" -ForegroundColor Cyan
Write-Host "   curl https://<SERVICE_URL>/health" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Green
