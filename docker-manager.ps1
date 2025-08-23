# VidVigilante Enhanced Docker Management Script
# Supports local PostgreSQL and cloud database configurations

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("local","cloud","studio","stop","status","logs","build","help")]
    [string]$Command = "help"
)

# Color functions
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Blue }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Warning { Write-Host "[WARNING] $args" -ForegroundColor Yellow }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }

function Start-Local {
    Write-Info "Starting VidVigilante with local PostgreSQL database..."
    docker-compose -f docker-compose.yml -f docker-compose.local.yml up -d
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services started successfully"
        Show-Status
    } else {
        Write-Error "Failed to start services"
        exit 1
    }
}

function Start-Cloud {
    Write-Info "Starting VidVigilante with cloud database (no PostgreSQL container)..."
    Write-Warning "Make sure DATABASE_URL is set in your .env file"
    docker-compose -f docker-compose.yml -f docker-compose.cloud.yml up -d
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services started successfully"
        Show-Status
    } else {
        Write-Error "Failed to start services"
        exit 1
    }
}

function Start-Studio {
    Write-Info "Starting VidVigilante with Prisma Studio..."
    docker-compose -f docker-compose.yml -f docker-compose.local.yml --profile studio up -d
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services started successfully"
        Show-Status
    } else {
        Write-Error "Failed to start services"
        exit 1
    }
}

function Stop-Services {
    Write-Info "Stopping VidVigilante services..."
    docker-compose -f docker-compose.yml -f docker-compose.local.yml down
    docker-compose -f docker-compose.yml -f docker-compose.cloud.yml down
    Write-Success "Services stopped"
}

function Build-Services {
    Write-Info "Building VidVigilante services..."
    docker-compose -f docker-compose.yml -f docker-compose.local.yml build
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services built successfully"
    } else {
        Write-Error "Failed to build services"
        exit 1
    }
}

function Show-Status {
    Write-Info "Service Status:"
    docker-compose ps
    Write-Host ""
    Write-Info "Service URLs:"
    Write-Host "  - Frontend: http://localhost:5173"
    Write-Host "  - Backend API: http://localhost:3000"
    Write-Host "  - Redis: localhost:6379"
    
    # Check if PostgreSQL is running
    $postgresRunning = docker ps | Select-String "vidvigilante-postgres"
    if ($postgresRunning) {
        Write-Host "  - PostgreSQL: localhost:5432"
    } else {
        Write-Host "  - Database: Cloud/External (configured via DATABASE_URL)"
    }
    
    # Check if Prisma Studio is running
    $studioRunning = docker ps | Select-String "vidvigilante-prisma-studio"
    if ($studioRunning) {
        Write-Host "  - Prisma Studio: http://localhost:5555"
    }
    
    Write-Host "  - Static Files: http://localhost:3000/database/media/"
}

function Show-Logs {
    Write-Info "Recent logs from all services:"
    docker-compose logs --tail=20
}

function Show-Help {
    Write-Host ""
    Write-Host "VidVigilante Enhanced Docker Management Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\docker-manager.ps1 {local|cloud|studio|stop|status|logs|build|help}"
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  local     - Start with local PostgreSQL database (default)"
    Write-Host "  cloud     - Start with cloud/external database (no PostgreSQL container)"
    Write-Host "  studio    - Start with Prisma Studio on port 5555"
    Write-Host "  stop      - Stop all services"
    Write-Host "  status    - Show service status and URLs"
    Write-Host "  logs      - Show recent logs from all services"
    Write-Host "  build     - Build all Docker images"
    Write-Host "  help      - Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Green
    Write-Host "  .\docker-manager.ps1 local         # Start with local PostgreSQL"
    Write-Host "  .\docker-manager.ps1 cloud         # Start with cloud database"
    Write-Host "  .\docker-manager.ps1 studio        # Start with Prisma Studio"
    Write-Host "  .\docker-manager.ps1 stop          # Stop all services"
    Write-Host "  .\docker-manager.ps1 status        # Check service status"
    Write-Host ""
    Write-Host "Environment Setup:" -ForegroundColor Cyan
    Write-Host "  For cloud database: Set DATABASE_URL in .env file to your cloud database"
    Write-Host "  For local database: Uses default PostgreSQL container"
    Write-Host ""
}

# Main execution
switch ($Command) {
    "local" { Start-Local }
    "cloud" { Start-Cloud }
    "studio" { Start-Studio }
    "stop" { Stop-Services }
    "status" { Show-Status }
    "logs" { Show-Logs }
    "build" { Build-Services }
    "help" { Show-Help }
    default { Show-Help }
}
