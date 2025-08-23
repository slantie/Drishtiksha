@echo off
setlocal enabledelayedexpansion

REM VidVigilante Backend Docker Startup Script for Windows
REM This script helps you start the VidVigilante backend services with Docker

set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Check if Docker is running
echo %BLUE%[INFO]%NC% Checking Docker status...
docker info >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Docker is not running. Please start Docker and try again.
    exit /b 1
)
echo %GREEN%[SUCCESS]%NC% Docker is running

REM Check if Docker Compose is available
echo %BLUE%[INFO]%NC% Checking Docker Compose...
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Docker Compose is not installed. Please install Docker Compose and try again.
    exit /b 1
)
echo %GREEN%[SUCCESS]%NC% Docker Compose is available

REM Check if .env file exists
if not exist .env (
    echo %YELLOW%[WARNING]%NC% .env file not found. Creating from template...
    copy .env.docker .env >nul
    echo %YELLOW%[WARNING]%NC% Please edit .env file and update the following:
    echo %YELLOW%[WARNING]%NC% - POSTGRES_PASSWORD
    echo %YELLOW%[WARNING]%NC% - JWT_SECRET
    echo %YELLOW%[WARNING]%NC% - JWT_REFRESH_SECRET
    echo %YELLOW%[WARNING]%NC% - SERVER_API_KEY
    pause
)
echo %GREEN%[SUCCESS]%NC% .env file found

REM Main script logic
if "%1"=="start" goto start_prod
if "%1"=="dev" goto start_dev
if "%1"=="stop" goto stop_services
if "%1"=="restart" goto restart_services
if "%1"=="status" goto show_status
if "%1"=="logs" goto show_logs
if "%1"=="cleanup" goto cleanup
goto show_help

:start_prod
echo %BLUE%[INFO]%NC% Starting VidVigilante services in production mode...
docker-compose up -d
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Failed to start services
    exit /b 1
)
echo %GREEN%[SUCCESS]%NC% Services started successfully
goto wait_and_show

:start_dev
echo %BLUE%[INFO]%NC% Starting VidVigilante services in development mode...
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Failed to start services
    exit /b 1
)
echo %GREEN%[SUCCESS]%NC% Services started successfully
goto wait_and_show

:stop_services
echo %BLUE%[INFO]%NC% Stopping VidVigilante services...
docker-compose down
echo %GREEN%[SUCCESS]%NC% Services stopped
exit /b 0

:restart_services
echo %BLUE%[INFO]%NC% Restarting VidVigilante services...
docker-compose down
if "%2"=="dev" (
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
) else (
    docker-compose up -d
)
echo %GREEN%[SUCCESS]%NC% Services restarted
goto wait_and_show

:wait_and_show
echo %BLUE%[INFO]%NC% Waiting for services to be healthy...
timeout /t 10 /nobreak >nul
goto show_status

:show_status
echo %BLUE%[INFO]%NC% Service Status:
docker-compose ps
echo.
echo %BLUE%[INFO]%NC% Service URLs:
echo   - Backend API: http://localhost:3000
echo   - PostgreSQL: localhost:5432
echo   - Redis: localhost:6379
echo   - pgAdmin (dev): http://localhost:8080
exit /b 0

:show_logs
echo %BLUE%[INFO]%NC% Recent logs from all services:
docker-compose logs --tail=20
exit /b 0

:cleanup
echo %YELLOW%[WARNING]%NC% This will remove all data including database and uploaded files!
set /p confirm="Are you sure you want to continue? (y/N): "
if /i "!confirm!"=="y" (
    echo %BLUE%[INFO]%NC% Removing services and volumes...
    docker-compose down -v
    echo %GREEN%[SUCCESS]%NC% Cleanup completed
) else (
    echo %BLUE%[INFO]%NC% Cleanup cancelled
)
exit /b 0

:show_help
echo VidVigilante Backend Docker Management Script
echo.
echo Usage: %0 {start^|dev^|stop^|restart^|status^|logs^|cleanup}
echo.
echo Commands:
echo   start     - Start services in production mode
echo   dev       - Start services in development mode (with hot reload)
echo   stop      - Stop all services
echo   restart   - Restart all services
echo   status    - Show service status and URLs
echo   logs      - Show recent logs from all services
echo   cleanup   - Stop services and remove all data (WARNING: destructive)
echo.
echo Examples:
echo   %0 start     # Start in production mode
echo   %0 dev       # Start in development mode
echo   %0 logs      # View logs
exit /b 1
