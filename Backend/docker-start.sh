#!/bin/bash

# VidVigilante Backend Docker Startup Script
# This script helps you start the VidVigilante backend services with Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose > /dev/null 2>&1; then
        print_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Check if .env file exists
check_env_file() {
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from template..."
        cp .env.docker .env
        print_warning "Please edit .env file and update the following:"
        print_warning "- POSTGRES_PASSWORD"
        print_warning "- JWT_SECRET"
        print_warning "- JWT_REFRESH_SECRET"
        print_warning "- SERVER_API_KEY"
        read -p "Press Enter to continue after editing .env file..."
    fi
    print_success ".env file found"
}

# Start services
start_services() {
    print_status "Starting VidVigilante services..."
    
    case "$1" in
        cloud)
            print_status "Starting with cloud database (no local PostgreSQL)..."
            docker-compose -f docker-compose.yml -f docker-compose.cloud.yml up -d
            ;;
        local)
            print_status "Starting with local PostgreSQL database..."
            docker-compose -f docker-compose.yml -f docker-compose.local.yml up -d
            ;;
        studio)
            print_status "Starting with Prisma Studio..."
            docker-compose -f docker-compose.yml -f docker-compose.local.yml --profile studio up -d
            ;;
        *)
            print_status "Starting in default mode (local PostgreSQL)..."
            docker-compose -f docker-compose.yml -f docker-compose.local.yml up -d
            ;;
    esac
    
    print_success "Services started successfully"
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to be healthy..."
    
    max_attempts=30
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker-compose ps | grep -q "Up (healthy)"; then
            print_success "Services are healthy"
            return 0
        fi
        
        attempt=$((attempt + 1))
        print_status "Waiting for services... ($attempt/$max_attempts)"
        sleep 2
    done
    
    print_warning "Services may not be fully healthy yet. Check logs with: docker-compose logs"
}

# Show service status
show_status() {
    print_status "Service Status:"
    docker-compose ps
    
    echo ""
    print_status "Service URLs:"
    echo "  - Backend API: http://localhost:3000"
    echo "  - Redis: localhost:6379"
    
    # Check if PostgreSQL is running (local mode)
    if docker ps | grep -q vidvigilante-postgres; then
        echo "  - PostgreSQL: localhost:5432"
    else
        echo "  - Database: Cloud/External (configured via DATABASE_URL)"
    fi
    
    # Check if Prisma Studio is running
    if docker ps | grep -q vidvigilante-prisma-studio; then
        echo "  - Prisma Studio: http://localhost:5555"
    fi
}

# Show logs
show_logs() {
    print_status "Recent logs from all services:"
    docker-compose logs --tail=20
}

# Stop services
stop_services() {
    print_status "Stopping VidVigilante services..."
    docker-compose down
    print_success "Services stopped"
}

# Cleanup (remove volumes)
cleanup() {
    print_warning "This will remove all data including database and uploaded files!"
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing services and volumes..."
        docker-compose down -v
        print_success "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Main script logic
case "$1" in
    start)
        check_docker
        check_docker_compose
        check_env_file
        start_services "$2"
        wait_for_services
        show_status
        ;;
    local)
        check_docker
        check_docker_compose
        check_env_file
        start_services "local"
        wait_for_services
        show_status
        ;;
    cloud)
        check_docker
        check_docker_compose
        check_env_file
        start_services "cloud"
        wait_for_services
        show_status
        ;;
    studio)
        check_docker
        check_docker_compose
        check_env_file
        start_services "studio"
        wait_for_services
        show_status
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        start_services "$2"
        wait_for_services
        show_status
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    cleanup)
        cleanup
        ;;
    *)
        echo "VidVigilante Backend Docker Management Script"
        echo ""
        echo "Usage: $0 {start|local|cloud|studio|stop|restart|status|logs|cleanup}"
        echo ""
        echo "Commands:"
        echo "  start     - Start services with local PostgreSQL (default)"
        echo "  local     - Start services with local PostgreSQL database"
        echo "  cloud     - Start services with cloud/external database (no PostgreSQL container)"
        echo "  studio    - Start services with Prisma Studio included"
        echo "  stop      - Stop all services"
        echo "  restart   - Restart all services"
        echo "  status    - Show service status and URLs"
        echo "  logs      - Show recent logs from all services"
        echo "  cleanup   - Stop services and remove all data (WARNING: destructive)"
        echo ""
        echo "Examples:"
        echo "  $0 start         # Start with local PostgreSQL"
        echo "  $0 local         # Start with local PostgreSQL"
        echo "  $0 cloud         # Start with cloud database (set DATABASE_URL in .env)"
        echo "  $0 studio        # Start with Prisma Studio on port 5555"
        echo "  $0 logs          # View logs"
        echo ""
        echo "Environment Setup:"
        echo "  For cloud database: Set DATABASE_URL in .env file"
        echo "  For local database: Use default PostgreSQL container"
        exit 1
        ;;
esac
