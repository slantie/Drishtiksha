#!/bin/bash
set -e

# Database initialization script for VidVigilante
echo "Initializing VidVigilante database..."

# Create the main database if it doesn't exist
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create extensions if needed
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pg_trgm";
    
    -- Grant necessary permissions
    GRANT ALL PRIVILEGES ON DATABASE "$POSTGRES_DB" TO "$POSTGRES_USER";
    
    -- Log successful initialization
    SELECT 'VidVigilante database initialized successfully' as status;
EOSQL

echo "Database initialization completed!"
