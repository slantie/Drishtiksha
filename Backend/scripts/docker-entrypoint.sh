#!/bin/bash
set -e

echo "🚀 Starting VidVigilante Backend..."

# Function to wait for database
wait_for_db() {
    echo "⏳ Waiting for PostgreSQL to be ready..."
    until npx prisma db push --force-reset 2>/dev/null || npx prisma migrate deploy 2>/dev/null; do
        echo "⏳ Database not ready, waiting 2 seconds..."
        sleep 2
    done
    echo "✅ Database is ready!"
}

# Function to run database setup
setup_database() {
    echo "🗄️ Setting up database schema..."
    
    # Try to apply migrations first (for production)
    if npx prisma migrate deploy 2>/dev/null; then
        echo "✅ Migrations applied successfully"
    else
        echo "📝 No migrations found or failed, trying db push..."
        # If migrations fail, try db push (for development)
        if npx prisma db push --accept-data-loss 2>/dev/null; then
            echo "✅ Database schema pushed successfully"
        else
            echo "❌ Failed to set up database schema"
            exit 1
        fi
    fi
    
    # Generate Prisma client if needed
    echo "🔧 Generating Prisma client..."
    npx prisma generate
    echo "✅ Prisma client generated"
}
# Main execution
main() {
    # Wait for database to be available
    wait_for_db
    
    # Set up database schema and migrations
    setup_database
    
    echo "🎉 Database setup completed!"
    echo "🚀 Starting application..."
    
    # Execute the original command
    exec "$@"
}

# Run main function with all arguments
main "$@"
