# Dockerfile for VidVigilante Backend & Worker

# --- STAGE 1: Builder Stage ---
FROM node:22-alpine AS builder

WORKDIR /app

RUN apk add --no-cache python3 make g++

COPY package*.json ./
RUN npm install

COPY prisma ./prisma/
RUN npx prisma generate


# --- STAGE 2: Create the final, lean production image ---
FROM node:22-alpine

RUN apk add --no-cache \
    postgresql-client \
    curl \
    bash \
    dumb-init

RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup

# Set the working directory and change ownership to appuser
WORKDIR /app
RUN chown appuser:appgroup /app

# Switch to the non-root user
USER appuser

# --- NEW FIX: Use the --chown flag on all COPY commands after USER is set ---
# This ensures the copied files are owned by the correct user from the start.
COPY --chown=appuser:appgroup package*.json ./

# Install only production dependencies and clean cache
# This will now succeed because package-lock.json is owned by appuser
RUN npm install --omit=dev && npm cache clean --force

# Copy the pre-built node_modules and generated Prisma client from the builder stage
COPY --from=builder --chown=appuser:appgroup /app/node_modules ./node_modules
COPY --from=builder --chown=appuser:appgroup /app/prisma ./prisma/

# Copy the rest of the application source code
COPY --chown=appuser:appgroup . .

# Make the entrypoint script executable
USER root
RUN chmod +x /app/scripts/docker-entrypoint.sh
USER appuser

# Create necessary directories with proper permissions
USER root
RUN mkdir -p /app/storage/media/videos /app/storage/media/visualizations \
    /app/uploads/videos /app/uploads/visualizations \
    /app/temp /app/logs && \
    chown -R appuser:appgroup /app/storage /app/uploads /app/temp /app/logs
USER appuser

# Expose the application port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1

# Use the custom entrypoint script that handles database setup
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]

# Default command to start the application
CMD ["dumb-init", "--", "npm", "run", "start:full"]