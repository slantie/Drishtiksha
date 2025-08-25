# Dockerfile for Drishtiksha Backend & Worker

# --- STAGE 1: Builder Stage ---
# Use a Node.js Alpine image for a small footprint.
FROM node:22-alpine AS builder

# Install build-time dependencies needed for some npm packages (like Prisma).
RUN apk add --no-cache python3 make g++ openssl

WORKDIR /app

# Copy package files and install ALL dependencies (incl. dev).
# This leverages Docker cache for dependencies.
COPY package*.json ./
RUN npm install

# Copy prisma schema and generate the client.
COPY prisma ./prisma/
RUN npx prisma generate

# --- STAGE 2: Production Stage ---
# Start from a fresh, lean Node.js Alpine image.
FROM node:22-alpine

# Install production-only system dependencies.
# postgresql-client is for prisma db push, curl for healthchecks.
RUN apk add --no-cache postgresql-client curl dumb-init

# Create a non-root user and group for security.
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup

# Set the working directory.
WORKDIR /app

# Switch to the non-root user.
USER appuser

# --- BEST PRACTICE: Use --chown on all subsequent COPY commands ---
# This ensures files are owned by the correct user from the start.
COPY --chown=appuser:appgroup package*.json ./

# Install ONLY production dependencies and clean cache.
RUN npm install --omit=dev && npm cache clean --force

# Copy pre-built node_modules and generated Prisma client from the builder stage.
COPY --from=builder --chown=appuser:appgroup /app/node_modules ./node_modules
COPY --from=builder --chown=appuser:appgroup /app/prisma ./prisma/

# Copy the rest of the application source code.
COPY --chown=appuser:appgroup . .

# Switch back to root temporarily to create directories and set permissions.
USER root
RUN mkdir -p /app/storage/media /app/temp /app/logs && \
    chown -R appuser:appgroup /app/storage /app/temp /app/logs && \
    chmod +x /app/scripts/docker-entrypoint.sh
# Switch back to the non-root user permanently.
USER appuser

# Expose the application port.
EXPOSE 3000

# Use the custom entrypoint script that handles database migrations.
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]

# Default command to start the API server. This will be overridden
# by the 'command' in docker-compose.yml for the worker service.
CMD ["dumb-init", "--", "npm", "run", "start"]