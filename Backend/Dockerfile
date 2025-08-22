# --- STAGE 1: Build dependencies and generate Prisma client ---
FROM node:18-alpine AS builder

# Set the working directory
WORKDIR /app

# Install dependencies needed for 'psql' command if you use PostgreSQL seeding/scripts
# If you only use SQLite or don't need the command-line tool, you can remove this line.
# RUN apk add --no-cache postgresql-client

# Copy package files
COPY package*.json ./

# Install all dependencies, including devDependencies for Prisma
RUN npm install

# Copy the Prisma schema and migrations
COPY prisma ./prisma/

# Generate the Prisma client. This is crucial for the app to connect to the database.
# The client is generated based on the schema and the target database provider.
RUN npx prisma generate

# --- STAGE 2: Create the final production image ---
FROM node:18-alpine

# Set the working directory
WORKDIR /app

# Copy package files again to install only production dependencies
COPY package*.json ./

# Install only production dependencies and clean the npm cache
RUN npm install --omit=dev && npm cache clean --force

# Copy the built node_modules and the generated Prisma client from the builder stage
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/prisma ./prisma/

# Copy the application source code
COPY . .

# Expose the port the application will run on (default is 3000)
EXPOSE 3000

# The command to run the application in production
# This runs "node server.js" as defined in your package.json 'start' script.
CMD ["npm", "start"]