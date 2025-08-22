# --- STAGE 1: Build the React application ---
# Use an official Node.js image. The 'alpine' version is lightweight.
# Match the Node version to your project's requirements (e.g., node:18 or node:20).
FROM node:18-alpine AS builder

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json (or npm-shrinkwrap.json)
# This step is separated to leverage Docker's layer caching.
# Dependencies are only re-installed if these files change.
COPY package*.json ./

# Install project dependencies
RUN npm install

# Copy the rest of the application's source code
COPY . .

# IMPORTANT: Handle Build-Time Environment Variables
# Use ARG to allow passing the backend URL during the Docker build process.
# We will set this in our docker-compose.yml file later.
ARG VITE_BACKEND_URL
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}

# Build the application for production
# This command runs "vite build" as defined in your package.json
RUN npm run build

# --- STAGE 2: Serve the application using Nginx ---
# Use a lightweight, official Nginx image
FROM nginx:stable-alpine

# Copy the custom Nginx configuration file
# We will create this file next. It's crucial for React Router.
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built static files from the 'builder' stage
# The 'dist' folder is Vite's default build output directory.
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 80 to the outside world
EXPOSE 80

# The command to start Nginx when the container launches
CMD ["nginx", "-g", "daemon off;"]