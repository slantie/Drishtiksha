# Dockerfile for the Frontend React Application

# --- STAGE 1: Build Stage ---
# Use a Node.js image to build the React app. 'alpine' is lightweight.
FROM node:18-alpine AS builder

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json to leverage Docker's layer caching.
# Dependencies are only re-installed if these files change.
COPY package*.json ./

# Install project dependencies
RUN npm install

# Copy the rest of the application's source code
COPY . .

# Use ARG to pass environment variables during the Docker build process.
# These are used by Vite to bake values into the static files.
ARG VITE_BACKEND_URL
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}

# Build the application for production. This runs "vite build".
RUN npm run build

# --- STAGE 2: Production Stage ---
# Use a lightweight, official Nginx image to serve the static files.
FROM nginx:stable-alpine

# Copy the custom Nginx configuration. This is crucial for React Router.
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built static files from the 'builder' stage's 'dist' folder.
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 80 to the outside world (Nginx's default port)
EXPOSE 80

# The command to start Nginx when the container launches.
# '-g daemon off;' ensures Nginx runs in the foreground, which is standard for containers.
CMD ["nginx", "-g", "daemon off;"]