#!/bin/bash

# Pull the latest version of the code
git pull origin main

# Build the Docker images
docker-compose build

# Start the containers
docker-compose up -d

# Apply database migrations (if using Flask-Migrate or Alembic)
# docker-compose exec backend flask db upgrade

# Optional: Collect static files for production (if needed)
# docker-compose exec backend flask collect

# Restart the backend service to apply changes
docker-compose restart backend

echo "Deployment completed successfully."
