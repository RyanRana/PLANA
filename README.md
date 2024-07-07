# PLANA: Plan Learning, Analysis, and Advisory

This repository contains a Python script for analyzing various aspects of a business plan, including executive summary, market analysis, financial projections, and additional documentation. The script leverages a combination of machine learning models and OpenAI's GPT-3 to generate a comprehensive grade and actionable feedback for the business.

## Features

- **Executive Summary Analysis**: Uses BERT-based sentiment analysis to evaluate the sentiment of the executive summary, mission, and vision statement.
- **Market Analysis**: Analyzes market sentiment, segments the market, and forecasts market trends using text analysis and clustering.
- **Financial Projections**: Forecasts financial metrics and performs ratio analysis and anomaly detection.
- **Credit Risk and Fraud Detection**: Assesses credit risk and detects potential fraud in financial data.
- **Document Classification and Named Entity Recognition**: Classifies additional documentation and extracts named entities.

## Requirements

You can install all the required packages using the `requirements.txt` file included in this repository.

## Architecture Overview

The application consists of:
- Frontend: User interface for uploading business plans and viewing results
- Backend: Flask server for processing and analysis
- Database: PostgreSQL for storing data
- Machine Learning Models: Various models for analysis tasks

## Setup and Running the Application

### Prerequisites

- Python 3.8 or later
- Docker and Docker Compose

### Step 1: Set Up the Environment

1. Create a virtual environment and install dependencies.

    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment
    # On Windows
    venv\Scripts\activate
    # On Unix or MacOS
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```

### Step 2: Create the Database

2. Ensure the database schema is created.

    ```bash
    # Create the database
    python
    ```

    In the Python shell, run:

    ```python
    from app import db
    db.create_all()
    exit()
    ```

### Step 3: Run the Flask Application

3. Run the Flask application.

    ```bash
    # Set the FLASK_APP environment variable
    export FLASK_APP=app.py
    # On Windows use `set` instead of `export`
    set FLASK_APP=app.py

    # Run the Flask development server
    flask run
    ```

The application will be available at `http://localhost:5000`.

## Running the Application using Docker Compose

To run the application using Docker Compose, follow these steps:

### Step 1: Build and Run the Docker Containers

1. Ensure your `Dockerfile` and `docker-compose.yml` are correctly set up.

#### Dockerfile

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
