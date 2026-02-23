FROM python:3.10-slim

WORKDIR /app

# Copy model artifacts
COPY model /app/model
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt mlflow

# No entrypoint for now (we are just building/pushing image)