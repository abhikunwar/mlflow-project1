import mlflow
import os
import sys

# CLI args
model_name = sys.argv[1] if len(sys.argv) > 1 else "dev-version1"
model_stage = sys.argv[2] if len(sys.argv) > 2 else "champion"

model_uri = f"models:/{model_name}/{model_stage}"
print(f"Downloading model from MLflow: {model_uri}")

# Download artifacts to local folder
docker_model_path = "model"
os.makedirs(docker_model_path, exist_ok=True)
mlflow.artifacts.download_artifacts(model_uri, docker_model_path)

print(f"Model saved to: {docker_model_path}")
