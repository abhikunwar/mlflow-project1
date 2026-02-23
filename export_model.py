import sys
import os
import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

model_name = sys.argv[1]
model_alias = sys.argv[2]

model_uri = f"models:/{model_name}@{model_alias}"

print(f"Downloading model from MLflow: {model_uri}")

mlflow.artifacts.download_artifacts(
    artifact_uri=model_uri,
    dst_path="model"
)
