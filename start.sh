#!/bin/bash
set -e

# Bucket and destination setup
BUCKET_PATH=gs://yolo_pose/vertex_ai_server/app
LOCAL_PATH=/app/app

# Ensure the local directory exists
mkdir -p ${LOCAL_PATH}

echo "📦 Downloading latest app code from $BUCKET_PATH ..."
gsutil -m cp -r ${BUCKET_PATH}/* ${LOCAL_PATH}/

echo "✅ Code synced. Launching FastAPI..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8080
