#!/usr/bin/env bash

# Simple helper script to run the Anime Spark app inside a Docker container with Spark.
# It will check if the Docker image exists; if not, it will build it from the local Dockerfile.
#
# Default image name: my-spark-img
# Usage:
#   ./run_docker_mac_linux.sh
#   IMAGE_NAME=custom-spark-img ./run_docker_mac_linux.sh

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-my-spark-img}"

echo "=== Anime Spark – Docker launcher (image: ${IMAGE_NAME}) ==="
echo

# Check if image exists; if not, build it
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Docker image '${IMAGE_NAME}' not found. Building from local Dockerfile..."
  docker build -t "${IMAGE_NAME}" .
  echo
fi

echo "Running Anime Spark in Docker..."
echo
echo "Current directory will be mounted into /app inside the container."
echo

docker run --rm -it \
  -v "$(pwd)":/app \
  -w /app \
  "${IMAGE_NAME}" \
  python main.py


