FROM python:3.9-slim

LABEL maintainer="Anime Spark Project"
LABEL description="Single-node PySpark environment for Anime Spark app"

# -----------------------------------------------------------------------------
# System dependencies: Java (for Spark), basic tools
# Use default-jre-headless so it works on newer Debian (e.g. trixie)
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        default-jre-headless \
        ca-certificates \
        bash \
    && rm -rf /var/lib/apt/lists/*

ENV PYSPARK_PYTHON=python

# -----------------------------------------------------------------------------
# Python dependencies
# -----------------------------------------------------------------------------
WORKDIR /app

# Copy only requirements first for better build caching
COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------
# The project source code will be mounted from the host into /app by the
# helper scripts (run_docker_mac_linux.sh / run_docker_windows.bat),
# so we don't COPY the whole repo here.

CMD ["python", "main.py"]


