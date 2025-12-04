FROM eclipse-temurin:11-jre-jammy

LABEL maintainer="Anime Spark Project"
LABEL description="Single-node PySpark environment for Anime Spark app (Java 11 + Python 3.9)"

# -----------------------------------------------------------------------------
# System dependencies: Python, basic tools
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ca-certificates \
        bash \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Ensure `python` command is available and points to Python 3
RUN ln -s /usr/bin/python3 /usr/local/bin/python

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


