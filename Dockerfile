FROM python:3.9-slim

# Встановлюємо Java 11 (рекомендована для Spark 3.4.1)
# Завантажуємо Java 11 безпосередньо з Adoptium
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    && wget -O /tmp/openjdk.tar.gz https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.21+9/OpenJDK11U-jdk_aarch64_linux_hotspot_11.0.21_9.tar.gz || \
    wget -O /tmp/openjdk.tar.gz https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.21+9/OpenJDK11U-jdk_x64_linux_hotspot_11.0.21_9.tar.gz && \
    mkdir -p /usr/lib/jvm && \
    tar -xzf /tmp/openjdk.tar.gz -C /usr/lib/jvm && \
    mv /usr/lib/jvm/jdk-11.0.21+9 /usr/lib/jvm/java-11 && \
    rm /tmp/openjdk.tar.gz && \
    apt-get remove -y wget && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Встановлюємо змінні середовища для Java
ENV JAVA_HOME=/usr/lib/jvm/java-11
ENV PATH=$PATH:$JAVA_HOME/bin

# Встановлюємо PySpark та інші залежності
COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

# Копіюємо код проекту
COPY . .

# Встановлюємо робочу директорію
WORKDIR /app

# Команда запуску
CMD ["python", "main.py"]
