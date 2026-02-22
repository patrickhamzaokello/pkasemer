FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Create data directory for SQLite DB and logs
RUN mkdir -p /data /app/logs

# Default: run the scheduler (overridden in docker-compose per service)
CMD ["python", "scheduler.py"]