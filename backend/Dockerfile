FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use this instead of CMD with uvicorn directly
CMD ["python", "backend/main.py"]