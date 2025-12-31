# Python 3.10 is REQUIRED (KenLM compatible)
FROM python:3.10-slim

# Install ALL required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libboost-all-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (Docker cache optimization)
COPY requirements.txt .

RUN pip install --upgrade pip

# Install gunicorn explicitly
RUN pip install gunicorn

# Install KenLM BEFORE flashlight-text
RUN pip install git+https://github.com/kpu/kenlm.git

# Install remaining Python deps (opencv, ultralytics, torch, etc.)
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Render provides $PORT automatically
EXPOSE 10000

CMD gunicorn --bind 0.0.0.0:$PORT app:app
