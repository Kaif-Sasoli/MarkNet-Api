# Use Python 3.10 (compatible with KenLM)
FROM python:3.10-slim

# Install system dependencies
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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt /app/

# Upgrade pip and install dependencies
RUN pip install --upgrade pip

# Install KenLM first (needed for flashlight-text)
RUN pip install git+https://github.com/kpu/kenlm.git

# Install remaining Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the project files
COPY . /app

# Expose port
EXPOSE 10000

# Use shell form CMD so Render can find gunicorn
CMD gunicorn --bind 0.0.0.0:10000 app:app
