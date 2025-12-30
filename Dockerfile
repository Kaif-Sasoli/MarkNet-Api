# Use Python 3.10 (supported by KenLM)
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

# Copy project files
COPY . /app

# Upgrade pip and install KenLM first
RUN pip install --upgrade pip
RUN pip install git+https://github.com/kpu/kenlm.git

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the Flask app port
EXPOSE 10000

# Start Flask app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
