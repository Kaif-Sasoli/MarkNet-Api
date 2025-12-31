FROM python:3.10-slim

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
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install gunicorn

# KenLM first
RUN pip install git+https://github.com/kpu/kenlm.git

RUN pip install -r requirements.txt

COPY . .

EXPOSE 10000

CMD gunicorn --bind 0.0.0.0:$PORT app:app
