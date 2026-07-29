FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/CompMVR

WORKDIR /CompMVR

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    pkg-config \
    ninja-build \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libgomp1 \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    triton==2.0.0 \
    --index-url https://download.pytorch.org/whl/cu117

COPY . /CompMVR

CMD ["/bin/bash"]