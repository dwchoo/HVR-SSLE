FROM nvcr.io/nvidia/pytorch:24.06-py3

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential python3-dev pkg-config git \
        libgl1 libsm6 libxext6 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt
    
RUN python -m pip install --no-cache-dir 'opencv-python-headless==4.8.0.74'

# ENV WANDB_DISABLED=true  

CMD ["/bin/bash"]
# docker build -t hvr-ssle:latest .