FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

WORKDIR /home/vmn8si/repos/Sparse4D

# Install basic dependencies (you can add more as needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev && \
    rm -rf /var/lib/apt/lists/* 

# Install Python packages (optional)
RUN pip3 install --upgrade pip && \
    pip3 install \
    torch==1.13.0 \
    torchvision==0.14.0 \
    torchaudio==0.13.0 \
    numpy==1.23.5 \
    mmcv_full==1.7.1 \
    mmdet==2.28.2 \
    urllib3==1.26.16 \
    pyquaternion==0.9.9 \
    nuscenes-devkit==1.1.10 \
    yapf==0.33.0 \
    tensorboard==2.14.0 \
    motmetrics==1.1.3 

RUN pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html


RUN python3 -m ipykernel install --user --name=my-env --display-name "Python (my-env)"

    # Optionally set environment variables for CUDA
ENV PATH /usr/local/cuda-11.3/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

# Default command (can be adjusted as needed)
CMD ["/bin/bash"]
