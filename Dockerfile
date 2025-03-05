#CUDA version: 11.3.1  <-->  CUDNN version: 8  <-->  Ubuntu version: 20.04
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04



#Install basic library
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
    libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



#Install nvidia-docker
RUN apt update && \
    apt upgrade -y

RUN apt-get install -y curl
RUN curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
RUN curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu22.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
RUN apt-get update
RUN apt-get install -y nvidia-docker2



#Install python
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install --assume-yes apt-utils
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y python3.8
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN apt-get install -y python3-pip



#Install python site-packages/packages, requirements
COPY requirements.txt /

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install -r /requirements.txt
RUN pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
