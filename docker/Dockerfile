FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
# Install some basic utilities
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
    g++ \
    gdb \
    make \
    cmake \
    nano \
    libsparsehash-dev \
    wget \
    unzip \
    openmpi-bin \
    libopenmpi-dev \
    python3-opencv

RUN chmod -R  a+rwX /opt/conda
#RUN useradd -ms /bin/bash user
RUN useradd -ms /bin/bash -u 1003 user

USER user    
ENV HOME=/home/user
ENV PATH="${PATH}:/user/.local/bin"
# Create a working directory
WORKDIR ${HOME}
#RUN chmod 777 ${HOME}


