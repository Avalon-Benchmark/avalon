FROM ubuntu:20.04

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install -yqq \
    htop \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3.9-venv \
    wget \
    unzip \
    git

RUN apt-get -yqq install --no-install-recommends \
    libegl-dev \
    libopengl-dev \
    libglew-dev \
    libglfw3-dev \
    libosmesa6 \
    mesa-utils-extra

RUN apt-get clean

RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install -U pip wheel
RUN pip install --no-cache --extra-index-url https://download.pytorch.org/whl/cu113 torch==1.12.0+cu113 torchvision==0.13.0+cu113

COPY ./docker/godot_install.sh /tmp/
RUN bash /tmp/godot_install.sh

COPY ./docker/gl_setup.sh /tmp/
RUN bash /tmp/gl_setup.sh

COPY . /opt/projects/avalon
RUN pip install -e /opt/projects/avalon[train]

WORKDIR /opt/projects/avalon

# from bawr
# apt-get install libegl-dev  libglew-dev libglfw3-dev libnvidia-gl libopengl-dev libosmesa6 mesa-utils-extra --no-install-recommends

# CUDA_VISIBLE_DEVICES=6 python notebooks/avalon_simplest_example.sync.py
# CUDA_VISIBLE_DEVICES=6 python -m avalon.agent.train_ppo_avalon
