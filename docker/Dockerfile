
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ros2_distro=foxy
###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=user \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:iscilab" | chpasswd

## basic tools
RUN apt-get update && apt-get install -y \
    curl \
    lsb-release \
    sudo \
    software-properties-common \
    git \
    cmake \
    wget \
    make \
    vim \
    build-essential \
    libblkid-dev \
    e2fslibs-dev \
    libboost-all-dev \
    libaudit-dev \
    tzdata \
    python3-pip

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
RUN pip3 install --upgrade pip

## Dependency installation
RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

## ROS2 installation
RUN apt update && \
    apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \ 
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    apt-get update && \
    apt-get install -y gnupg2 
RUN sh -c 'echo "deb http://mirrors.aliyun.com/ros2/ubuntu/ $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - 
RUN apt-get update 
RUN apt-get install -y ros-$ros2_distro-desktop 

# Install ROS 2 build tools
RUN apt-get update && \
    apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstools \
    python3-flake8 \
    python3-setuptools \
    python3-pytest-cov

RUN echo "ROS $ros2_distro installed successfully"

## Setup ompl
# COPY ./requirement/setup_repo.sh /
# RUN sh /setup_repo.sh
# RUN rm /setup_repo.sh

## setting
RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "${USER} ALL=(ALL) ALL" > /etc/sudoers
RUN echo "root ALL=(ALL) ALL" > /etc/sudoers

# GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# setup entrypoint
COPY ./ros_bashrc /

ENTRYPOINT ["/ros_bashrc"]

USER ${USER}
WORKDIR ${HOME}
RUN ls
