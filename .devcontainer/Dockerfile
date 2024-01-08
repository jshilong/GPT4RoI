FROM mcr.microsoft.com/devcontainers/base:ubuntu-20.04

SHELL [ "bash", "-c" ]

# update apt and install packages
RUN apt update && \
    apt install -yq \
        build-essential \
        cmake \
        dkms \
        ffmpeg \
        libturbojpeg \
        libgl1-mesa-glx \
        python3-dev \
        ninja-build \
        jq \
        jp \
        tree \
        tldr

# add git-lfs and install
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
    sudo apt-get install -yq git-lfs && \
    git lfs install

############################################
# Setup user
############################################

USER vscode

# Setup conda
RUN cd /tmp && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b && \
    rm ./Miniconda3-latest-Linux-x86_64.sh
