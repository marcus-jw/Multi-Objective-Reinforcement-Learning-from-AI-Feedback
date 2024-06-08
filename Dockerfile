#Uses the PyTorch NGC Container as a base image
FROM nvcr.io/nvidia/pytorch:24.02-py3 as base

#ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

WORKDIR /workspace

# Set HF_HOME to a writable directory
ENV HF_HOME=/workspace/.cache/huggingface
RUN mkdir -p $HF_HOME 

# Install SSH server
RUN apt-get update && apt-get install -y openssh-server tmux 
RUN apt-get update && apt-get install -y libaio-dev
RUN mkdir -p ~/.ssh

RUN pip install OhMyRunPod runpod
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . /workspace


EXPOSE 22


