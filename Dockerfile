#Uses the PyTorch NGC Container as a base image
FROM nvcr.io/nvidia/pytorch:24.02-py3 as base


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

ARG UID=10001
RUN useradd -m -s /bin/bash -N -u $UID appuser

# Set HF_HOME to a writable directory
ENV HF_HOME=/workspace/.cache/huggingface
# Ensure the directory exists and is owned by appuser
RUN mkdir -p $HF_HOME && chown -R appuser /workspace/.cache


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER appuser

COPY . /workspace
