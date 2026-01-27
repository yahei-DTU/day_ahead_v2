FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y \
      build-essential gcc \
      git && \
    apt clean && rm -rf /var/lib/apt/lists/*
