FROM python:3.11-slim

# Install system dependencies for Playwright/Chromium
RUN apt-get update && apt-get install -y \
    wget gnupg curl git \
    libnss3 libatk-bridge2.0-0 libdrm2 libxkbcommon0 \
    libgtk-3-0 libgbm1 libasound2 && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"
WORKDIR /app