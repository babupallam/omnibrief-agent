# Build:
#   docker build -t omnibrief .
#
# Run with your environment file:
#   docker run --rm --env-file .env -v "$(pwd)/output:/app/output" omnibrief

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        fonts-liberation \
        libasound2 \
        libatk-bridge2.0-0 \
        libatk1.0-0 \
        libcairo2 \
        libcups2 \
        libdbus-1-3 \
        libdrm2 \
        libexpat1 \
        libfontconfig1 \
        libgbm1 \
        libglib2.0-0 \
        libgtk-3-0 \
        libnspr4 \
        libnss3 \
        libpango-1.0-0 \
        libx11-6 \
        libx11-xcb1 \
        libxcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxkbcommon0 \
        libxrandr2 \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium

COPY src/ ./src/
COPY targets.json .
COPY .env.example .

RUN mkdir -p output

CMD ["python", "src/main.py"]
