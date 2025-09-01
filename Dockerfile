# # Use Python 3.11 slim image as base
# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONPATH=/app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     wget \
#     curl \
#     unzip \
#     gnupg \
#     && rm -rf /var/lib/apt/lists/*

# # Install Chrome for Selenium with better container support
# RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
#     && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
#     && apt-get update \
#     && apt-get install -y google-chrome-stable \
#     && rm -rf /var/lib/apt/lists/* \
#     && google-chrome --version

# # Install ChromeDriver
# RUN CHROME_DRIVER_VERSION=$(curl -s https://storage.googleapis.com/chrome-for-testing-public/last-known-good-versions-with-downloads.json | grep -oP '"linux64":\[{"url":"[^"]*","version":"\K[0-9.]*' | head -1) \
#     && wget -q --continue -P /tmp/ "https://storage.googleapis.com/chrome-for-testing-public/${CHROME_DRIVER_VERSION}/linux64/chromedriver-linux64.zip" \
#     && unzip /tmp/chromedriver-linux64.zip -d /tmp/ \
#     && mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/ \
#     && chmod +x /usr/local/bin/chromedriver \
#     && rm -rf /tmp/chromedriver*

# # Copy requirements first for better caching
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the application code
# # COPY test_api.py .
# COPY backend/ ./backend/
# COPY output/ ./output/
# # COPY logs/ ./logs/

# # Create necessary directories
# RUN mkdir -p /app/output /app/logs
# # Create necessary directories
# RUN mkdir -p /app/logs /app/output/models

# # Expose the port that FastAPI will run on
# EXPOSE 10000

# # Health check
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:10000/api/health || exit 1

# # Command to run the application
# CMD ["uvicorn", "backend.app.api:app", "--host", "0.0.0.0", "--port", "10000"]



# ------------------------------------------------------


    FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    SELENIUM_MANAGER_VERBOSE=1

WORKDIR /app

# OS deps + headless Chrome libs
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl wget ca-certificates gnupg unzip xdg-utils \
      libnss3 libnspr4 libx11-xcb1 libxcb1 libxcomposite1 libxrandr2 \
      libxdamage1 libxfixes3 libgbm1 libasound2 libdrm2 libxshmfence1 \
      libglib2.0-0 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome (keyring method; no apt-key)
RUN wget -qO /usr/share/keyrings/googlechrome.gpg https://dl.google.com/linux/linux_signing_key.pub \
 && echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/googlechrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' \
    > /etc/apt/sources.list.d/google-chrome.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends google-chrome-stable \
 && rm -rf /var/lib/apt/lists/* \
 && google-chrome --version

# Python deps first for cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY backend/ ./backend/
COPY output/ ./output/
RUN mkdir -p /app/logs /app/output/models

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:10000/api/health || exit 1

CMD ["uvicorn", "backend.app.api:app", "--host", "0.0.0.0", "--port", "10000"]
