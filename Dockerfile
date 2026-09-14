# 1. Use a highly optimized, lightweight Python base image
FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files to disk and buffering stdout (Better for Render logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Install OS-level dependencies: Tesseract OCR
# We clean up the apt cache immediately to keep the image size extremely small
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Copy the requirements file and install Python dependencies
# Using --no-cache-dir is CRITICAL to prevent RAM spikes during the Render build process
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code into the container
COPY . .

# 7. Start the FastAPI server using Uvicorn
# Render dynamically assigns a PORT environment variable, so we bind to it.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]