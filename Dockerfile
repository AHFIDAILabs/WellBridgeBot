# Dockerfile for Lighthouse HealthConnect. Use an official Python runtime as a parent image
FROM python:3.11.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements-dev.txt .

# Add this line to copy env vars if using local builds (file if it exists (for local development))
#COPY .env .env
# COPY .env* ./

# Install any needed packages specified in requirements.txt
# Add ffmpeg for audio processing
#RUN apt-get update && apt-get install -y ffmpeg
#RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies including Redis
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    portaudio19-dev \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Set up HuggingFace cache directories
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# Create cache directory with proper permissions (writable for runtime downloads)
RUN mkdir -p /app/.cache/huggingface/transformers && chmod -R 777 /app/.cache

# Note: Models will be downloaded at first runtime when HUGGINGFACE_API_TOKEN 
# environment variable is available from Cloud Run settings

# Make port 10000 available to the world outside this container
EXPOSE 10000

# Ensure Python can find modules
ENV PYTHONPATH=/app

# Run app.py when the container launches with extended timeout for model downloads
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--chdir", "/app", "--timeout", "600", "--workers", "1"]