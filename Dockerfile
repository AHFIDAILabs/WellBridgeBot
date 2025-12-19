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

# Pre-download Hugging Face models to cache them in the image
# This prevents runtime download failures and speeds up container startup
ARG HUGGINGFACE_API_TOKEN
ENV HUGGINGFACE_API_TOKEN=$HUGGINGFACE_API_TOKEN
RUN if [ -n "$HUGGINGFACE_API_TOKEN" ]; then \
    python3 -c "from transformers import pipeline; \
    import os; \
    token = os.getenv('HUGGINGFACE_API_TOKEN'); \
    print('Downloading N-ATLAS models...'); \
    try: \
        pipeline('automatic-speech-recognition', model='NCAIR1/Yoruba-ASR', token=token); \
        print('✓ Yoruba-ASR cached'); \
    except Exception as e: \
        print(f'⚠ Yoruba-ASR failed: {e}'); \
    try: \
        pipeline('automatic-speech-recognition', model='NCAIR1/Hausa-ASR', token=token); \
        print('✓ Hausa-ASR cached'); \
    except Exception as e: \
        print(f'⚠ Hausa-ASR failed: {e}'); \
    try: \
        pipeline('automatic-speech-recognition', model='NCAIR1/Igbo-ASR', token=token); \
        print('✓ Igbo-ASR cached'); \
    except Exception as e: \
        print(f'⚠ Igbo-ASR failed: {e}');" || echo "Skipping model cache - token not provided"; \
    fi

# Make port 10000 available to the world outside this container
EXPOSE 10000

# Define environment variable
# ENV STREAMLIT_SERVER_PORT 8501
# ENV STREAMLIT_SERVER_HEADLESS true

# Ensure Python can find modules
ENV PYTHONPATH=/app

# Run app.py when the container launches
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--chdir", "/app"]