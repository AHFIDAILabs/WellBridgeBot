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

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg git portaudio19-dev && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Make port 10000 available to the world outside this container
EXPOSE 10000

# Define environment variable
# ENV STREAMLIT_SERVER_PORT 8501
# ENV STREAMLIT_SERVER_HEADLESS true

# Run app.py when the container launches
CMD gunicorn app:app --bind 0.0.0.0:$PORT