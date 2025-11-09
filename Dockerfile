# -------------------------------------------------------------
# FIX: Changed from 'python:3.8-slim-buster' to 
# 'python:3.8-slim-bullseye' to use supported Debian repositories, 
# resolving the "404 Not Found" error during 'apt update'.
# -------------------------------------------------------------
    FROM python:3.8-slim-bullseye

    # Set the working directory inside the container
    WORKDIR /app
    
    # Copy project files
    COPY . .
    
    # -------------------------------------------------------------
    # Install AWS CLI and clean up the package cache in a single 
    # layer to ensure a minimal final image size.
    # '--no-install-recommends' keeps the installation minimal.
    # -------------------------------------------------------------
    RUN apt update \
        && apt install -y --no-install-recommends awscli \
        && rm -rf /var/lib/apt/lists/*
    
    # Install Python dependencies
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Start the application
    CMD ["python3", "app.py"]