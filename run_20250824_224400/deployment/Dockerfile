FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install \
    ultralytics \
    opencv-python \
    numpy \
    pillow

# Copy model and inference code
COPY champion_model.pth /app/
COPY inference_server.py /app/
COPY deployment_config.json /app/

WORKDIR /app

# Expose port for inference server
EXPOSE 8000

# Run inference server
CMD ["python", "inference_server.py"]
