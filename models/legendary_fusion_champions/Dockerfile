# Legendary Fusion Champion - Docker Deployment
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY *.py ./

EXPOSE 8080
ENV MODEL_PATH=/app/models/LEGENDARY_CHAMPION_ULTIMATE_20250816_032559.onnx
CMD ["python", "legendary_inference_server.py"]
