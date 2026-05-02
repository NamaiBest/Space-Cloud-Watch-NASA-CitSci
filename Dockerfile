FROM python:3.11-slim

# System dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install CPU-only PyTorch (keeps image lean — no CUDA overhead)
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Hugging Face Spaces requires port 7860
ENV PORT=7860
ENV HOST=0.0.0.0

EXPOSE 7860

CMD ["python", "hf_app.py"]
