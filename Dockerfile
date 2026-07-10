FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY papers/ ./papers/
COPY scripts/ ./scripts/

EXPOSE 7860

CMD ["sh", "-c", "python scripts/preload_papers.py && uvicorn src.api:app --host 0.0.0.0 --port 7860"]