FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8001
ENV SERVER_URL=http://localhost:8000
ENV MONGO_URI=mongodb://localhost:27017
ENV DATABASE_NAME=network_checks
ENV CLIENT_PORT=8001
ENV DEFAULT_TIMEOUT=1.0
ENV DEFAULT_COUNT=10
ENV LOG_LEVEL=INFO

CMD ["uvicorn", "client.main:app", "--host", "0.0.0.0", "--port", "8001"]
