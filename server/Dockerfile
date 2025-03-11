FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./server /app

ENV HOST=0.0.0.0
ENV PORT=8000
# ENV DEBUG=True
# ENV MONGO_URI=mongodb://localhost:27017
# ENV DATABASE_NAME=network_checks
# ENV SECRET_KEY=your-super-secret-key-here
# ENV ACCESS_TOKEN_EXPIRE_MINUTES=60

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD sleep 5000