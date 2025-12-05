# Dockerfile
FROM python:3.11-slim

# set workdir
WORKDIR /app

# copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy entire repo
COPY . .

# expose port
EXPOSE 8000

# default command: run uvicorn on the FastAPI app
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
