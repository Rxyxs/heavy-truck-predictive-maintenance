# API de riesgo de SCANIA Component X con el modelo publicado en el release (no hace falta entrenar).
#   docker build -t truck-risk .
#   docker run -p 8000:8000 -e RISK_API_KEY=change-me truck-risk
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY examples ./examples
ARG MODEL_TAG=v1.0.0
RUN python -c "from src.demo import fetch_model; fetch_model('${MODEL_TAG}')"

ENV RISK_API_KEY=change-me
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
