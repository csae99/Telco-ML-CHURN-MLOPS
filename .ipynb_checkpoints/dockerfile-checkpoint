FROM python:3.11-slim

WORKDIR /app

COPY requirements-serving.txt .

RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/ ./src/

# Only copy model artifacts (NOT whole repo)
COPY src/serving/model /app/model
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/model /app/model
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/preprocessing.pkl /app/model/preprocessing.pkl
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]