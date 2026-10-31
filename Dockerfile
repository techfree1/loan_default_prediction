FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y nltk || true \
    && apt-get purge -y --auto-remove build-essential binutils \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN chmod +x deploy/entrypoint.sh run_pipeline.py

EXPOSE 8000 8501

ENTRYPOINT ["deploy/entrypoint.sh"]
CMD ["all"]
