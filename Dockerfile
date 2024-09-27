FROM python:3.11-slim

RUN pip install poetry

WORKDIR /app

COPY digitalTwinFramework/pyproject.toml /app/
COPY smartDropletsDataAdapters /app/smartDropletsDataAdapters
COPY digitalTwinFramework/digitaltwin /app/digitaltwin

RUN poetry install