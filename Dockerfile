FROM python:3.11-slim

RUN apt-get update && apt-get install -y git  # Install git
RUN pip install poetry && \
    poetry config virtualenvs.create false

WORKDIR /app/

COPY digitalTwinFramework/pyproject.toml /app/
COPY digitalTwinFramework/README.md /app/
COPY digitalTwinFramework/digitaltwin /app/digitaltwin

RUN poetry install

# Run python script which starts a Flask server
CMD ["python", "/app/digitaltwin/demo-receive-notification.py"]