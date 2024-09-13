# install minikube
# helm repo add fiware https://fiware.github.io/helm-charts/
# helm install orion fiware/orion
# eval $(minikube -p minikube docker-env)
# docker build -t digitaltwin -f Dockerfile ../

FROM python:3.11-slim

RUN pip install poetry

WORKDIR /app

COPY digitalTwinFramework/pyproject.toml /app/
COPY smartDropletsDataAdapters /app/smartDropletsDataAdapters
COPY digitalTwinFramework/digitaltwin /app/digitaltwin

RUN poetry install --no-root