
# Installation Guide

## 1. Set kubectl context

```bash
kubectl config set-context k8s-wur-test --namespace=training-student16
```

## 2. Install Orion Context Broker

```bash
kubectl apply -f orion/service.yaml
kubectl apply -f orion/deploy-mongo.yaml
kubectl apply -f orion/deployments.yaml
kubectl apply -f orion/ingress.yaml
```

Note: These files were obtained via:

```bash
helm repo add fiware https://fiware.github.io/helm-charts/
helm pull fiware/orion --untar
helm install --dry-run --debug orion ./orion
```

Some modifications:
- Enable resources
- Remove `livenessProbe` and `readinessProbe`

## 3. Deploy the Digital Twin Container

Start a container from the digital twin image by applying the Kubernetes configuration:

```bash
kubectl apply -f start-container.yaml
```

## 4. Run the Demo

Execute the demo script:

```bash
./run-simulation.sh
```

## 5. Fetch some data

Retrieve data from the Smart Droplets endpoint:

```bash
curl "https://smart-droplets.containers-test.wur.nl/v2/entities"
```

## Build the Digital Twin Image and Upload to WUR Container Platform

To build the digital twin image and upload it to the WUR container platform:

```bash
docker login https://harbor.containers.wurnet.nl # paste CLI secret from https://harbor.containers.wurnet.nl
docker build -t harbor.containers.wurnet.nl/training/digitaltwin -f Dockerfile ../ --push
```
