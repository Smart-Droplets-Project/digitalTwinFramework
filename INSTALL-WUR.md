
# Installation Guide

## 1. Set kubectl context

```bash
kubectl config use-context k8s-wur-test --namespace=training-student16
```

## 2. Install Orion Context Broker

```bash
kubectl apply -f orion/service.yaml
kubectl apply -f orion/service-mongo.yaml
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

With modifications:
- Enable resources

## 3. [Optional] Build the Digital Twin Image and Upload to WUR Container Platform

```bash
docker login https://harbor.containers.wurnet.nl # paste CLI secret from https://harbor.containers.wurnet.nl
docker build -t harbor.containers.wurnet.nl/training/digitaltwin -f Dockerfile ../ --push
```

## 4. Deploy the Digital Twin Container

```bash
kubectl apply -f start-container-wur.yaml
```

## 5. Run simulations

```bash
./run-simulation.sh
```

## 6. Fetch some data

```bash
curl "https://smart-droplets.containers-test.wur.nl/v2/entities"
```