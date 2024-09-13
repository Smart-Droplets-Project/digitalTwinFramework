### First time setup
1. Install [Minikube](https://minikube.sigs.k8s.io/docs/)
2. Install the [Orion Context Broker](https://fiware-orion.readthedocs.io/en/master/) with [Helm](https://helm.sh/):
   1. `helm repo add fiware https://fiware.github.io/helm-charts/`
   2. `helm install orion fiware/orion`
3. Create the digitaltwin image and push it to minikube:
   1. `eval $(minikube -p minikube docker-env)`
   2. `docker build -t digitaltwin -f Dockerfile ../`
4. Start a container from the digitaltwin image:
   1. `kubectl apply -f start-container.yaml`
5. Run the demo:
   1. `bash demo.sh`


