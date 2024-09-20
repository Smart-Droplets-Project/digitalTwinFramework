## Installation Guide

1. **Install Minikube:**
   - Follow the instructions on the [Minikube documentation](https://minikube.sigs.k8s.io/docs/) to install and start Minikube on your system.

2. **Install Orion Context Broker using Helm:**
   - Add the FIWARE Helm repository:  
     ```bash
     helm repo add fiware https://fiware.github.io/helm-charts/
     ```
   - Install the Orion Context Broker:  
     ```bash
     helm install orion fiware/orion
     ```

3. **Build and Deploy the Digital Twin Image:**
   - Configure your shell to use the Minikube Docker environment:  
     ```bash
     eval $(minikube -p minikube docker-env)
     ```
   - Build the digital twin Docker image:  
     ```bash
     docker build -t digitaltwin -f Dockerfile ../
     ```
     > **Note:** Ensure that the `smartDropletsDataAdapters` repository is located in the parent directory (`../`).

4. **Deploy the Digital Twin Container:**
   - Start a container from the digital twin image by applying the Kubernetes configuration:  
     ```bash
     kubectl apply -f start-container.yaml
     ```

5. **Run the Demo:**
   - Execute the demo script:  
     ```bash
     bash demo.sh
     ```

