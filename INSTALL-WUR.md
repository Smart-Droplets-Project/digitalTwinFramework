## Installation Guide
1. ** Set kubectl context
     ```bash
     kubectl config set-context k8s-wur-test --namespace=training-student16
     ```

2. **Install Orion Context Broker
     ```bash
     kubectl apply -f orion/service.yaml
     kubectl apply -f orion/service.yaml
     kubectl apply -f orion/deploy-mongo.yaml
     kubectl apply -f orion/deployments.yaml
     ```

3. **Deploy the Digital Twin Container:**
   - Start a container from the digital twin image by applying the Kubernetes configuration:  
     ```bash
     kubectl apply -f start-container.yaml
     ```

4. **Run the Demo:**
   - Execute the demo script:  
     ```bash
     demo.sh
     ```

