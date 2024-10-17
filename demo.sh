# fresh start
kubectl delete pod digitaltwin-container
kubectl apply -f start-container.yaml
sleep 10
digital_twin_host=$(kubectl get pod digitaltwin-container -o jsonpath='{.status.podIP}')
orion_host=$(kubectl get service "orion" -o jsonpath='{.spec.clusterIP}')

# run the digital twin; store simulations
kubectl cp digitaltwin/demo.py digitaltwin-container:/app/digitaltwin/demo.py
kubectl exec digitaltwin-container -- poetry run python /app/digitaltwin/demo.py --host ${orion_host}

# set up a server that listens to incoming measurements
kubectl cp digitaltwin/demo-receive-notification.py digitaltwin-container:/app/digitaltwin/demo-receive-notification.py
kubectl cp digitaltwin/demo-upload-measurement.py digitaltwin-container:/app/digitaltwin/demo-upload-measurement.py
kubectl exec digitaltwin-container -- poetry run python /app/digitaltwin/demo-receive-notification.py --webserverhost ${digital_twin_host} --orionhost ${orion_host} &
sleep 5
kubectl exec digitaltwin-container -- poetry run python /app/digitaltwin/demo-upload-measurement.py --orionhost ${orion_host}