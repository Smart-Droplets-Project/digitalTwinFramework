orion_host=$(kubectl get service "orion" -o jsonpath='{.spec.clusterIP}')
kubectl cp digitaltwin/demo.py digitaltwin-container:/app/digitaltwin/demo.py
kubectl exec digitaltwin-container -- bash -c "PYTHONPATH=/app poetry run python /app/digitaltwin/demo.py --host ${orion_host}"
