import argparse
from fastapi import FastAPI, Request
from ngsildclient import SubscriptionBuilder
from sd_data_adapter.client import DAClient

parser = argparse.ArgumentParser()
parser.add_argument("--orionhost", type=str, default="localhost", help="Hostname orion")
parser.add_argument("--webserverhost", type=str, default="localhost", help="Hostname webserver")
args = parser.parse_args()

app = FastAPI()

client = DAClient.get_instance(host=args.orionhost, port=1026)


@app.post("/notification")
async def receive_notification(request: Request, debug=False):
    if debug:
        headers = request.headers
        print(f"Received headers: {headers}")
    try:
        notification = await request.json()
        print("Received notification:")
        print(notification)
        return {"status": "received"}
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"status": "error", "message": str(e)}


def subscribe_to_ocb(host):
    MY_SERVER_URL = f'http://{host}:8000/notification'

    subscr = SubscriptionBuilder(MY_SERVER_URL)\
        .context('https://raw.githubusercontent.com/smart-data-models/dataModel.Device/master/context.jsonld')\
        .description("Notification for new Device Measurements.")\
        .select_type("DeviceMeasurement")\
        .build()

    try:
        client.subscriptions.create(subscr)
        print("Subscription created successfully.")
    except Exception as e:
        print(f"Failed to create subscription: {e}a")


if __name__ == "__main__":
    subscribe_to_ocb(args.webserverhost)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

