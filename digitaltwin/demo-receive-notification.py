import os

from digitaltwin.utils.simulator import run_cropmodel

from fastapi import FastAPI, Request
from ngsildclient import SubscriptionBuilder
from sd_data_adapter.client import DAClient

app = FastAPI()

# Host and Port are injected as environment variables
ORION_HOST = os.environ['ORION_HOST']
ORION_PORT = os.environ['ORION_PORT']

client = DAClient.get_instance(host=ORION_HOST, port=ORION_PORT)


# We need a new 
@app.post("/manual-sim")
async def receive_notification(request: Request, debug=False):
    # TODO: @michiel @hilmy
    # This is a new endpoint which can react to manually triggered user simulations
    # We need to define what is needed as input. Maybe just the parcel ID (or crop ID)?
    pass

@app.post("/notification")
async def receive_notification(request: Request, debug=False):
    if debug:
        headers = request.headers
        print(f"Received headers: {headers}")
    try:
        notification = await request.json()
        print("Received notification:")
        print(notification)
        run_cropmodel(debug=False)
        return {"status": "received"}
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"status": "error", "message": str(e)}


def subscribe_to_ocb(host):
    MY_SERVER_URL = f'http://{host}:8000/notification'

    subscr = SubscriptionBuilder(MY_SERVER_URL)\
        .context('https://raw.githubusercontent.com/smart-data-models/dataModel.AgriFood/master/context.jsonld')\
        .description("Notification for new AgriParcelOperations.")\
        .select_type("AgriParcelOperation")\
        .build()

    try:
        client.subscriptions.create(subscr)
        print("Subscription created successfully.")
    except Exception as e:
        print(f"Failed to create subscription: {e}a")


if __name__ == "__main__":
    my_host = os.environ["DIGITAL_TWIN_HOST"]
    subscribe_to_ocb(my_host)

    # TODO: @hilmy @michiel Create a CRON job to run every morning and run the simulation

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

