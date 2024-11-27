import argparse

from digitaltwin.utils.simulator import run_cropmodel
from digitaltwin.utils.data_adapter import fill_database
from digitaltwin.utils.database import has_demodata, clear_database
from digitaltwin.cropmodel.crop_model import get_default_variables

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
        run_cropmodel(debug=False)
        return {"status": "received"}
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"status": "error", "message": str(e)}


def subscribe_to_ocb(host):
    MY_SERVER_URL = f'https://{host}/notification'

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

    # Seed initial data
    clear_database()

    # fill database with demo data
    if not has_demodata():
        fill_database(variables=get_default_variables())

    # Set up subscription
    subscribe_to_ocb(args.webserverhost)

    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

