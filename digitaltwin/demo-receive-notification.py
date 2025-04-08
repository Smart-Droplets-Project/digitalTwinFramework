import os
import datetime
import uvicorn
import json
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Request
from ngsildclient import SubscriptionBuilder

from sd_data_adapter.client import DAClient
from sd_data_adapter.api import search
from sd_data_adapter.models import AgriFood
from digitaltwin.utils.simulator import run_cropmodel
from digitaltwin.utils.data_adapter import fill_database
from digitaltwin.utils.database import clear_database


app = FastAPI()

# Host and Port are injected as environment variables
ORION_HOST = os.environ["ORION_HOST"]
ORION_PORT = os.environ["ORION_PORT"]

client = DAClient.get_instance(host=ORION_HOST, port=ORION_PORT)


def daily_run():
    current_day = datetime.date.today()
    run_cropmodel(end_date=current_day, debug=True)


# Set up APScheduler to run daily at midnight
scheduler = BackgroundScheduler()
scheduler.add_job(daily_run, "cron", hour=0, minute=0)


def get_crop_id_from_request(request: Request):
    # for now: just pick crop_id from database
    crops = search({"type": "AgriCrop", "q": f'description=="wheat"'}, ctx=AgriFood.ctx)
    result = crops[0].id
    return result


def get_parcel_id_from_request(request: Request):
    crop_id = get_crop_id_from_request(request)
    crop_id_str = json.dumps(crop_id)
    parcels = search(
        {"type": "AgriParcel", "q": f"hasAgriCrop=={crop_id_str}"}, ctx=AgriFood.ctx
    )
    return parcels


@app.post("/manual-sim")
async def receive_notification(request: Request, debug=False):
    if debug:
        headers = request.headers
        print(f"Received headers: {headers}")
    try:
        parcels = get_parcel_id_from_request(request)
        run_cropmodel(parcels=parcels, debug=False)
        return {"status": "received"}
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/notification")
async def receive_notification(request: Request, debug=False):
    if debug:
        headers = request.headers
        print(f"Received headers: {headers}")
    try:
        run_cropmodel(debug=False)
        return {"status": "received"}
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"status": "error", "message": str(e)}


def subscribe_to_ocb(host):
    MY_SERVER_URL = f"http://{host}:8000/notification"

    subscr = (
        SubscriptionBuilder(MY_SERVER_URL)
        .context(
            "https://raw.githubusercontent.com/smart-data-models/dataModel.AgriFood/master/context.jsonld"
        )
        .description("Notification for new AgriParcelOperations.")
        .select_type("AgriParcelOperation")
        .build()
    )

    try:
        client.subscriptions.create(subscr)
        print("Subscription created successfully.")
    except Exception as e:
        print(f"Failed to create subscription: {e}a")


if __name__ == "__main__":
    my_host = os.environ["DIGITAL_TWIN_HOST"]
    subscribe_to_ocb(my_host)

    clear_database()
    fill_database()

    # start a simulation when the script starts
    daily_run()

    # Start the scheduler to run daily crop simulations
    scheduler.start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
