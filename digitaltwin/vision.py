from datetime import datetime
import argparse

from sd_data_adapter.client import DAClient
from sd_data_adapter.api import search
from sd_data_adapter.models import AgriFood, Devices

from digitaltwin.utils.database import (
    get_demo_parcels,
    get_by_id,
    find_device,
    has_demodata,
    clear_database,
)

from digitaltwin.utils.data_adapter import (
    create_device_measurement,
    get_coordinates,
    fill_database,
)
from digitaltwin.cropmodel.crop_model import get_default_variables


def get_detection_score_in_parcel(parcel_area):
    # your code here
    return 0.7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="localhost", help="Hostname orion context broker"
    )
    args = parser.parse_args()

    DAClient.get_instance(host=args.host, port=1026)

    clear_database()

    # fill database with demo data
    if not has_demodata():
        fill_database(variables=get_default_variables())

    parcels = search(get_demo_parcels(), ctx=AgriFood.ctx)

    for parcel in parcels:
        parcel_area = get_coordinates(parcel.location, "Polygon")
        score = get_detection_score_in_parcel(parcel_area)
        crop = get_by_id(parcel.hasAgriCrop["object"])
        pest = get_by_id(crop.hasAgriPest["object"])
        devices = find_device(pest.id)
        device_dict = {device.controlledProperty: device for device in devices}

        for variable, device in device_dict.items():
            if variable == "obs-dectection_score":
                create_device_measurement(
                    device=device,
                    date_observed=datetime.utcnow().isoformat() + "Z",
                    value=score,
                )
    print("The following DeviceMeasurements were stored:\n")
    device_measures = search(
        {
            "type": "DeviceMeasurement",
            "q": f'controlledProperty=="obs-dectection_score"',
        },
        ctx=Devices.ctx,
    )

    print(device_measures)


if __name__ == "__main__":
    main()
