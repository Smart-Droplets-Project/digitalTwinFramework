from datetime import datetime
import argparse
from geojson import MultiPoint

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
    # your code here: get a score for the given parcel area
    return 0.7


def get_locations_of_detections() -> MultiPoint:
    coordinates_detections = [
        (23.9100961, 55.126702),
        (23.9081435, 55.1224445),
    ]
    multi_point = MultiPoint(coordinates_detections)
    return multi_point


def get_locations_of_detections_ascab() -> MultiPoint:
    coordinates_detections = [
        (3.0936112, 42.1625702),
        (3.0945098, 42.1619982),
    ]
    multi_point = MultiPoint(coordinates_detections)
    return multi_point


def get_locations_of_detections_apple_alternaria() -> MultiPoint:
    coordinates_detections = [
        (3.0936112, 42.1623702),
        (3.0945098, 42.1620982),
    ]
    multi_point = MultiPoint(coordinates_detections)
    return multi_point


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
        # Option 1: a single score per parcel
        score = get_detection_score_in_parcel(parcel_area)
        # Option 2: get locations of detections within the given parcel
        locations = get_locations_of_detections()
        crop = get_by_id(parcel.hasAgriCrop["object"])
        pest = [get_by_id(pest_id) for pest_id in crop.hasAgriPest["object"]]
        devices = [find_device(p.id) for p in pest]
        #  Flatten the devices list
        devices = [device for sublist in devices for device in sublist]
        #  the device (with its device measurements) is linked to a pest
        #  the pest is linked to a crop (that is linked to a given parcel)
        device_dict = {device.controlledProperty: device for device in devices}

        for variable, device in device_dict.items():
            # Option 1
            if variable == "obs-detection_score":
                create_device_measurement(
                    device=device,
                    date_observed=datetime.utcnow().isoformat() + "Z",
                    value=score,
                )
            # Option 2
            if variable == "obs-detections":
                print(f"save detections {locations}")
                create_device_measurement(
                    device=device,
                    date_observed=datetime.utcnow().isoformat() + "Z",
                    value=1.0,
                    location=locations,
                )
    print("The following DeviceMeasurements were stored:\n")
    obs_scores = search(
        {
            "type": "DeviceMeasurement",
            "q": 'controlledProperty=="obs-detection_score"',
        },
        ctx=Devices.ctx,
    )
    obs_detections = search(
        {
            "type": "DeviceMeasurement",
            "q": 'controlledProperty=="obs-detections"',
        },
        ctx=Devices.ctx,
    )
    print(obs_scores)
    print(obs_detections)


if __name__ == "__main__":
    main()
