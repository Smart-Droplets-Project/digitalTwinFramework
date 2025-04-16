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
    fill_database_ascab,
    map_pest_detections_to_device_id,
    map_pest_detections_to_parcel,
    check_points_in_parcel,
    get_parcel_geometry
)
from digitaltwin.cropmodel.crop_model import get_default_variables


def get_detection_score_in_parcel(parcel_area):
    # your code here: get a score for the given parcel area
    return 0.7


def get_locations_of_detections() -> MultiPoint:
    coordinates_detections = [
        (23.5824108, 55.7546751),
        (23.5812092, 55.7529604),
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
        (3.0956322, 42.1615223),
        (3.0935098, 42.1622142),
    ]
    multi_point = MultiPoint(coordinates_detections)
    return multi_point


def get_demo_pest_location() -> dict[str, MultiPoint]:
    return {
        'alternaria': get_locations_of_detections_apple_alternaria(),
        'ascab': get_locations_of_detections_ascab()
    }


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
        # fill_database(variables=get_default_variables())
        fill_database_ascab()

    parcels = search(get_demo_parcels("Serrater"), ctx=AgriFood.ctx)

    for parcel in parcels:
        # parcel_area = get_coordinates(parcel.location, "Polygon")
        parcel_geometry = get_parcel_geometry(parcel.location)
        # Option 1: a single score per parcel
        score = get_detection_score_in_parcel(parcel_geometry)
        # Option 2: get locations of detections within the given parcel
        # locations = get_locations_of_detections()
        pest_detections = get_demo_pest_location()

        crop = get_by_id(parcel.hasAgriCrop["object"])
        pests = [get_by_id(pest_id) for pest_id in crop.hasAgriPest["object"]]
        devices = [find_device(p.id) for p in pests]
        #  Flatten the devices list
        devices = [device for sublist in devices for device in sublist]
        #  Maps detection locations to pest
        pest_map = map_pest_detections_to_device_id(pests, pest_detections)  # uncomment for id method
        #  the device (with its device measurements) is linked to a pest
        #  the pest is linked to a crop (that is linked to a given parcel)
        device_dict = {device.controlledProperty: device for device in devices}

        for variable, device in device_dict.items():
            # Option 1
            if variable == "obs-detection_score":
                score_object = create_device_measurement(
                    device=device,
                    date_observed=datetime.utcnow().isoformat() + "Z",
                    value=score,
                    # location=pest_map[device.controlledAsset],  # can use either this option `detections->device->pest id`
                    location=map_pest_detections_to_parcel(  # or this option `detections->parcel area->device->pest id`
                        parcel_geometry,
                        pests,
                        device,
                        pest_detections
                    )
                )
            # Option 2
            if variable == "obs-detections":
                print(f"save detections {pest_detections}")
                detection_object = create_device_measurement(
                    device=device,
                    date_observed=datetime.utcnow().isoformat() + "Z",
                    value=1.0,
                    # location=pest_map[device.controlledAsset],
                    location=map_pest_detections_to_parcel(
                        parcel_geometry,
                        pests,
                        device,
                        pest_detections
                    )
                )
                print(detection_object)
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
