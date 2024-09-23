import datetime
import argparse

from sd_data_adapter.client import DAClient
from sd_data_adapter.api import search
from sd_data_adapter.models import AgriFood

from digitaltwin.cropmodel.crop_model import create_digital_twins
from digitaltwin.services.data_adapter import (
    create_device_measurement,
    create_command_message,
    fill_database,
    get_row_coordinates,
)
from digitaltwin.services.database import (
    get_by_id,
    get_demo_parcels,
    has_demodata,
    find_parcel_operations,
    find_device,
    find_crop,
    find_command_messages,
    clear_database,
)


def placeholder_recommendation(crop_model_output: dict):
    year = crop_model_output["day"].year
    fertilization_dates = [datetime.date(year, 4, 1), datetime.date(year, 5, 1)]
    result = 0
    if crop_model_output["day"] in fertilization_dates:
        result = 60
    return result


def get_recommendation_message(recommendation: float, day: str, parcel_id: str):
    return f"rec-fertilize:{recommendation}:day:{day}:parcel_id:{parcel_id}"


def generate_rec_message_id(day: str, parcel_id: str):
    return f"urn:ngsi-ld:CommandMessage:rec-{day}-'{parcel_id}'"


def get_parcel_operation_by_date(parcel_operations, target_date):
    matching_operations = list(
        filter(
            lambda op: datetime.datetime.strptime(op.plannedStartAt, "%Y%m%d").date()
            == target_date,
            parcel_operations,
        )
    )
    return matching_operations[0] if matching_operations else None


def get_matching_device(devices, variable: str):
    matching_devices = list(
        filter(
            lambda op: op.controlledProperty == variable,
            devices,
        )
    )
    return matching_devices[0] if matching_devices else None


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
        fill_database()

    # get parcels from database
    parcels = search(get_demo_parcels(), ctx=AgriFood.ctx)
    print(f"parcels: {parcels}")

    # create digital twins
    digital_twin_dicts = create_digital_twins(parcels)

    # run digital twins
    for parcel_id, crop_model in digital_twin_dicts.items():
        parcel_operations = find_parcel_operations(parcel_id)
        devices = find_device(find_crop(parcel_id))
        lai_device = get_matching_device(devices, "LAI")

        # run crop model
        while crop_model.flag_terminate is False:
            parcel_operation = get_parcel_operation_by_date(
                parcel_operations, crop_model.day
            )
            action = parcel_operation.quantity if parcel_operation else 0
            crop_model.run(1, action)

            if crop_model.get_output()[-1]["LAI"] is not None:
                create_device_measurement(
                    device=lai_device,
                    date_observed=crop_model.day.isoformat() + "T00:00:00Z",
                    value=0.3,
                )

            # get AI recommendation
            recommendation = placeholder_recommendation(crop_model.get_output()[-1])

            # create command message
            if recommendation > 0:
                recommendation_message = get_recommendation_message(
                    recommendation=recommendation,
                    day=crop_model.day.isoformat(),
                    parcel_id=parcel_id,
                )

                command_message_id = generate_rec_message_id(
                    day=crop_model.day.isoformat(), parcel_id=parcel_id
                )
                parcel = get_by_id(parcel_id, ctx=AgriFood.ctx)
                command = create_command_message(
                    message_id=command_message_id,
                    command=recommendation_message,
                    command_time=crop_model.day.isoformat(),
                    waypoints=get_row_coordinates(parcel.location),
                )

        print(crop_model.get_summary_output())
        print("The following commands were stored\n")
        print(find_command_messages())


if __name__ == "__main__":
    main()
