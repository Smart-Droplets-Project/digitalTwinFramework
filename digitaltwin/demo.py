import argparse

from sd_data_adapter.client import DAClient
from sd_data_adapter.api import search
from sd_data_adapter.models import AgriFood

from digitaltwin.cropmodel.crop_model import create_digital_twins
from digitaltwin.cropmodel.recommendation import standard_practice
from digitaltwin.utils.data_adapter import (
    create_device_measurement,
    create_command_message,
    fill_database,
    get_row_coordinates,
    generate_rec_message_id,
    get_recommendation_message,
)
from digitaltwin.utils.database import (
    get_by_id,
    get_demo_parcels,
    has_demodata,
    find_parcel_operations,
    find_device,
    find_crop,
    find_command_messages,
    clear_database,
    get_matching_device,
    get_parcel_operation_by_date,
)


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
                    value=crop_model.get_output()[-1]["LAI"],
                )

            # get AI recommendation
            recommendation = standard_practice(crop_model.get_output()[-1])

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
