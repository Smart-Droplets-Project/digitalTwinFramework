import argparse

from sd_data_adapter.client import DAClient
from sd_data_adapter.api import search, upsert
from sd_data_adapter.models import AgriFood
import sd_data_adapter.models.device as device_model

from digitaltwin.cropmodel.crop_model import create_digital_twins, get_default_variables
from digitaltwin.cropmodel.recommendation import standard_practice
from digitaltwin.utils.data_adapter import (
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
    find_device_measurement,
    find_command_messages,
    clear_database,
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
        fill_database(variables=get_default_variables())

    # get parcels from database
    parcels = search(get_demo_parcels(), ctx=AgriFood.ctx)
    print(f"parcels: {parcels}")

    # create digital twins
    digital_twins = create_digital_twins(parcels)

    # run digital twins
    for digital_twin in digital_twins:
        parcel_operations = find_parcel_operations(digital_twin._locatedAtParcel)
        devices = find_device(digital_twin._isAgriCrop)
        device_dict = {device.controlledProperty: (device, device_model.DeviceMeasurement(refDevice=device.id, controlledProperty=device.controlledProperty)) for device in devices}

        # run crop model
        while digital_twin.flag_terminate is False:
            parcel_operation = get_parcel_operation_by_date(
                parcel_operations, digital_twin.day
            )
            action = parcel_operation.quantity if parcel_operation else 0
            digital_twin.run(1, action)

            for variable, (device, device_measurement) in device_dict.items():
                if digital_twin.get_output()[-1][variable] is not None:
                    device_measurement.numValue = digital_twin.get_output()[-1][variable]
                    device_measurement.dateObserved = digital_twin.day.isoformat() + "T00:00:00Z"
                    upsert(device_measurement)

            # get AI recommendation
            recommendation = standard_practice(digital_twin.get_output()[-1])

            # create command message
            if recommendation > 0:
                recommendation_message = get_recommendation_message(
                    type="fertilize",
                    amount=recommendation,
                    day=digital_twin.day.isoformat(),
                    parcel_id=digital_twin._locatedAtParcel,
                )

                command_message_id = generate_rec_message_id(
                    day=digital_twin.day.isoformat(),
                    parcel_id=digital_twin._locatedAtParcel,
                )
                parcel = get_by_id(digital_twin._locatedAtParcel, ctx=AgriFood.ctx)
                command = create_command_message(
                    message_id=command_message_id,
                    command=recommendation_message,
                    command_time=digital_twin.day.isoformat(),
                    waypoints=get_row_coordinates(parcel.location),
                )

        print(digital_twin.get_summary_output())
        print("The following commands were stored:\n")
        print(find_command_messages())
        print("The following DeviceMeasurements were stored:\n")
        print(find_device_measurement(digital_twin._isAgriCrop))


if __name__ == "__main__":
    main()
