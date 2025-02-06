import argparse
import datetime

from sd_data_adapter.client import DAClient
from sd_data_adapter.api import search, upsert
from sd_data_adapter.models import AgriFood
import sd_data_adapter.models.device as device_model

from digitaltwin.utils.data_adapter import (
    fill_database_ascab,
    create_device_measurement,
    create_command_message,
    generate_rec_message_id,
    create_geojson_from_feature_collection,
    get_recommendation_message,
)
from digitaltwin.utils.database import (
    clear_database,
    get_demo_parcels,
    find_device,
    get_by_id,
    find_command_messages,
)
from digitaltwin.ascabmodel.ascab_model import create_digital_twins
from digitaltwin.ascabmodel.recommendation import minimize_risk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="localhost", help="Hostname orion context broker"
    )
    args = parser.parse_args()

    DAClient.get_instance(host=args.host, port=1026)

    clear_database()
    fill_database_ascab()

    parcels = search(get_demo_parcels("Serrater"), ctx=AgriFood.ctx)
    digital_twins = create_digital_twins(parcels)

    # run digital twins
    for digital_twin in digital_twins:
        devices = find_device(digital_twin._isAgriCrop)

        sim_dict = {
            device.controlledProperty: (
                device,
                device_model.DeviceMeasurement(
                    refDevice=device.id,
                    controlledProperty=device.controlledProperty,
                    dateCreated=datetime.datetime.now().isoformat() + "T00:00:00Z",
                    dataProvider="digital-twin-simulator",
                ),
            )
            for device in devices
            if device.controlledProperty.startswith("sim-")
        }

        terminated = False
        digital_twin.reset()
        action = 0.0
        while not terminated:
            _, _, terminated, _, _ = digital_twin.step(action)

            info = digital_twin.get_wrapper_attr("info")
            for variable, (device, device_measurement) in sim_dict.items():
                stripped_variable = variable.split("-", 1)[1]
                if info[stripped_variable] is not None:
                    device_measurement.numValue = info[stripped_variable][-1]
                    device_measurement.dateObserved = (
                        info["Date"][-1].isoformat() + "T00:00:00Z"
                    )
                    upsert(device_measurement)
            recommendation = minimize_risk(info)

            # create command message
            parcel = get_by_id(digital_twin._locatedAtParcel, ctx=AgriFood.ctx)
            if recommendation > 0:
                recommendation_message = get_recommendation_message(
                    type="fungicide",
                    amount=recommendation,
                    day=info["Date"][-1].isoformat(),
                    parcel_id=digital_twin._locatedAtParcel,
                )

                command_message_id = generate_rec_message_id(
                    day=info["Date"][-1].isoformat(),
                    parcel_id=digital_twin._locatedAtParcel,
                )

                command = create_command_message(
                    message_id=command_message_id,
                    command=recommendation_message,
                    command_time=info["Date"][-1].isoformat(),
                    waypoints=create_geojson_from_feature_collection(
                        parcel.location, target_rate_value=recommendation
                    ),
                )

        # get output
        summary_output = digital_twin.get_info(to_dataframe=True)
        print(digital_twin)
        print(summary_output)
        print("The following commands were stored\n")
        print(find_command_messages())


if __name__ == "__main__":
    main()
