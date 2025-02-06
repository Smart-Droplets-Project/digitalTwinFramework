import argparse

from sd_data_adapter.client import DAClient
from sd_data_adapter.api import search
from sd_data_adapter.models import AgriFood

from digitaltwin.utils.data_adapter import (
    fill_database_ascab,
    create_device_measurement,
    create_command_message,
    generate_rec_message_id,
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
    print(f"parcels: {parcels}")

    digital_twins = create_digital_twins(parcels)

    # run digital twins
    for digital_twin in digital_twins:
        devices = find_device(digital_twin._isAgriCrop)
        device_dict = {device.controlledProperty: device for device in devices}

        terminated = False
        digital_twin.reset()
        action = 0.0
        while not terminated:
            _, _, terminated, _, _ = digital_twin.step(action)

            info = digital_twin.get_wrapper_attr("info")
            for variable, device in device_dict.items():
                if info[variable] is not None:
                    create_device_measurement(
                        device=device,
                        date_observed=info["Date"][-1].isoformat() + "T00:00:00Z",
                        value=info[variable][-1],
                    )
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
                    waypoints=get_row_coordinates(parcel.location),
                )

        # get output
        summary_output = digital_twin.get_info(to_dataframe=True)
        print(digital_twin)
        print(summary_output)
        print("The following commands were stored\n")
        print(find_command_messages())


if __name__ == "__main__":
    main()
