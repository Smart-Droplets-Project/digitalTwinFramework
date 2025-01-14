from sd_data_adapter.api import search, upsert
from sd_data_adapter.models import AgriFood
import sd_data_adapter.models.device as device_model

from digitaltwin.cropmodel.crop_model import (
    create_digital_twins,
    create_cropgym_agents,
    get_dummy_measurements,
)
from digitaltwin.cropmodel.recommendation import fill_it_up
from digitaltwin.utils.data_adapter import (
    create_command_message,
    generate_rec_message_id,
    get_recommendation_message,
    create_geojson_from_feature_collection,
    split_device_dicts,
)
from digitaltwin.utils.database import (
    get_by_id,
    get_demo_parcels,
    find_parcel_operations,
    find_device,
    find_device_measurement,
    find_command_messages,
    get_parcel_operation_by_date,
)

from digitaltwin.cropmodel.crop_model import (
    get_default_variables,
    get_titles,
    calibrate,
)
import matplotlib.pyplot as plt
import pandas as pd
import datetime


def run_cropmodel(calibrate_flag=True, debug=False):
    parcels = search(get_demo_parcels(), ctx=AgriFood.ctx)
    digital_twins = create_digital_twins(parcels)
    cropgym_agents = create_cropgym_agents(parcels, digital_twins)
    recommendation = 0

    # run digital twins
    for digital_twin, cropgym_agent in zip(digital_twins, cropgym_agents):
        parcel_operations = find_parcel_operations(digital_twin._locatedAtParcel)
        devices = find_device(digital_twin._isAgriCrop)

        sim_dict = {
            device.controlledProperty: (
                device,
                device_model.DeviceMeasurement(
                    refDevice=device.id, controlledProperty=device.controlledProperty
                ),
            )
            for device in devices
            if device.controlledProperty.startswith("sim-")
        }

        device_measurements = find_device_measurement()
        dvs_measurements = [
            measurement
            for measurement in device_measurements
            if measurement.controlledProperty == "obs-DVS"
        ]

        lai_measurements = [
            measurement
            for measurement in device_measurements
            if measurement.controlledProperty == "obs-LAI"
        ]

        if calibrate_flag:

            data_dvs = [
                [dvs_measurement.dateObserved, dvs_measurement.numValue]
                for dvs_measurement in dvs_measurements
            ]
            data_lai = [
                [lai_measurement.dateObserved, lai_measurement.numValue]
                for lai_measurement in lai_measurements
            ]
            df_dvs = pd.DataFrame(data_dvs, columns=["day", "DVS"])
            df_lai = pd.DataFrame(data_lai, columns=["day", "LAI"])
            df_assimilate = pd.merge(df_dvs, df_lai, on="day", how="outer")
            df_assimilate["day"] = pd.to_datetime(df_assimilate["day"])
            df_assimilate = df_assimilate.set_index("day")
            calibrate(digital_twin, df_assimilate)

        # run crop model
        while digital_twin.flag_terminate is False:
            parcel_operation = get_parcel_operation_by_date(
                parcel_operations, digital_twin.day
            )
            action = parcel_operation.quantity if parcel_operation else 0
            # if digital_twin.get_output()[-1]["day"].strftime("%Y-%m-%d") == "2023-03-10" and action == 0:
            #    action = 60
            action = action + recommendation
            if debug:
                print(digital_twin.get_output()[-1]["day"])
            digital_twin.run(1, action)
            for variable, (device, device_measurement) in sim_dict.items():
                stripped_variable = variable.split("-", 1)[1]
                if digital_twin.get_output()[-1][stripped_variable] is not None:
                    device_measurement.numValue = digital_twin.get_output()[-1][
                        stripped_variable
                    ]
                    device_measurement.dateObserved = (
                        digital_twin.day.isoformat() + "T00:00:00Z"
                    )
                    upsert(device_measurement)
            # get AI recommendation
            if not cropgym_agent:
                recommendation = fill_it_up(digital_twin.get_output()[-1])
            else:
                recommendation = cropgym_agent(digital_twin.get_output())

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
                    waypoints=create_geojson_from_feature_collection(
                        parcel.location, target_rate_value=recommendation
                    ),
                )

        print(digital_twin.get_summary_output())
        print("The following commands were stored:\n")
        print(find_command_messages())

        print("The following DeviceMeasurements were stored:\n")
        print(find_device_measurement())

        if debug:
            model_output = pd.DataFrame(digital_twin.get_output())
            model_output = model_output.set_index("day")

            plot_variables = get_default_variables()
            fig, axes = plt.subplots(
                len(plot_variables), 1, sharex=True, figsize=(12, 10)
            )
            titles = get_titles()

            for i, var in enumerate(plot_variables):
                ax = axes if len(plot_variables) == 1 else axes[i]
                ax.plot_date(
                    model_output.index, model_output[var], "r-", label="AI agent"
                )
                if calibrate_flag and var == "DVS":
                    ax.plot_date(df_assimilate.index, df_assimilate.DVS, label="DVS Observation")
                if calibrate_flag and var == "LAI":
                    ax.plot_date(df_assimilate.index, df_assimilate.LAI, label="LAI Observation")
                name, unit = titles[var]
                title = f"{var} - {name}"
                ax.set_ylabel(f"[{unit}]")
                ax.set_title(title, fontsize="8.5")
                if i == 0:
                    ax.legend(fontsize="6.5")
                ax.grid()
                ax.set_xlim(
                    [
                        model_output.index[5],
                        model_output.index[-1] + datetime.timedelta(days=7),
                    ]
                )
            fig.autofmt_xdate()
            plt.show()
