import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sd_data_adapter.api import search, upsert
from sd_data_adapter.models import AgriFood
import sd_data_adapter.models.device as device_model
from sd_data_adapter.models.agri_food.agriParcel import AgriParcel

from digitaltwin.cropmodel.crop_model import (
    create_digital_twins,
    create_cropgym_agents,
    get_weather_provider,
)
from digitaltwin.cropmodel.recommendation import fill_it_up
from digitaltwin.utils.data_adapter import (
    create_command_message,
    generate_rec_message_id,
    get_recommendation_message,
    create_geojson_from_feature_collection,
    create_fertilizer_application,
)
from digitaltwin.utils.database import (
    get_by_id,
    get_demo_parcels,
    find_parcel_operations,
    find_device,
    find_device_measurement,
    find_agriproducttype,
    find_command_messages,
    get_parcel_operation_by_date,
)
from digitaltwin.utils.helpers import get_weather, get_simulated_days

from digitaltwin.cropmodel.crop_model import (
    get_default_variables,
    get_titles,
    calibrate,
)


def run_cropmodel(
    parcels: list[AgriParcel] = None,
    calibrate_flag=True,
    debug=False,
    end_date=None,
    use_cropgym_agent=True,
):
    if parcels is None:
        parcels = search(get_demo_parcels(), ctx=AgriFood.ctx)
    digital_twins = create_digital_twins(parcels)
    cropgym_agents = create_cropgym_agents(parcels, digital_twins)
    fertilizer_object = find_agriproducttype(name="Nitrogen")[0]

    # run digital twins
    for digital_twin, cropgym_agent in zip(digital_twins, cropgym_agents):
        devices = find_device(digital_twin._isAgriCrop)
        parcel = get_by_id(digital_twin._locatedAtParcel, ctx=AgriFood.ctx)
        weather_provider = get_weather_provider(parcel)

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
            min_date = df_assimilate.index.min().date()
            if end_date and end_date > min_date:
                calibrate(digital_twin, df_assimilate, end_date=end_date)

        # run crop model
        while digital_twin.flag_terminate is False:
            if end_date is not None and digital_twin.day >= end_date:
                digital_twin.flag_terminate = True
            parcel_operations = find_parcel_operations(digital_twin._locatedAtParcel)
            parcel_operation = get_parcel_operation_by_date(
                parcel_operations, digital_twin.day
            )
            action = parcel_operation.quantity if parcel_operation else 0
            store_simulations = digital_twin.day == end_date
            ask_recommendation = digital_twin.day == end_date

            digital_twin.run(1, action)

            dates = get_simulated_days(digital_twin.get_output())

            # store simulations
            if store_simulations:
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
            if ask_recommendation:
                if use_cropgym_agent is False or not cropgym_agent:
                    recommendation = fill_it_up(digital_twin.get_output()[-1])
                else:
                    week_weather = get_weather(dates, weather_provider)
                    recommendation = cropgym_agent(
                        digital_twin.get_output(),
                        week_weather,
                    )

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
                    command = create_command_message(
                        message_id=command_message_id,
                        command=recommendation_message,
                        command_time=digital_twin.day.isoformat(),
                        waypoints=create_geojson_from_feature_collection(
                            parcel.location, target_rate_value=recommendation
                        ),
                    )
                    operation = create_fertilizer_application(
                        parcel=parcel,
                        product=fertilizer_object,
                        quantity=recommendation,
                        date=digital_twin.day.strftime("%Y%m%d"),
                        operationtype="sim-fertilizer",
                    )

        if debug:
            print(digital_twin.get_summary_output())
            print("The following commands were stored:\n")
            print(find_command_messages())

            # print("The following DeviceMeasurements were stored:\n")
            # print(find_device_measurement())

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
                    ax.plot_date(
                        df_assimilate.index, df_assimilate.DVS, label="DVS Observation"
                    )
                if calibrate_flag and var == "LAI":
                    ax.plot_date(
                        df_assimilate.index, df_assimilate.LAI, label="LAI Observation"
                    )
                name, unit = titles[var]
                title = f"{var} - {name}"
                ax.set_ylabel(f"[{unit}]")
                ax.set_title(title, fontsize="8.5")
                if i == 0:
                    ax.legend(fontsize="6.5")
                ax.grid()
                first_valid_index = model_output.dropna(how="any").first_valid_index()
                ax.set_xlim(
                    [
                        first_valid_index,
                        model_output.index[-1] + datetime.timedelta(days=7),
                    ]
                )
            fig.autofmt_xdate()
            plt.show()
