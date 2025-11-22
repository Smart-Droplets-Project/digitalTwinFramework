import matplotlib.pyplot as plt
import pandas as pd
import datetime
import tempfile
import os

from sd_data_adapter.api import search, upsert
from sd_data_adapter.models import AgriFood, Devices
import sd_data_adapter.models.device as device_model
from sd_data_adapter.models.agri_food.agriParcel import AgriParcel
from ascab.utils.plot import plot_results

from digitaltwin.ascabmodel.ascab_model import (
    create_digital_twins as create_digital_twins_ascab,
)
from digitaltwin.ascabmodel.recommendation import minimize_risk

from digitaltwin.cropmodel.crop_model import (
    create_digital_twins,
    create_cropgym_agents,
    get_weather_provider,
    get_default_calibration_parameters,
    CropModel,
)
from digitaltwin.cropmodel.recommendation import fill_it_up
from digitaltwin.utils.data_adapter import (
    create_command_message,
    generate_rec_message_id,
    get_recommendation_message,
    create_geojson_from_feature_collection,
    create_agriparcel_operation,
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
    get_matching_device,
)
from digitaltwin.utils.helpers import get_weather, get_simulated_days

from digitaltwin.cropmodel.crop_model import (
    get_default_variables,
    get_titles,
    calibrate,
)


def get_sim_dict(devices: list[Devices]):
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
    return sim_dict


def run_cropmodel(
    parcels: list[AgriParcel] = None,
    calibrate_flag=True,
    debug=False,
    end_date: datetime.date = None,
    use_cropgym_agent=True,
) -> tuple[list[CropModel], list[pd.DataFrame]]:
    print("run cropmodel")
    if parcels is None:
        parcels = search(get_demo_parcels(), ctx=AgriFood.ctx)
    digital_twins = create_digital_twins(parcels)
    cropgym_agents = create_cropgym_agents(parcels, digital_twins)
    fertilizer_object = find_agriproducttype(name="Nitrogen")[0]

    df_assimilates: list[pd.DataFrame] = []
    # run digital twins
    for digital_twin, cropgym_agent in zip(digital_twins, cropgym_agents):
        devices = find_device(digital_twin._isAgriCrop)
        parcel = get_by_id(digital_twin._locatedAtParcel, ctx=AgriFood.ctx)
        weather_provider = get_weather_provider(parcel)

        sim_dict = get_sim_dict(devices)

        device_obs_dvs = get_matching_device(devices, "obs-DVS")
        dvs_measurements = find_device_measurement(
            controlled_property="obs-DVS", ref_device=device_obs_dvs.id
        )
        device_obs_lai = get_matching_device(devices, "obs-LAI")
        lai_measurements = find_device_measurement(
            controlled_property="obs-LAI", ref_device=device_obs_lai.id
        )

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
            df_assimilates.append(df_assimilate)
            min_date = df_assimilate.index.min().date()
            if end_date is None or end_date and end_date > min_date:
                if True:
                    print(f"calibrate")
                    calibration_parameters = get_default_calibration_parameters()
                    flowered = df_assimilate[df_assimilate["DVS"] >= 1.2].index.min()
                    if flowered <= pd.Timestamp(end_date):
                        calibration_parameters = ["TSUM1", "TSUM2", "TDWI", "SPAN"]
                    calibrate(
                        digital_twin,
                        df_assimilate,
                        parameters=calibration_parameters,
                        end_date=end_date,
                    )
        else:
            pass
            # digital_twin.parameterprovider.set_override("TSUM1", 892.5)
            # digital_twin.parameterprovider.set_override("TDWI", 537.41)
            # digital_twin.parameterprovider.set_override("SPAN", 28.5)

        # run crop model
        while digital_twin.flag_terminate is False:
            if end_date is not None and digital_twin.day >= end_date:
                digital_twin.flag_terminate = True
            parcel_operations = find_parcel_operations(digital_twin._locatedAtParcel)
            parcel_operation = get_parcel_operation_by_date(
                parcel_operations, digital_twin.day
            )
            action = parcel_operation.quantity if parcel_operation else 0
            if use_cropgym_agent and cropgym_agent and action > 0:
                cropgym_agent.update_action(action * 0.10)
            digital_twin.run(1, action)
            ask_recommendation = (
                digital_twin.day == end_date if end_date is not None else True
            )
            ask_recommendation = False
            dates = get_simulated_days(digital_twin.get_output())

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
                    recommendation = 10.0 * cropgym_agent(
                        digital_twin.get_output(),
                        week_weather,
                    )

                # create command message
                if recommendation > 0:
                    print(f"{digital_twin.day}: {recommendation}")
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
                    operation = create_agriparcel_operation(
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
                        df_assimilate.index, df_assimilate.DVS, label="Observations"
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
            today = datetime.datetime.today().strftime("%Y-%m-%d")
            filename = os.path.join(tempfile.gettempdir(), f"model_output_{today}.png")
            plt.savefig(filename, dpi=300)
            plt.show()
    return digital_twins, df_assimilates


def run_ascabmodel(
    parcels: list[AgriParcel] = None,
    end_date: datetime.date = None,
    debug=False,
):
    if parcels is None:
        parcels = search(get_demo_parcels("Serrater"), ctx=AgriFood.ctx)
    pesticide_object = find_agriproducttype(name="fungicide")[0]
    digital_twins = create_digital_twins_ascab(parcels, end_date=end_date)

    # run digital twins
    for digital_twin in digital_twins:
        devices = find_device(digital_twin._isAgriCrop)
        sim_dict = get_sim_dict(devices)

        terminated = False
        digital_twin.reset()
        while not terminated:
            parcel_operations = find_parcel_operations(digital_twin._locatedAtParcel)
            parcel_operation = get_parcel_operation_by_date(
                parcel_operations,
                (digital_twin.date - datetime.timedelta(days=1)),
            )
            action = parcel_operation.quantity if parcel_operation else 0

            _, _, terminated, _, _ = digital_twin.step(action)

            ask_recommendation = (
                digital_twin.date == end_date if end_date is not None else True
            )
            ask_recommendation = True  # TODO

            info = digital_twin.get_wrapper_attr("info")
            for variable, (device, device_measurement) in sim_dict.items():
                stripped_variable = variable.split("-", 1)[1]
                if info[stripped_variable] is not None:
                    device_measurement.numValue = info[stripped_variable][-1]
                    device_measurement.dateObserved = (
                        info["Date"][-1].isoformat() + "T00:00:00Z"
                    )
                    upsert(device_measurement)
            recommendation = 0.0
            if ask_recommendation:
                recommendation = minimize_risk(info)

            # create command message
            if recommendation > 0:
                parcel = get_by_id(digital_twin._locatedAtParcel, ctx=AgriFood.ctx)
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
                operation = create_agriparcel_operation(
                    parcel=parcel,
                    product=pesticide_object,
                    quantity=recommendation,
                    date=datetime.datetime.fromisoformat(
                        info["Date"][-1].isoformat()
                    ).strftime("%Y%m%d"),
                    operationtype="sim-pesticide",
                )

        # get output
        summary_output = digital_twin.get_info(to_dataframe=True)
        if debug:
            print(digital_twin)
            print("The following commands were stored\n")
            print(find_command_messages())

            today = datetime.datetime.today().strftime("%Y-%m-%d")
            plot_results(
                summary_output,
                save_path=os.path.join(
                    tempfile.gettempdir(), f"model_ascab_{today}.png"
                ),
            )

    return
