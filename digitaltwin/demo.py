import datetime
import os
from geojson import (
    Polygon,
    MultiLineString,
    Point,
    Feature,
    FeatureCollection,
)
from typing import Union
import yaml
import argparse

import pcse
from pcse.base.parameter_providers import ParameterProvider
from typing import List

from sd_data_adapter.client import DAClient
from sd_data_adapter.api import upload, search, get_by_id
import sd_data_adapter.models.agri_food as agri_food_model
import sd_data_adapter.models.device as device_model
import sd_data_adapter.models.autonomous_mobile_robot as autonomous_mobile_robot
from sd_data_adapter.models import AgriFood, Devices, AutonomousMobileRobot


from utils.agromanagement_util import AgroManagement

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIGS_DIR = os.path.join(SRC_DIR, "configs")
PCSE_MODEL_CONF_DIR = os.path.join(CONFIGS_DIR, "Wofost81_NWLP_MLWB_SNOMIN.conf")


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


class CropModel(pcse.engine.Engine):
    """
    Wraps around the PCSE engine/crop model for correct rate updates after fertilization action and
    to set a flag when the simulation has terminated
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flag_terminated = False

    def _run(self, action):
        """Make one time step of the simulation."""

        # Update timer
        self.day, delt = self.timer()

        # State integration
        self.integrate(self.day, delt)

        # Driving variables
        self.drv = self._get_driving_variables(self.day)

        # Agromanagement decisions
        self.agromanager(self.day, self.drv)

        # Do actions
        if action > 0:
            self._send_signal(
                signal=pcse.signals.apply_n_snomin,
                amount=action,
                application_depth=10.0,
                cnratio=0.0,
                f_orgmat=0.0,
                f_NH4N=0.5,
                f_NO3N=0.5,
                initial_age=0,
            )
        # Rate calculation
        self.calc_rates(self.day, self.drv)

        if self.flag_terminate is True:
            self._terminate_simulation(self.day)

    def run(self, days=1, action=0):
        """Advances the system state with given number of days"""

        # do action at end of time step
        days_counter = days
        days_done = 0
        while (days_done < days) and (self.flag_terminate is False):
            days_done += 1
            days_counter -= 1
            if days_counter > 0:
                self._run(0)
            else:
                self._run(action)

    @property
    def terminated(self):
        return self._flag_terminated

    def _terminate_simulation(self, day):
        super()._terminate_simulation(day)
        self._flag_terminated = True


def get_weather_provider(
    parcel: agri_food_model.AgriParcel,
) -> pcse.input.NASAPowerWeatherDataProvider:

    location = None
    for feature in parcel.location["features"]:
        if (
            feature["properties"]["name"] == "weather location"
            and feature["geometry"]["type"] == "Point"
        ):
            location = feature["geometry"]["coordinates"]
    return pcse.input.NASAPowerWeatherDataProvider(*location)


def create_crop(crop_type: str, do_upload=True) -> agri_food_model.AgriCrop:
    """
    Function to create SDM crop entity

    :param do_upload: upload to registry
    :param crop_type: String of generic crop type
    :return: AgriCrop entity
    """
    model = agri_food_model.AgriCrop(
        alternateName="Arminda" if crop_type == "wheat" else "",
        description=crop_type,
        dateCreated=str(datetime.datetime.now()),
        dateModified=str(datetime.datetime.now()),
        # TODO grab from somewhere
        plantingFrom=[
            "20221003",
            "20230820",
        ],  # List of planting and harvest date in YYYYMMDD in str
    )
    if do_upload:
        upload(model)
    return model


def create_fertilizer(do_upload=True) -> agri_food_model.AgriProductType:
    model = agri_food_model.AgriProductType(type="fertilizer", name="Nitrogen")
    if do_upload:
        upload(model)
    return model


def create_fertilizer_application(
    parcel: agri_food_model.AgriParcel,
    product: agri_food_model.AgriProductType,
    quantity=60,
    date: str = "20230401",
    do_upload=True,
) -> agri_food_model.AgriParcelOperation:
    model = agri_food_model.AgriParcelOperation(
        operationType="fertiliser",
        hasAgriParcel=parcel.id,
        hasAgriProductType=product.id,
        plannedStartAt=date,
        quantity=quantity,
    )
    if do_upload:
        upload(model)
    return model


def create_parcel(
    location: Union[FeatureCollection, Point, MultiLineString, Polygon],
    area_parcel: float,
    crop: agri_food_model.AgriCrop,
    soil: agri_food_model.AgriSoil,
    do_upload=True,
) -> agri_food_model.AgriParcel:
    """
    Function to initialize a parcel Entity.

    :param location: A geoJSON object describing multiple things:
                    - Point: A point to reference for weather/meteo provider
                    - MultiLineString: Multi lines describing tractor rows for the retrofitted tractor
                    - Polygon: Describes parcel area
                    - FeatureCollection: a collection of the above features
    :param area_parcel: A float of parcel area in hectares (ha)
    :param crop: A SmartDataModel crop entity
    :param soil: A SmartDataModel soil entity
    :param do_upload: Bool to upload entity to Data Management Platform
    :return: AgriParcel SmartDataModel entity
    """
    model = agri_food_model.AgriParcel(
        location=location,
        area=area_parcel,
        hasAgriCrop=crop.id,
        hasAgriSoil=soil.id,
        description="initial_site",  # TODO placeholder description
    )
    if do_upload:
        upload(model)
    return model


def get_crop_and_variety_name(crop: agri_food_model.AgriCrop):
    crop_name, variety_name = crop.description, crop.alternateName
    return crop_name, variety_name


def get_soil_parameters(soil: agri_food_model.AgriSoil):
    soil_type = soil.description
    soil_parameters = yaml.safe_load(
        open(os.path.join(CONFIGS_DIR, "soil", f"{soil_type}.yaml"))
    )
    return soil_parameters


def get_site_parameters(site: agri_food_model.agriParcel):
    site_name = site.description
    site_parameters = yaml.safe_load(
        open(os.path.join(CONFIGS_DIR, "site", f"{site_name}.yaml"))
    )
    return site_parameters


# TODO placeholder description
def create_agrisoil(do_upload=True):
    model = agri_food_model.AgriSoil(description="layered_soil")
    if do_upload:
        upload(model)
    return model


def create_device(crop: agri_food_model.agriCrop, variable: str, do_upload=True):
    model = device_model.Device(controlledProperty=variable, controlledAsset=crop.id)
    if do_upload:
        upload(model)
    return model


def create_device_measurement(
    device: device_model.Device, date_observed: str, value: float, do_upload=True
):
    model = device_model.DeviceMeasurement(
        dateObserved=date_observed,
        numValue=value,
        refDevice=device.id,
    )
    if do_upload:
        upload(model)
    return model


def create_command_message(
    message_id, command, command_time, waypoints, do_upload=True
):
    model = autonomous_mobile_robot.CommandMessage(
        id=message_id,
        command=command,
        commandTime=command_time,
        waypoints=waypoints,
    )
    if do_upload:
        upload(model)
    return model


def get_agro_config(
    crop_name: str,
    variety_name: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    start_type: str = "sowing",
    end_type: str = "harvest",
    max_duration: int = 365,
):

    with open(os.path.join(CONFIGS_DIR, "agro", "wheat_cropcalendar.yaml"), "r") as f:
        init_agro_config = yaml.load(f, Loader=yaml.SafeLoader)
    agro_config_container = AgroManagement(init_agro_config)

    crop_name = "winterwheat" if crop_name == "wheat" else "wheat"
    agro_config_container.set_crop_name(crop_name)
    agro_config_container.set_variety_name(variety_name)
    agro_config_container.set_start_date(start_date)
    agro_config_container.set_end_date(end_date)
    agro_config_container.set_start_type(start_type)
    agro_config_container.set_end_type(end_type)
    agro_config_container.set_max_duration(max_duration)

    agro_config = agro_config_container.load_agromanagement_file
    return agro_config


def generate_feature_collections(
    point: Point = None,
    point_name: str = "weather location",
    multilinestring: MultiLineString = None,
    multilinestring_name: str = "rows",
    polygon: Polygon = None,
    polygon_name: str = "parcel area",
) -> FeatureCollection:
    ...
    point = point if point is not None else Point()
    point_feature = Feature(
        geometry=point, properties={"name": point_name}
    )  # check if any properties are needed

    multilinestring = (
        multilinestring if multilinestring is not None else MultiLineString()
    )
    multilinestring_feature = Feature(
        geometry=multilinestring, properties={"name": multilinestring_name}
    )

    polygon = polygon if polygon is not None else Polygon()
    polygon_feature = Feature(geometry=polygon, properties={"name": polygon_name})

    return FeatureCollection([point_feature, multilinestring_feature, polygon_feature])


def get_row_coordinates(
    parcel_loc: Union[FeatureCollection, agri_food_model.AgriParcel.location],
):
    multi_line_string_coords = []
    for feature in parcel_loc["features"]:
        if feature["geometry"]["type"] == "MultiLineString":
            coords = feature["geometry"]["coordinates"]
            multi_line_string_coords.append(coords)
    return multi_line_string_coords


def create_digital_twins(
    parcels: List[agri_food_model.AgriParcel],
) -> dict[agri_food_model.AgriParcel.id, pcse.engine.Engine]:
    crop_parameters = pcse.input.YAMLCropDataProvider(
        fpath=os.path.join(CONFIGS_DIR, "crop"), force_reload=True
    )
    digital_twin_dict = {}
    for parcel in parcels:
        crop = get_by_id(parcel.hasAgriCrop["object"])
        soil = get_by_id(parcel.hasAgriSoil["object"])
        crop_name, variety_name = get_crop_and_variety_name(crop)
        planting_date, harvest_date = (
            datetime.datetime.strptime(crop.plantingFrom[0], "%Y%m%d"),
            datetime.datetime.strptime(crop.plantingFrom[1], "%Y%m%d"),
        )
        soil_parameters = get_soil_parameters(soil)
        site_parameters = get_site_parameters(parcel)
        agro_config = get_agro_config(
            crop_name,
            variety_name,
            planting_date,
            harvest_date,
        )
        weatherdataprovider = get_weather_provider(parcel)

        parameter_provider = ParameterProvider(
            cropdata=crop_parameters, sitedata=site_parameters, soildata=soil_parameters
        )
        crop_growth_model = CropModel(
            parameterprovider=parameter_provider,
            weatherdataprovider=weatherdataprovider,
            agromanagement=agro_config,
            config=PCSE_MODEL_CONF_DIR,
        )
        digital_twin_dict[parcel.id] = crop_growth_model

    return digital_twin_dict


def get_demo_parcels():
    return {"type": "AgriParcel", "q": 'description=="initial_site"'}


def find_parcel_operations(parcel):
    return search(
        {"type": "AgriParcelOperation", "q": f'hasAgriParcel=="{parcel}"'},
        ctx=AgriFood.ctx,
    )


def find_crop(parcel_id):
    parcel = get_by_id(parcel_id, ctx=AgriFood.ctx)
    return parcel.hasAgriCrop["object"]


def find_device(crop_id):
    return search(
        {"type": "Device", "q": f'controlledAsset=="{crop_id}"'},
        ctx=Devices.ctx,
    )


def find_command_messages():
    return search(
        {"type": "CommandMessage"},
        ctx=AutonomousMobileRobot.ctx,
    )


def fill_database():
    wheat_crop = create_crop("wheat")
    soil = create_agrisoil()
    geo_feature_collection = generate_feature_collections(
        point=Point((52.0, 5.5)),  # for weather data (latitude, longitude)
        multilinestring=(
            MultiLineString(
                [
                    [(3.75, 9.25), (-130.95, 1.52)],
                    [(23.15, -34.25), (-1.35, -4.65), (3.45, 77.95)],
                ]
            )
        ),  # for rows
        polygon=Polygon(),  # for parcel area
    )
    parcel = create_parcel(
        location=geo_feature_collection, area_parcel=20, crop=wheat_crop, soil=soil
    )
    fertilizer = create_fertilizer()
    fertilizer_application = create_fertilizer_application(
        parcel=parcel, product=fertilizer
    )
    device = create_device(crop=wheat_crop, variable="LAI")


def has_demodata():
    return bool(search(get_demo_parcels()))


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


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

    # clear database with error handling
    try:
        with DAClient.get_instance() as client:
            client.purge()
        print("Purge completed successfully.")
    except Exception as e:
        # Catch any exceptions that occur and print the error message
        print(f"An error occurred while purging the database: {e}")

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
