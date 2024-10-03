import datetime
import os
from geojson import Polygon, MultiPoint, MultiLineString, Point, Feature, FeatureCollection
from typing import Union
import yaml
import numpy as np

import pcse
from pcse.base import WeatherDataProvider
from pcse.engine import Engine
from pcse.base.parameter_providers import ParameterProvider
from typing import List

# import onnx
# import onnxruntime as ort

# from ngsildclient import Entity, Client, SmartDataModels

from sd_data_adapter.api import upload, search, get_by_id, update
import sd_data_adapter.models.agri_food as agri_food_model
import sd_data_adapter.models.autonomous_mobile_robot as robot_model

from utils.agromanagement_util import AgroManagement
from utils.cropgym_helpers import extract_digital_twin_obs, placeholder_recommendation

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
# ROOT_DIR = os.path.dirname(os.path.dirname(SRC_DIR))
CONFIGS_DIR = os.path.join(SRC_DIR, "configs")
PCSE_MODEL_CONF_DIR = os.path.join(CONFIGS_DIR, "Wofost81_NWLP_MLWB_SNOMIN.conf")




# TODO placeholders for now. Config files will change when needed.
def get_config_files() -> dict:
    crop_parameters = pcse.input.YAMLCropDataProvider(
        fpath=os.path.join(CONFIGS_DIR, "crop"), force_reload=True
    )
    site_parameters = yaml.safe_load(
        open(os.path.join(CONFIGS_DIR, "site", "initial_site.yaml"))
    )
    soil_parameters = yaml.safe_load(
        open(os.path.join(CONFIGS_DIR, "soil", "layered_soil.yaml"))
    )

    parameter_provider = ParameterProvider(
        crop_parameters, site_parameters, soil_parameters
    )

    agro_config = os.path.join(
        os.path.join(CONFIGS_DIR, "agro", "wheat_cropcalendar.yaml")
    )
    model_config = os.path.join(CONFIGS_DIR, "Wofost81_NWLP_MLWB_SNOMIN.conf")

    return {
        "parameter_provider": parameter_provider,
        "agro_config": agro_config,
        "model_config": model_config,
    }


class DMPWeatherProvider(pcse.base.weather.WeatherDataProvider):
    def add(self, weather_dict):
        wdc = pcse.base.weather.WeatherDataContainer(**weather_dict)
        self._store_WeatherDataContainer(wdc, weather_dict["DAY"])


def get_weather(df, day) -> dict:
    return df.loc[df["DAY"] == day].to_dict(orient="records")[0]


# some helper functions that we will use later on
def daterange(start_date, end_date) -> datetime.date:
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


# to_obs converts state variables of the crop growth model weather data to a "tensor" that can be fed to the AI agent
def to_obs(crop_data, weather_data) -> Union[list, np.array]:
    crop_variables = [
        "DVS",
        "TAGP",
        "LAI",
        "NuptakeTotal",
        "TRA",
        "NO3",
        "NH4",
        "SM",
        "RFTRA",
        "WSO",
    ]
    weather_variables = ["IRRAD", "TMIN", "RAIN"]
    timestep = 7

    obs = np.zeros(len(crop_variables) + timestep * len(weather_variables))
    for i, feature in enumerate(crop_variables):
        obs[i] = crop_data[feature]

    for d in range(timestep):
        for i, feature in enumerate(weather_variables):
            j = d * len(weather_variables) + len(crop_variables) + i
            obs[j] = getattr(weather_data[d], feature)
    return obs


def get_weather_provider(parcel: agri_food_model.AgriParcel) -> pcse.input.NASAPowerWeatherDataProvider:
    location = parcel.location['features'][0]['geometry']['coordinates']  #TODO replace with robust way to reference Point object
    return pcse.input.NASAPowerWeatherDataProvider(*location)
    #return pcse.input.NASAPowerWeatherDataProvider(*(55.0, 23.5))


def create_crop(crop_type: str, do_upload=True) -> agri_food_model.AgriCrop:
    """
    Function to create SDM crop entity

    :param crop_type: String of generic crop type
    :return: AgriCrop entity
    """
    model = agri_food_model.AgriCrop(
        alternateName="Triticum aestivum L." if crop_type == "wheat" else "",
        description=crop_type,
        dateCreated=str(datetime.datetime.now()),
        dateModified=str(datetime.datetime.now()),
        # TODO grab from somewhere
        plantingFrom=["20221003", "20230820"]  # List of planting and harvest date in YYYYMMDD in str
    )
    if do_upload:
        upload(model)
    return model


def create_parcel(location: Union[FeatureCollection, Point, MultiLineString, Polygon],
                  area_parcel: float,
                  crop: agri_food_model.AgriCrop,
                  soil: agri_food_model.AgriSoil,
                  do_upload=True) -> agri_food_model.AgriParcel:
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
        description="initial_site"  # TODO placeholder description
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
        open(os.path.join(CONFIGS_DIR, "soil", f"{soil_type}.yaml")))
    return soil_parameters


def get_site_parameters(site: agri_food_model.agriParcel):
    site_name = site.description
    site_parameters = yaml.safe_load(
        open(os.path.join(CONFIGS_DIR, "site", f"{site_name}.yaml")))
    return site_parameters


# TODO placeholder description
def create_agrisoil(do_upload=True):
    model = agri_food_model.AgriSoil(
        description="layered_soil"
    )
    if do_upload:
        upload(model)
    return model


def get_agro_config(crop_name: str,
                    variety_name: str,
                    start_date: datetime.datetime,
                    end_date: datetime.datetime,
                    start_type: str = 'sowing',
                    end_type: str = 'harvest',
                    max_duration: int = 365):

    with open(os.path.join(CONFIGS_DIR, "agro", "wheat_cropcalendar.yaml"), 'r') as f:
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


def generate_feature_collections(point: Point = None,
                                 point_name: str = 'weather location',
                                 multilinestring: MultiLineString = None,
                                 multilinestring_name: str = 'rows',
                                 polygon: Polygon = None,
                                 polygon_name: str = 'parcel area',
                                 ) -> FeatureCollection:
    ...
    point = point if point is not None else Point()
    point_feature = Feature(geometry=point, properties={"name": point_name})  # check if any properties are needed

    multilinestring = multilinestring if multilinestring is not None else MultiLineString()
    multilinestring_feature = Feature(geometry=multilinestring, properties={"name": multilinestring_name})

    polygon = polygon if polygon is not None else Polygon()
    polygon_feature = Feature(geometry=polygon, properties={"name": polygon_name})


    return FeatureCollection([point_feature, multilinestring_feature, polygon_feature])


def get_row_coordinates(parcel_loc: Union[FeatureCollection, agri_food_model.AgriParcel.location], ):
    multi_line_string_coords = []
    for feature in parcel_loc['features']:
        if feature['geometry']['type'] == 'MultiLineString':
            coords = feature['geometry']['coordinates']
            multi_line_string_coords.append(coords)
    return multi_line_string_coords


def create_digital_twins(parcels: List[agri_food_model.AgriParcel]) -> dict:
    crop_parameters = pcse.input.YAMLCropDataProvider(
        fpath=os.path.join(CONFIGS_DIR, "crop"), force_reload=True
    )

    digital_twin_dict = {}
    for parcel in parcels:
        crop = get_by_id(parcel.hasAgriCrop)
        soil = get_by_id(parcel.hasAgriSoil)
        crop_name, variety_name = get_crop_and_variety_name(crop)
        planting_date, harvest_date = (datetime.datetime.strptime(crop.plantingFrom[0], "%Y%m%d"),
                                       datetime.datetime.strptime(crop.plantingFrom[1], "%Y%m%d"))

        soil_parameters = get_soil_parameters(soil)
        site_parameters = get_site_parameters(parcel)
        agro_config = get_agro_config(crop_name,
                                      variety_name,
                                      planting_date,
                                      harvest_date,)
        weatherdataprovider = get_weather_provider(parcel)

        parameter_provider = ParameterProvider(crop_parameters, site_parameters, soil_parameters)
        crop_growth_model = pcse.engine.Engine(
            parameterprovider=parameter_provider,
            weatherdataprovider=weatherdataprovider,
            agromanagement=agro_config,
            config=PCSE_MODEL_CONF_DIR,
        )

        digital_twin_dict[parcel.id] = crop_growth_model

    return digital_twin_dict


def get_recommendation_message(recommendation, day, parcel_id):
    return f"rec-fertilize:{recommendation}:day:{day}:parcel_id:{parcel_id}"  # might need to change


def create_command_message(message_id,
                           command,
                           command_time,
                           waypoints,
                           do_upload=True):
    model = robot_model.CommandMessage(
        id=message_id,
        command=command,
        commandTime=command_time,
        waypoints=waypoints,
    )
    if do_upload:
        upload(model)
    return model


# TODO do executions in appropriate methods
def generate_rec_message_id(day, parcel_id):
    return f"urn:ngsi-ld:CommandMessage:rec-{day}-'{parcel_id}'"


def main():
    wheat_crop = create_crop("wheat")
    soil = create_agrisoil()
    geo_feature_collection = generate_feature_collections(
        point=Point((5.5, 52.0)),  # for weather data
        multilinestring=MultiLineString(),  # for rows
        polygon=Polygon(),  # for parcel area
    )
    wheat_parcel = create_parcel(location=geo_feature_collection, area_parcel=20, crop=wheat_crop, soil=soil)
    search_params = {
        'type': 'AgriParcel',
        'q': 'description=="WheatParcel"'
    }
    my_parcels = search(search_params)
    print(f'database contains {my_parcels}')

    digital_twin_dicts = create_digital_twins([wheat_parcel])

    # generate example commandmessage for tractor
    obs, day = extract_digital_twin_obs(output=digital_twin_dicts[wheat_parcel.id].get_output())

    recommendation = placeholder_recommendation(obs)                # replace when digital twin logic ready

    recommendation_message = get_recommendation_message(recommendation=recommendation,
                                                        day=day,
                                                        parcel_id=wheat_parcel.id)

    command_message_id = generate_rec_message_id(day=day,
                                                 parcel_id=wheat_parcel.id)

    command = create_command_message(message_id=command_message_id,
                                     command=recommendation_message,
                                     command_time=day,
                                     waypoints=get_row_coordinates(wheat_parcel.location))

    print(command.command)


if __name__ == "__main__":
    main()
