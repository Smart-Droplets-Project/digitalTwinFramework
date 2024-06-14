import datetime
import os
from geojson import Polygon, MultiPoint, MultiLineString
from typing import Union
import yaml

import numpy as np
import pandas as pd

import pcse
from pcse.base import WeatherDataProvider
from pcse.engine import Engine
from pcse.base.parameter_providers import ParameterProvider

import pcse_gym
from pcse_gym.envs.winterwheat import WinterWheat
import pcse_gym.utils.defaults as defaults

import onnx
import onnxruntime as ort

import ngsildclient
from ngsildclient import Entity, Client, SmartDataModels

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SRC_DIR))
CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")


def init_cropgym(
    crop_features=defaults.get_default_crop_features(pcse_env=1),
    costs_nitrogen: int = 10,
    reward: str = "DEF",
    nitrogen_levels: int = 7,
    action_multiplier: float = 1.0,
    years: list = defaults.get_default_train_years(),
    locations: list = defaults.get_default_location(),
    crop_parameters: str = None,
    site_parameters: str = None,
    soil_parameters: str = None,
    agro_config: str = None,
    model_config: str = None,
):
    def get_action_space(nitrogen_levels=7):
        import gymnasium as gym

        space_return = gym.spaces.Discrete(nitrogen_levels)
        return space_return

    action_space = get_action_space(nitrogen_levels=nitrogen_levels)
    env_return = WinterWheat(
        crop_features=crop_features,
        costs_nitrogen=costs_nitrogen,
        years=years,
        locations=locations,
        action_space=action_space,
        action_multiplier=action_multiplier,
        reward=reward,
        crop_parameters=crop_parameters,
        site_parameters=site_parameters,
        soil_parameters=soil_parameters,
        agro_config=agro_config,
        model_config=model_config,
    )

    return env_return


# Returns digital twin instance
def init_digital_twin(
    parameter_provider, agro_config, model_config, weather_data_provider
) -> Engine:

    crop_growth_model = pcse.engine.Engine(
        parameterprovider=parameter_provider,
        weatherdataprovider=weather_data_provider,
        agromanagement=agro_config,
        config=model_config,
    )

    return crop_growth_model


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


# TODO: Get from local weather station. For now use NASAPOWER
def get_weather_provider() -> pcse.input.NASAPowerWeatherDataProvider:
    return pcse.input.NASAPowerWeatherDataProvider(*(55.0, 23.5))


def create_crop(crop_type) -> Entity:
    global N_CROPS
    N_CROPS += 1

    # Load exmaple model and edit relevant data
    crop = Entity.load(SmartDataModels.SmartAgri.Agrifood.AgriCrop)

    crop.id = f"urn:ngsi-ld:AgriCrop:crop-{N_CROPS}-id"
    crop.prop("description", crop_type)
    crop.prop("name", crop_type)
    if crop_type == "wheat":
        alt_name = "Triticum aestivum L."
    else:
        alt_name = ""
    crop.prop("alternateName", f"{alt_name}")

    return crop


# currently naively divide subparcels into equal parts
# TODO need to divide subparcels more accurately
def _get_area_sub_parcels(area: float, n_rows: int):
    # assume rows are straight and segments the parcel
    n_sub_parcels = n_rows + 1
    area_sub_parcels = area / n_rows

    return [area_sub_parcels for _ in range(n_sub_parcels)]


def _get_sub_parcels(crop: Entity, location: tuple, area: float, n_rows: int):
    areas = _get_area_sub_parcels(area, n_rows)
    id_sub_parcels = 0
    sub_parcel_entities = []
    for area in areas:
        sub_parcel_entity = create_parcel(
            crop, location, is_sub_parcel=True, area=area, id_sub_parcels=id_sub_parcels
        )
        sub_parcel_entities.append(sub_parcel_entity)
        id_sub_parcels += id_sub_parcels

    return sub_parcel_entities


def create_parcel(
    crop: Entity,
    location: Union[Polygon, MultiPoint],  # tuple of lat and lon coordinates
    parameter_provider: ParameterProvider = None,
    weather_data_provider: WeatherDataProvider = None,
    model_config: str = None,
    agro_config: str = None,
    area_parcel: float = 1.0,
    n_rows: int = 0,
    is_sub_parcel: bool = False,
    id_sub_parcel: int = None,
    **kwargs,
):
    # Load model and edit relevant data
    global N_PARCELS
    parcel = Entity.load(SmartDataModels.SmartAgri.Agrifood.AgriParcel)

    # Load planting dates
    planting_date = ...

    if not is_sub_parcel:
        N_PARCELS += 1
    """
    1. URN - Uniform Resource Name
    2. NGSI-LD 
    3. AgriParcel definition from the Smart Data Initiative
    4. Custom ID part, usually a UUID (Universally Unique IDentifier) or some kind of date format
    """
    if not is_sub_parcel:
        parcel.id = f"urn:ngsi-ld:AgriParcel:parcel-{N_PARCELS}-id"
        parcel_dt = init_digital_twin(
            parameter_provider=parameter_provider,
            model_config=model_config,
            agro_config=agro_config,
            weather_data_provider=weather_data_provider,
        )
    else:
        parcel.id = (
            f"urn:ngsi-ld:AgriParcel:subparcel-{id_sub_parcel}-id_parcel-{N_PARCELS}-id"
        )
        parcel_dt = None

    # parcel.landLocation = None
    parcel.prop("location", location)
    parcel.prop("area", area_parcel)  # in ha
    parcel.rel("hasAgriCrop", crop.id)
    if n_rows:
        sub_parcels, _ = _get_sub_parcels(crop, location, area_parcel, n_rows)
        parcel.rel("hasAgriParcelChildren", sub_parcels)
    parcel.rel("hasDevices", f"LAI-parcel-{N_PARCELS}")
    parcel.rel("hasDevices", f"soilMoisture-parcel-{N_PARCELS}")

    parcel.pprint()

    return parcel, parcel_dt


def register_dt_entity(entity):
    with Client() as client:
        print(f"Inserting entity! [{entity.id} {entity.type}]")
        client.create(entity)


def main():
    wheat_crop_entity = create_crop("wheat")

    location = MultiPoint([])

    # load CropGym / WOFOST

    parcel_entitiy, dt_instance = create_parcel(
        crop=wheat_crop_entity,
        location=location,
        **get_config_files(),
        weather_data_provider=get_weather_provider(),
        area_parcel=50,
        n_rows=3,
    )


if __name__ == "__main__":
    main()
