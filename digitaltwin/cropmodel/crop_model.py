import datetime
import os
import pcse
from pcse.engine import Engine
from pcse.base.parameter_providers import ParameterProvider
import sd_data_adapter.models.agri_food as agri_food_model
import yaml
from typing import Optional, List

from sd_data_adapter.api import search, get_by_id
from sd_data_adapter.models.smartDataModel import Relationship
from .agromanagement import AgroManagement

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIGS_DIR = os.path.join(SRC_DIR, "configs")
PCSE_MODEL_CONF_DIR = os.path.join(CONFIGS_DIR, "Wofost81_NWLP_MLWB_SNOMIN.conf")


class CropModel(pcse.engine.Engine):
    """
    Wraps around the PCSE engine/crop model for correct rate updates after fertilization action and
    to set a flag when the simulation has terminated
    """

    def __init__(self, parcel_id: str, crop_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flag_terminated = False
        self._locatedAtParcel: Optional[Relationship] = parcel_id
        self._isAgriCrop: Optional[Relationship] = crop_id

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


def get_default_variables():
    # TODO: make sure that the following aligns with OUTPUT_VARS in Wofost81_NWLP_MLWB_SNOMIN.conf
    return ["DVS", "LAI", "TAGP", "TWSO"]


def create_digital_twins(
    parcels: List[agri_food_model.AgriParcel],
) -> List[CropModel]:
    crop_parameters = pcse.input.YAMLCropDataProvider(
        fpath=os.path.join(CONFIGS_DIR, "crop"), force_reload=True
    )
    results = []
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
            parcel_id=parcel.id,
            crop_id=crop.id,
        )
        results.append(crop_growth_model)

    return results
