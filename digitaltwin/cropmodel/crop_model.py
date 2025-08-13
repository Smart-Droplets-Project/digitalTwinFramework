import datetime
import os
import yaml
import pandas as pd
import numpy as np
import nlopt
from typing import Optional, List
import pcse
from pcse.engine import Engine
from pcse.base.parameter_providers import ParameterProvider
import sd_data_adapter.models.agri_food as agri_food_model
from sd_data_adapter.api import get_by_id
from sd_data_adapter.models.smartDataModel import Relationship
from .agromanagement import AgroManagement

from digitaltwin.cropmodel.recommendation import scheduled, CropgymAgent
from digitaltwin.utils.database import (
    find_parcel_operations,
    get_parcel_operation_by_date,
)
from digitaltwin.utils.openmeteo import OpenMeteoWeatherProvider

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIGS_DIR = os.path.join(SRC_DIR, "configs")
PCSE_MODEL_CONF_DIR = os.path.join(CONFIGS_DIR, "Wofost81_NWLP_MLWB_SNOMIN.conf")


class CropModel(pcse.engine.Engine):

    def __init__(self, parcel_id: str, crop_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flag_terminated = False
        self._locatedAtParcel: Optional[Relationship] = parcel_id
        self._isAgriCrop: Optional[Relationship] = crop_id
        self._agromanagement = kwargs["agromanagement"]

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
        while days_done < days:
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

    @property
    def get_agromanagement(self):
        return self._agromanagement


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
    provider: str = "openmeteo",
) -> pcse.base.WeatherDataProvider:
    """
    returns either NASA Power or OpenMeteo weather provider,
    by inputting "nasapower" or "openmeteo", respectively.
    """
    location = None
    for feature in parcel.location["features"]:
        if (
            feature["properties"]["name"] == "weather location"
            and feature["geometry"]["type"] == "Point"
        ):
            location = feature["geometry"]["coordinates"]
    if provider == "openmeteo":
        return OpenMeteoWeatherProvider(*location, force_update=True)
    elif provider == "nasapower":
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
    # TODO: Temp code, add Lithuanian parameters statically or do it dynamically
    if "initial_site" in site_parameters:
        site_parameters = yaml.safe_load(
            open(os.path.join(CONFIGS_DIR, "site", f"{site_parameters}"))
        )
    return site_parameters


def get_default_variables():
    # TODO: make sure that the following aligns with OUTPUT_VARS in Wofost81_NWLP_MLWB_SNOMIN.conf
    return ["DVS", "LAI", "NAVAIL", "TWSO"]


def get_titles():
    result = {
        "DVS": ("Development stage", "-"),
        "TAGP": ("Total aboveground biomass", "kg/ha"),
        "LAI": ("Leaf area Index", "-"),
        "NuptakeTotal": ("Total nitrogen uptake", "kgN/ha"),
        "NAVAIL": ("Total soil inorganic nitrogen", "kgN/ha"),
        "TWSO": ("Weight storage organs", "kg/ha"),
    }
    return result


def create_cropgym_agents(
    parcels: List[agri_food_model.AgriParcel],
    digital_twins: List[CropModel],
) -> List[CropgymAgent]:
    agents = []
    for parcel, digital_twin in zip(parcels, digital_twins):
        cropgym_agent = CropgymAgent(
            parcel_id=parcel.id,
            agromanagement=digital_twin.get_agromanagement,
        )
        agents.append(cropgym_agent)
    return agents


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
        config_date = list(agro_config[0].keys())[0]
        crop_name = agro_config[0][config_date]["CropCalendar"]["crop_name"]
        variety_name = agro_config[0][config_date]["CropCalendar"]["variety_name"]
        parameter_provider.set_active_crop(crop_name, variety_name)
        results.append(crop_growth_model)

    return results


class ModelRerunner(object):
    """Reruns a given model with different values of parameters

    Returns a pandas DataFrame with simulation results of the model with given
    parameter values.
    """

    def __init__(
        self, params, wdp, agro, model_config, parameters_calibration, **kwargs
    ):
        self.params = params
        self.wdp = wdp
        self.agro = agro
        self.model_config = model_config
        self.parameters_calibration = parameters_calibration
        self.parcel_operations = kwargs.get("parcel_operations", None)
        self.end_date = kwargs.get("end_date", None)

    def update_cropcropmodel(self, par_values):
        if len(par_values) != len(self.parameters_calibration):
            msg = "Optimizing %i parameters, but only % values were provided!" % (
                len(self.parameters_calibration, len(par_values))
            )
            raise RuntimeError(msg)
        # Clear any existing overrides
        self.params.clear_override()
        # Set overrides for the new parameter values
        for parname, value in zip(self.parameters_calibration, par_values):
            if parname in self.params._unique_parameters:
                self.params.set_override(parname, value)

        # Run the model with given parameter values
        model_mod = pcse.engine.Engine(
            self.params, self.wdp, self.agro, config=self.model_config
        )
        return model_mod

    def __call__(self, par_values):
        # Check if correct number of parameter values were provided
        model_mod = self.update_cropcropmodel(par_values)
        # add agromanagement
        # model_mod.run_till_terminate()
        model_mod = self.run_till_terminate_with_recommendations(model_mod)
        df = pd.DataFrame(model_mod.get_output())
        df.index = pd.to_datetime(df.day)
        return df

    # run the updated parameters with management
    def run_till_terminate_with_recommendations(self, model_mod):
        while model_mod.flag_terminate is False:
            model_mod = self.run(model_mod)

        return model_mod

    # TODO maybe a redundant function
    def run(self, model_mod):
        """Make one time step of the simulation."""

        # Update timer
        model_mod.day, delt = model_mod.timer()

        # State integration
        model_mod.integrate(model_mod.day, delt)

        # Driving variables
        model_mod.drv = model_mod._get_driving_variables(model_mod.day)

        # Agromanagement decisions
        model_mod.agromanager(model_mod.day, model_mod.drv)

        # Grab recommendations from parcel operation
        if self.parcel_operations is None:
            action = scheduled(model_mod.get_output()[-1])
        else:
            parcel_operation = get_parcel_operation_by_date(
                self.parcel_operations, model_mod.day
            )
            action = parcel_operation.quantity if parcel_operation else 0

        if action is not None and action > 0:
            model_mod._send_signal(
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
        model_mod.calc_rates(model_mod.day, model_mod.drv)

        if self.end_date is not None and model_mod.day >= self.end_date:
            model_mod.flag_terminate = True
        if model_mod.flag_terminate is True:
            model_mod._terminate_simulation(model_mod.day)
        return model_mod


class ObjectiveFunctionCalculator(object):
    """Computes the objective function.

        This class runs the simulation model with given parameter values and returns the objective
        function as the sum of squared difference between observed and simulated.
    ."""

    def __init__(
        self,
        params,
        wdp,
        agro,
        model_config,
        parameters_calibration,
        observations,
        **kwargs,
    ):
        self.parcel_operations = kwargs.get("parcel_operations", None)
        self.end_date = kwargs.get("end_date", None)
        self.modelrerunner = ModelRerunner(
            params,
            wdp,
            agro,
            model_config,
            parameters_calibration,
            parcel_operations=self.parcel_operations,
            end_date=self.end_date,
        )
        self.df_observations = observations
        self.n_calls = 0

    def __call__(self, par_values, grad=None):
        """Runs the model and computes the objective function for given par_values.

        The input parameter 'grad' must be defined in the function call, but is only
        required for optimization methods where analytical gradients can be computed.
        """
        self.n_calls += 1
        # print(".", end="")
        # Run the model and collect output
        self.df_simulations = self.modelrerunner(par_values)
        # compute the differences by subtracting the DataFrames
        # Note that the dataframes automatically join on the index (dates) and column names
        df_differences = self.df_simulations - self.df_observations
        # display(df_differences[df_differences.DVS.notnull()].DVS)
        # With two features to assimilate, shall we do a weighted sum? Or do we not care if we prioritize DVS or LAI?
        # Now using simple addition
        # For DVS, calculate the error, ignoring NaN values
        error_dvs = (
            np.sqrt(np.mean(df_differences.DVS.dropna() ** 2))
            if df_differences.DVS.notna().any()
            else 0
        )

        # put threshold on LAI observations; ignore below 1.0
        lai_mask = (self.df_observations.LAI > 1.0).reindex(df_differences.index)
        lai_mask = lai_mask.astype("boolean")
        lai_mask = lai_mask.fillna(False)
        lai_mask = lai_mask.astype(bool)

        if lai_mask.any():
            error_lai = np.sqrt(np.mean(df_differences.LAI[lai_mask] ** 2))
        else:
            error_lai = 0

        obj_func = error_dvs + error_lai
        return obj_func


def optimize(objfunc_calc, p_mod, lower, upper, steps):
    # Calibration with optimizer
    opt = nlopt.opt(nlopt.LN_SBPLX, len(p_mod))
    opt.set_min_objective(objfunc_calc)
    opt.set_lower_bounds(lower)
    opt.set_upper_bounds(upper)
    opt.set_initial_step(steps)
    opt.set_maxeval(60)
    opt.set_ftol_rel(0.1)

    x = opt.optimize(list(p_mod.values()))
    np.set_printoptions(precision=2, suppress=True)
    print(f"\noptimum at {x}")
    print("minimum value = ", opt.last_optimum_value())
    print("result code = ", opt.last_optimize_result())
    print("With %i function calls" % objfunc_calc.n_calls)
    return x


def get_dvs_measurements() -> pd.DataFrame:
    dvs = [
        [datetime.date(2025, 6, 22), 1.0],
    ]
    df_dvs = pd.DataFrame(dvs, columns=["day", "DVS"])
    df_dvs = df_dvs.set_index("day")
    return df_dvs


def get_lai_measurements() -> pd.DataFrame:
    lai = [
        [datetime.date(2024, 9, 5), 0.15825132310945078],
        [datetime.date(2024, 9, 8), 0.17682688073471048],
        [datetime.date(2024, 9, 18), 0.23117254275813734],
        [datetime.date(2024, 9, 20), 0.1712588728224831],
        [datetime.date(2024, 9, 23), 0.17861915034643217],
        [datetime.date(2024, 9, 30), 0.17240729192555712],
        [datetime.date(2024, 10, 5), 0.23609693716500826],
        [datetime.date(2024, 10, 18), 0.27465273179123306],
        [datetime.date(2024, 10, 20), 0.2579485475217616],
        [datetime.date(2024, 10, 23), 0.2802925354855496],
        [datetime.date(2024, 10, 28), 0.45895537661481794],
        [datetime.date(2024, 11, 2), 0.5562276731616322],
        [datetime.date(2024, 11, 4), 0.5178871482182413],
        [datetime.date(2024, 11, 24), 0.5770340652618662],
        [datetime.date(2024, 12, 17), 0.5714625811879488],
        [datetime.date(2025, 2, 25), 0.2552811291428818],
        [datetime.date(2025, 3, 7), 0.36417558765800606],
        [datetime.date(2025, 3, 9), 0.24457786018278818],
        [datetime.date(2025, 3, 22), 0.3253683844596027],
        [datetime.date(2025, 3, 29), 0.3646847948339418],
        [datetime.date(2025, 4, 1), 0.37932862106457144],
        [datetime.date(2025, 4, 3), 0.4083808910338358],
        [datetime.date(2025, 4, 15), 1.1893672543054243],
        [datetime.date(2025, 4, 18), 1.3161926729891082],
        [datetime.date(2025, 4, 21), 0.7213238828026711],
        [datetime.date(2025, 4, 23), 2.016954573189108],
        [datetime.date(2025, 4, 25), 2.3080209461911125],
        [datetime.date(2025, 4, 26), 2.524632321108681],
        [datetime.date(2025, 5, 1), 3.0709683065614937],
        [datetime.date(2025, 5, 3), 2.4099542735805426],
        [datetime.date(2025, 5, 13), 2.824929257112378],
        [datetime.date(2025, 5, 21), 3.6935079398369384],
        [datetime.date(2025, 5, 25), 4.785199243686298],
        [datetime.date(2025, 6, 7), 5.443960387348966],
        [datetime.date(2025, 6, 14), 4.774278119381872],
        [datetime.date(2025, 6, 15), 5.256266299295123],
        [datetime.date(2025, 6, 17), 5.047314758147644],
        [datetime.date(2025, 7, 2), 3.8961900881086056],
        [datetime.date(2025, 7, 4), 4.077918580777735],
        [datetime.date(2025, 7, 12), 2.7178349839368043],
        [datetime.date(2025, 7, 17), 2.0032932850631533],
        [datetime.date(2025, 7, 22), 0.9215797273747098],
        [datetime.date(2025, 7, 27), 0.4908867288132728],
        [datetime.date(2025, 8, 1), 0.4111830569174219],
    ]

    results_lai = pd.DataFrame(lai, columns=["day", "LAI"])
    results_lai = results_lai.set_index("day")
    return results_lai


def get_default_calibration_parameters():
    return ["TSUM1", "TDWI", "SPAN"]


def calibrate(
    cropmodel: CropModel,
    measurements: pd.DataFrame = get_dvs_measurements(),
    parameters=get_default_calibration_parameters(),
    end_date=None,
):
    parameter_provider = cropmodel.parameterprovider
    weather_data_provider = cropmodel.weatherdataprovider
    agro_management = cropmodel._agromanagement
    parcel_operations = find_parcel_operations(cropmodel._locatedAtParcel) or None
    model_config = cropmodel.mconf.model_config_file
    parameters_mod = {x: parameter_provider._cropdata[x] for x in parameters}

    # Custom bounds for specific variables (multipliers)
    custom_bounds = {
        "SPAN": (0.95, 1.05),
        "TSUM1": (0.95, 1.05),
        "TDWI": (0.9, 1.1),
    }
    lower_bounds = []
    upper_bounds = []
    initial_steps = []

    for var, val in parameters_mod.items():
        low_mult, high_mult = custom_bounds.get(
            var, (0.75, 1.25)
        )  # default bounds if not specified
        lower_bounds.append(val * low_mult)
        upper_bounds.append(val * high_mult)
        initial_steps.append(val * 0.25)

    objfunc_calculator = ObjectiveFunctionCalculator(
        parameter_provider,
        weather_data_provider,
        agro_management,
        model_config,
        list(parameters_mod.keys()),
        measurements,
        parcel_operations=parcel_operations,
        end_date=end_date,
    )
    x = optimize(
        objfunc_calculator, parameters_mod, lower_bounds, upper_bounds, initial_steps
    )
    cropmodel.parameterprovider.clear_override()
    # Set overrides for the new parameter values
    for parname, value in zip(list(parameters_mod.keys()), x):
        if parname in cropmodel.parameterprovider._unique_parameters:
            cropmodel.parameterprovider.set_override(parname, value)
    updated_cropmodel = objfunc_calculator.modelrerunner.update_cropcropmodel(x)
    print_calibration_parameters(updated_cropmodel)

    return updated_cropmodel


def get_original_parameter(maps: dict, key: str):
    for data_map in maps[1:]:  # Skip the first dictionary
        if key in data_map:
            return data_map[key]
    raise KeyError(key)


def print_calibration_parameters(pcse_engine: pcse.engine.Engine):
    if pcse_engine.parameterprovider._override:
        orig_pars = {
            par: get_original_parameter(pcse_engine.parameterprovider._maps, par)
            for par in pcse_engine.parameterprovider._override.keys()
        }
        print(f"before calibration: {orig_pars}")
        print(f"after calibration: {pcse_engine.parameterprovider._override}")
