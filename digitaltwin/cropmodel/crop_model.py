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
from digitaltwin.cropmodel.recommendation import fill_it_up

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
    # TODO: Temp code, add Lithuanian parameters statically or do it dynamically
    if "initial_site" in site_parameters:
        site_parameters = yaml.safe_load(
            open(os.path.join(CONFIGS_DIR, "site", f"{site_parameters}"))
        )
    return site_parameters


def get_default_variables():
    # TODO: make sure that the following aligns with OUTPUT_VARS in Wofost81_NWLP_MLWB_SNOMIN.conf
    return ["DVS", "LAI", "TAGP", "TWSO", "NAVAIL", "NuptakeTotal"]


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

    def __init__(self, params, wdp, agro, model_config, parameters_calibration):
        self.params = params
        self.wdp = wdp
        self.agro = agro
        self.model_config = model_config
        self.parameters_calibration = parameters_calibration

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
    @staticmethod
    def run(model_mod):
        """Make one time step of the simulation.
        """

        # Update timer
        model_mod.day, delt = model_mod.timer()

        # State integration
        model_mod.integrate(model_mod.day, delt)

        # Driving variables
        model_mod.drv = model_mod._get_driving_variables(model_mod.day)

        # Agromanagement decisions
        model_mod.agromanager(model_mod.day, model_mod.drv)

        action = fill_it_up(model_mod.get_output()[-1])

        if action > 0:
            model_mod._send_signal(signal=pcse.signals.apply_n_snomin,
                                   amount=action,
                                   application_depth=10.,
                                   cnratio=0.,
                                   f_orgmat=0.,
                                   f_NH4N=0.5,
                                   f_NO3N=0.5,
                                   initial_age=0,
                                   )

        # Rate calculation
        model_mod.calc_rates(model_mod.day, model_mod.drv)

        if model_mod.flag_terminate is True:
            model_mod._terminate_simulation(model_mod.day)

        return model_mod


class ObjectiveFunctionCalculator(object):
    """Computes the objective function.

        This class runs the simulation model with given parameter values and returns the objective
        function as the sum of squared difference between observed and simulated.
    ."""

    def __init__(
            self, params, wdp, agro, model_config, parameters_calibration, observations
    ):
        self.modelrerunner = ModelRerunner(
            params, wdp, agro, model_config, parameters_calibration
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
        error_dvs = np.sqrt(np.mean(df_differences.DVS ** 2))
        error_lai = np.sqrt(np.mean(df_differences.LAI ** 2))

        obj_func = (error_dvs + error_lai)
        # print(f'{par_values} {obj_func}')
        return obj_func


def optimize(objfunc_calc, p_mod, lower, upper, steps):
    # Calibration with optimizer
    opt = nlopt.opt(nlopt.LN_SBPLX, len(p_mod))
    opt.set_min_objective(objfunc_calc)
    opt.set_lower_bounds(lower)
    opt.set_upper_bounds(upper)
    opt.set_initial_step(steps)
    opt.set_maxeval(20)
    opt.set_ftol_rel(0.1)

    x = opt.optimize(list(p_mod.values()))
    np.set_printoptions(precision=2, suppress=True)
    print(f"\noptimum at {x}")
    print("minimum value = ", opt.last_optimum_value())
    print("result code = ", opt.last_optimize_result())
    print("With %i function calls" % objfunc_calc.n_calls)
    return x


def get_dummy_measurements() -> pd.DataFrame:
    # DVS as published on MARS website
    MARS = [
        [datetime.date(2022, 12, 31), 0.03],
        [datetime.date(2023, 1, 1), 0.03],
        [datetime.date(2023, 2, 1), 0.04],
        [datetime.date(2023, 3, 1), 0.08],
        [datetime.date(2023, 3, 15), 0.15],
        [datetime.date(2023, 4, 1), 0.28],
        [datetime.date(2023, 4, 10), 0.38],
        [datetime.date(2023, 5, 1), 0.62],
        [datetime.date(2023, 5, 20), 1.0],
        [datetime.date(2023, 6, 1), 1.24],
        [datetime.date(2023, 6, 15), 1.6],
        [datetime.date(2023, 6, 20), 1.7],
        [datetime.date(2023, 7, 1), 1.93],
        [datetime.date(2023, 7, 10), 2.0],
    ]
    Results_MARS = pd.DataFrame(MARS, columns=["day", "DVS"])
    Results_MARS = Results_MARS.set_index("day")
    return Results_MARS


def get_dummy_lai_measurements() -> pd.DataFrame:
    lai = [
        [datetime.date(2023, 4, 15), 2.91],
        [datetime.date(2023, 4, 27), 3.01],
        [datetime.date(2023, 6, 3), 2.91],
    ]
    results_lai = pd.DataFrame(lai, columns=["day", "LAI"])
    results_lai = results_lai.set_index("day")
    return results_lai


def get_default_calibration_parameters():
    # SLATB for LAI is a table. Some hacky fixes has been implenmented for now
    return ["TSUM1", "TSUM2", "DLC", "DLO", "TSUMEM"]


def calibrate(
        cropmodel: CropModel,
        measurements: pd.DataFrame = get_dummy_measurements(),
        parameters=get_default_calibration_parameters(),
):
    # DVS variables and sowing
    parameter_provider = cropmodel.parameterprovider
    weather_data_provider = cropmodel.weatherdataprovider
    agro_management = cropmodel._agromanagement
    model_config = cropmodel.mconf.model_config_file
    parameters_mod = {x: parameter_provider._cropdata[x] for x in parameters}
    lower_bounds = [i * 0.2 for i in parameters_mod.values()]
    upper_bounds = [i * 1.2 for i in parameters_mod.values()]
    initial_steps = [i * 0.1 for i in parameters_mod.values()]
    objfunc_calculator = ObjectiveFunctionCalculator(
        parameter_provider,
        weather_data_provider,
        agro_management,
        model_config,
        list(parameters_mod.keys()),
        measurements,
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
