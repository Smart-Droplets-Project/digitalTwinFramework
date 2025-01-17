import os
import datetime

import numpy as np
import onnx
import onnxruntime as rt

from digitaltwin.cropmodel.agromanagement import AgroManagement
from digitaltwin.utils.helpers import get_nested_value

import pcse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)

# subject to change
AI_DIR = os.path.join(os.path.dirname(SRC_DIR), 'aiRecommender', 'AI_fertilizer_agent')


def fill_it_up(crop_model_output: dict):
    year = crop_model_output["day"].year
    fertilization_dates = [datetime.date(year, 4, 1), datetime.date(year, 5, 1)]
    result = 0
    if crop_model_output["day"] in fertilization_dates:
        navail = crop_model_output["NAVAIL"]
        result = max(0, 80 - navail)
    return result


# TODO: Work in progress
class CropgymAgent:
    def __init__(
            self,
            parcel_id: str,
            weather_provider: pcse.input.NASAPowerWeatherDataProvider,
            agromanagement: list,
            agent_dir: str = AI_DIR,
            timestep: int = 7,
    ) -> None:
        self.parcel_id = parcel_id

        self.weather_provider = weather_provider

        onnx_model_file = os.path.join(agent_dir, "model.onnx")
        self.cropgym_model = onnx.load(onnx_model_file)
        onnx.checker.check_model(self.cropgym_model)

        self.cropgym_ort_session = rt.InferenceSession(onnx_model_file)

        self.agromanagement = agromanagement

        self.timestep = timestep
        self.action_freq = 0
        self.action_history = 0

    def process_crop_model_output(self, crop_model_output: dict):
        '''
        Process WOFOST output for cropgym agent

        :return: it should return this list
        ["DVS", "LAI", "TAGP", "WSO", "NAVAIL", "NuptakeTotal", 'week', 'Naction', 'action_history',
        "IRRAD", "TMIN", "RAIN"]. The last three variables are for the last week.
        '''
        list_output = [crop_model_output[-1][key] for key in self.default_variable_list]

        week = self.get_week(self.get_latest_date(crop_model_output))

        list_output.append(week)
        list_output.append(self.action_freq)
        list_output.append(self.action_history)

        weather = self.get_weather(crop_model_output)

        output = [list_output + weather]  # model requires nested list as obs

        output = np.array(output)

        return output

    def update_action(self, action):
        self.action_freq += 1
        self.action_history += action

    def __call__(self, crop_model_output: dict):

        obs = self.process_crop_model_output(crop_model_output)

        action, value, constraint, prob = self.cropgym_ort_session.run(None, {'obs': obs.astype(np.float32)})

        if isinstance(action, np.ndarray):
            action = float(action[0])

        if action > 0:
            self.update_action(action)

        return action

    def get_week(self, date_now: datetime.date):

        start_date = get_nested_value(self.agromanagement[0], 'crop_start_date')

        delta = date_now - start_date
        week = delta.days // 7

        return week


    def get_weather(self, crop_model_output: dict) -> list:
        '''
        :return: return weather variables of the last week
        '''

        # days so far
        days = [day['day'] for day in crop_model_output]

        # Check if there are at least 7 days for the RL obs
        if len(days) < 7:
            earliest_date = days[0]

            # Add dates that predate the earliest date
            while len(days) < 7:
                earliest_date -= datetime.timedelta(days=1)
                days.insert(0, earliest_date)
        else:
            days = days[-7:]

        weather_data = [self.weather_provider(day) for day in days]

        weather_obs = [getattr(wdc, var) for wdc in weather_data for var in self.weather_variables]

        return weather_obs

    @staticmethod
    def get_latest_date(crop_model_output: dict):
        date = crop_model_output[-1]["day"]
        return date

    @property
    def default_variable_list(self) -> list:
        return ["DVS", "LAI", "TAGP", "TWSO", "NAVAIL", "NuptakeTotal"]

    @property
    def weather_variables(self) -> list:
        return ["IRRAD", "TMIN", "RAIN"]
