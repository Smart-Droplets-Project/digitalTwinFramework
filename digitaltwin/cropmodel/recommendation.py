import os
import datetime

import numpy as np
import onnx
import onnxruntime as rt

from digitaltwin.cropmodel.agromanagement import AgroManagement

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
            agromanagement: AgroManagement,
            agent_dir: str = AI_DIR,
    ) -> None:
        self.parcel_id = parcel_id

        onnx_model_file = os.path.join(agent_dir, "model.onnx")
        self.cropgym_model = onnx.load(onnx_model_file)
        onnx.checker.check_model(self.cropgym_model)

        self.cropgym_ort_session = rt.InferenceSession(onnx_model_file)

        self.agromanagement = agromanagement

        self.action_freq = 0
        self.action_history = 0

    def process_crop_model_output(self, crop_model_output: dict):
        '''
        Process WOFOST output for cropgym agent

        :return: it should return this list
        ["DVS", "LAI", "TAGP", "WSO", "NAVAIL", "NuptakeTotal", 'week', 'Naction', 'action_history',
        "IRRAD", "TMIN", "RAIN"]. The last three variables are for the last week.
        '''
        list_output = [crop_model_output[key] for key in self.default_variable_list()]

        week = self.get_week(self.get_latest_date(crop_model_output))

        list_output.append(week)
        list_output.append(self.action_freq)
        list_output.append(self.action_history)

        weather = self.get_weather(crop_model_output)

        output = list_output + weather

        output = np.array(output)

        return output

    def update_action(self, action):
        self.action_freq += 1
        self.action_history += action

    def __call__(self, crop_model_output: dict):

        date = self.get_latest_date(crop_model_output)

        obs = self.process_crop_model_output(crop_model_output, date)

        action, value, constraint, prob = self.cropgym_ort_session.run(None, {'obs': obs.astype(np.float32)})

        if action > 0:
            self.update_action(action)

        return action

    def get_week(self, date_now: datetime.date):

        start_date = self.agromanagement.get_start_date

        delta = date_now - start_date
        week = delta.days // 7

        return week


    def get_weather(self, crop_model_output: dict) -> list:
        '''
        :return: return weekly weather. If not available, fill with NaNs.
        '''

        # Need to get from weather data provider under CropModel object

        return []

    @staticmethod
    def get_latest_date(crop_model_output: dict):
        date = crop_model_output[-1]["day"]
        return date

    @staticmethod
    def default_variable_list() -> list:
        return ["DVS", "LAI", "TAGP", "TWSO", "NAVAIL", "NuptakeTotal"]

    @staticmethod
    def weather_variables() -> list:
        return ["IRRAD", "TMIN", "RAIN"]
