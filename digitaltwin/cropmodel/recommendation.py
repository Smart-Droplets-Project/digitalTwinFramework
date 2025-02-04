import os
import datetime

import numpy as np
import onnx
import onnxruntime as rt

from digitaltwin.utils.helpers import get_nested_value

AI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "configs",
    "AI_fertilizer_agent",
)


def fill_it_up(crop_model_output: dict):
    year = crop_model_output["day"].year
    fertilization_dates = [datetime.date(year, 4, 1), datetime.date(year, 5, 1)]
    result = 0
    if crop_model_output["day"] in fertilization_dates:
        navail = crop_model_output["NAVAIL"]
        result = max(0, 80 - navail)
    return result


class CropgymAgent:
    def __init__(
        self,
        parcel_id: str,
        agromanagement: list,
        agent_dir: str = AI_DIR,
        timestep: int = 7,
    ) -> None:
        self.parcel_id = parcel_id

        onnx_model_file = os.path.join(agent_dir, "model.onnx")
        self.cropgym_model = onnx.load(onnx_model_file)
        onnx.checker.check_model(self.cropgym_model)

        self.cropgym_ort_session = rt.InferenceSession(onnx_model_file)

        self.agromanagement = agromanagement

        self.timestep = timestep
        self.action_freq = 0
        self.action_history = 0

    def process_crop_model_output(self, crop_model_output: dict, weather: list):
        """
        Process WOFOST output for cropgym agent

        :return: it should return this list
        ["DVS", "LAI", "TAGP", "WSO", "NAVAIL", "NuptakeTotal", 'week', 'Naction', 'action_history',
        "IRRAD", "TMIN", "RAIN"]. The last three variables are for the last week.
        """
        list_output = [crop_model_output[-1][key] for key in self.default_variable_list]

        week = self.get_week(self.get_latest_date(crop_model_output))

        list_output.append(week)
        list_output.append(self.action_freq)
        list_output.append(self.action_history)

        output = [list_output + weather]  # model requires nested list as obs

        output = np.array(output)

        return output

    def update_action(self, action):
        self.action_freq += 1
        self.action_history += action

    def __call__(self, crop_model_output: dict, weather: list):

        obs = self.process_crop_model_output(crop_model_output, weather)

        action, value, constraint, prob = self.cropgym_ort_session.run(
            None, {"obs": obs.astype(np.float32)}
        )

        if isinstance(action, np.ndarray):
            action = float(action[0])

        if action > 0:
            self.update_action(action)

        return action

    def get_week(self, date_now: datetime.date):

        start_date = get_nested_value(self.agromanagement[0], "crop_start_date")

        delta = date_now - start_date
        week = delta.days // 7

        return week

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
