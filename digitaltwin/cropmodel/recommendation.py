import os
import datetime

import numpy as np
import onnx
import onnxruntime as rt

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
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
    def __init__(self, agent_dir: str = AI_DIR) -> None:
        onnx_model_file = os.path.join(agent_dir, "model.onnx")
        self.cropgym_model = onnx.load(onnx_model_file)
        onnx.checker.check_model(self.cropgym_model)

        self.cropgym_ort_session = rt.InferenceSession(onnx_model_file)

    @staticmethod
    def process_crop_model_output(crop_model_output: dict):

        output = ...

        return output

    def __call__(self, crop_model_output: dict):

        obs = self.process_crop_model_output(crop_model_output)

        action, value, constraint, prob = self.cropgym_ort_session.run(None, {'obs': obs.astype(np.float32)})

        return action
