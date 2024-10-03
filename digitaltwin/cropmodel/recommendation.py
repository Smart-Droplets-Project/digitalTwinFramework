import datetime
import numpy as np
import pandas as pd

from digitalTwinFramework.digitaltwin.cropmodel.crop_model import CropModel


def standard_practice(crop_model_output: dict):
    year = crop_model_output["day"].year
    fertilization_dates = [datetime.date(year, 4, 1), datetime.date(year, 5, 1)]
    result = 0
    if crop_model_output["day"] in fertilization_dates:
        result = 60
    return result


def convert_wofost_to_obs(crop_model: CropModel):

    # features to convert and flatten
    crop_variables = ["DVS", "TAGP", "LAI", "NuptakeTotal", "TRA", "NO3", "NH4", "WC", "RFTRA", "WSO", "NLOSSCUM",
                      'RNO3DEPOSTT', 'RNH4DEPOSTT', 'NamountSO', 'week', 'Naction']
    action_features = ["action_history"]
    weather_variables = ["IRRAD", "TMIN", "RAIN"]
    timestep = 7

    # grab output
    crop_model_output = crop_model.get_output()

    # grab weather data
    weather_data = [crop_model.weatherdataprovider(day) for day in crop_model_output[-timestep]['day']]

    # intialize obs
    obs = np.zeros(len(crop_variables) + len(action_features) + len(weather_variables))

    for i, feature in enumerate(crop_variables):
        if feature in ['NH4', 'NO3']:
            obs[i] = sum(crop_model_output[-1][feature]) / 1e-4
        elif feature in ['SM', 'WC']:
            obs[i] = np.mean(crop_model_output[-1][feature])
        elif feature in ['RNO3DEPOSTT', 'RNH4DEPOSTT']:
            obs[i] = crop_model_output[-1][feature] / 1e-4
        elif feature in ['week']:
            obs[i] = np.ceil(len(crop_model_output.values()) / timestep)
        elif feature in ['Naction']:
            obs[i] = crop_model.action_counter
        else:
            obs[i] = crop_model_output[-1][feature]

    for i, feature in enumerate(action_features):
        j = len(crop_variables) + i
        obs[j] = crop_model_output[-1]['RNH4AMTT'] / 1e-3 + crop_model_output[-1]['RNO3AMTT'] / 1e-3

    for d in range(timestep):
        for i, feature in enumerate(weather_variables):
            j = d * len(weather_variables) + len(crop_variables) + len(action_features) + i
            obs[j] = getattr(weather_data[d], feature)

    return obs
