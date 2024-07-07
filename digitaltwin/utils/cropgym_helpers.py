import gymnasium as gym
import numpy as np


def get_action_space():
    return gym.spaces.Discrete(7)


def get_observation_space():
    return gym.spaces.Box(-np.inf, np.inf, shape=(len(get_obs_var_list()),))


def extract_digital_twin_obs(output):
    obs = np.zeros(get_observation_space().shape)

    days = [day["day"] for day in output]

    for i, o in enumerate(get_crop_vars()):
        obs[i] = output[-1][o]

    # TODO get from weather data provider
    # for j, w in enumerate(get_weather_vars()):
    #     obs

    return obs, days[-1].strftime('%m/%d/%Y')


def get_obs_var_list():
    return get_crop_vars() + get_weather_vars()


def get_crop_vars():
    return ["DVS", "TAGP", "LAI", "NuptakeTotal", "NAVAIL", "SM"]


def get_weather_vars():
    return ["RAIN", "TMIN", "TMAX", "IRRAD"]


def placeholder_recommendation(_):
    return get_action_space().sample()
