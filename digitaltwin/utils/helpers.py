import datetime


def get_nested_value(data, target_key):
    """
    Recursively search for the target_key in a nested dictionary and return its value.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                return value
            result = get_nested_value(value, target_key)
            if result is not None:
                return result
    return None


def get_weather(crop_model_output: dict,
                weather_provider,
                weather_variables: list = ["IRRAD", "TMIN", "RAIN"]) -> list:
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

    weather_data = [weather_provider(day) for day in days]

    weather_obs = [getattr(wdc, var) for wdc in weather_data for var in weather_variables]

    return weather_obs