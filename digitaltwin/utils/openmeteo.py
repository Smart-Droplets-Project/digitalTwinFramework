from typing import Union

import requests

import datetime

import pandas as pd
import numpy as np

def format_date(d):
    """
    Converts a date or datetime object to a string in 'YYYY-MM-DD' format.
    If d is already a string, it is returned unchanged.
    """
    if isinstance(d, (datetime.date, datetime)):
        return d.strftime('%Y-%m-%d')
    return d


def fetch_open_meteo_weather(
        latitude: float,
        longitude: float,
        start_date: Union[str, datetime.date],
        end_date: Union[str, datetime.date],
        timezone: str = 'UTC'):
    """
    Fetches daily weather data from Open-Meteo.

    Parameters:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_date (str): Start date in 'YYYY-MM-DD' format or datetime.date object
        end_date (str): End date in 'YYYY-MM-DD' format or datetime.date object
        timezone (str): Timezone identifier.

    Returns:
        dict: Parsed JSON weather data.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": format_date(start_date),
        "end_date": format_date(end_date),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,shortwave_radiation_sum",
        "hourly": "windspeed_10m,dewpoint_2m",
        "timezone": timezone
    }

    response = requests.get(url, params=params)
    response.raise_for_status()  # will raise an error if the request failed
    data = response.json()
    return data


def prepare_weather_dataframe(weather_data):
    """
    Converts raw Open-Meteo weather data into a single daily DataFrame
    Currently, it is tailored to the inputs required by PCSE

    Parameters:
        weather_data (dict): JSON dictionary from Open-Meteo.

    Returns:
        DataFrame: Daily weather data with dates as index.
    """
    #  Process daily data 
    daily = weather_data.get('daily', {})
    df_daily = pd.DataFrame(daily)
    # Convert the 'time' column to datetime objects and set as index
    df_daily['time'] = pd.to_datetime(df_daily['time'])
    df_daily.set_index('time', inplace=True)

    # Rename daily columns for clarity
    df_daily.rename(columns={
        'temperature_2m_min': 'TMIN',
        'temperature_2m_max': 'TMAX',
        'precipitation_sum': 'RAIN',  # in mm/day
        'shortwave_radiation_sum': 'IRRAD'  # in MJ/m²/day
    }, inplace=True)

    #  Process hourly data
    hourly = weather_data.get('hourly', {})
    df_hourly = pd.DataFrame(hourly)
    # Convert hourly time to datetime objects
    df_hourly['time'] = pd.to_datetime(df_hourly['time'])
    # Set time as the DataFrame index
    df_hourly.set_index('time', inplace=True)

    # Compute daily averages from hourly data:
    df_hourly_daily = df_hourly.groupby(df_hourly.index.date).mean()
    # Convert the index back to datetime
    df_hourly_daily.index = pd.to_datetime(df_hourly_daily.index)
    df_hourly_daily.rename(columns={
        'windspeed_10m': 'WIND',
        'dewpoint_2m': 'dewpoint'
    }, inplace=True)

    # Merge on the date index (inner join to keep only days that exist in both)
    df_merged = pd.merge(df_daily, df_hourly_daily, left_index=True, right_index=True, how='inner')

    # Compute average temperature (tavg) as the mean of tmin and tmax.
    df_merged['TEMP'] = (df_merged['TMIN'] + df_merged['TMAX']) / 2.0

    # Convert irradiation from MJ/m²/day to W/m²/day.
    df_merged['IRRAD'] = df_merged['IRRAD'] * 1e6

    # Convert precipitation from mm/day to cm/day.
    df_merged['RAIN'] = df_merged['RAIN'] * 0.1

    # Calculate vapor pressure (in hPa) from dewpoint (°C) using the formula:
    # e = 6.112 * exp((17.67 * T_d) / (T_d + 243.5))
    df_merged['VAP'] = 6.112 * np.exp((17.67 * df_merged['dewpoint']) / (df_merged['dewpoint'] + 243.5))

    df_merged.drop(columns=['dewpoint'], inplace=True)

    df_merged['DAY'] = df_merged.index.date

    df_merged = df_merged[['TMIN', 'TMAX', 'TEMP', 'IRRAD', 'RAIN', 'WIND', 'VAP', 'DAY']]

    return df_merged


class OpenMeteoWeatherProvider:
    """
    A weather provider that only needs a location (latitude and longitude)
    at initialization. When you call it with a single date or a date range,
    it fetches and returns the corresponding weather data.
    """

    def __init__(
            self,
            latitude: float,
            longitude: float,
            timezone: str ='UTC',
            elevation: float = 0.0):
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.elevation = elevation  # in m

    def _fetch_data(self, start_date, end_date):
        """
        Internal method to fetch and prepare weather data for a given date range.
        Returns a DataFrame indexed by date.
        """
        raw_data = fetch_open_meteo_weather(self.latitude, self.longitude,
                                            start_date, end_date,
                                            self.timezone)
        df = prepare_weather_dataframe(raw_data)
        return df

    def __call__(self, target_date, end_date=None):
        """
        If end_date is not provided, returns weather data for a single day.
        If end_date is provided, returns weather data for the range between target_date and end_date.

        Parameters:
            target_date (date, datetime, or str): The starting date.
            end_date (date, datetime, or str, optional): The end date.

        Returns:
            dict: Weather data for a single day (if end_date is None) or
                  a dictionary with date strings as keys and weather data as values (if end_date is provided).
        """
        # When no end_date is provided, fetch data for a single day.
        if end_date is None:
            df = self._fetch_data(target_date, target_date)
            # Ensure the target_date is normalized.
            norm_date = pd.to_datetime(format_date(target_date))
            try:
                row = df.loc[norm_date]
            except KeyError:
                raise ValueError(f"No weather data available for date: {format_date(target_date)}")
            return row.to_dict()
        else:
            # Fetch data for the entire range.
            df = self._fetch_data(target_date, end_date)
            weather_dict = {}
            for dt, row in df.iterrows():
                key = dt.strftime('%Y-%m-%d')
                weather_dict[key] = row.to_dict()
            return weather_dict


if __name__ == '__main__':
    # Example of grabbing weather from Wageningen
    omwp = OpenMeteoWeatherProvider(51.98, 5.65)

    # Get weather for a single day.
    single_date = datetime.date(2025, 2, 3)
    weather_single = omwp(single_date)
    print(f"Weather on {single_date}:", weather_single)

    # Get weather for a date range.
    start = datetime.date(2025, 2, 1)
    end = datetime.date(2025, 2, 10)
    weather_range = omwp(start, end)
    print("Weather for the period:")
    for d, data in weather_range.items():
        print(f"{d}: {data}")