from typing import Union

import requests

import datetime

import openmeteo_requests
from openmeteo_sdk.Variable import Variable
from openmeteo_sdk.Aggregation import Aggregation

import requests_cache
from retry_requests import retry

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
    # df_daily['time'] = pd.to_datetime(df_daily['time'])
    df_daily.set_index('date', inplace=True)

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
    df_hourly['date'] = pd.to_datetime(df_hourly['date'])
    # Set time as the DataFrame index
    df_hourly.set_index('date', inplace=True)

    # Compute daily averages from hourly data:
    df_hourly_daily = df_hourly.groupby(df_hourly.index.date).mean()
    # Convert the index back to datetime
    df_hourly_daily.index = pd.to_datetime(df_hourly_daily.index)
    df_hourly_daily.rename(columns={
        'temperature_2m': 'TEMP',
        'windspeed_10m': 'WIND',
        'dewpoint_2m': 'dewpoint'
    }, inplace=True)

    # Merge on the date index (inner join to keep only days that exist in both)
    df_merged = pd.merge(df_daily, df_hourly_daily, left_index=True, right_index=True, how='inner')

    # Convert irradiation from MJ/m²/day to W/m²/day.
    df_merged['IRRAD'] = df_merged['IRRAD'] * 1e6

    # Convert precipitation from mm/day to cm/day.
    df_merged['RAIN'] = df_merged['RAIN'] * 0.1

    # Calculate vapor pressure (in hPa) from dewpoint (°C) using the formula:
    # e = 6.108 * exp((17.27 * T_d) / (T_d + 237.3))
    df_merged['VAP'] = (6.108 * np.exp((17.27 * df_merged['dewpoint']) / (df_merged['dewpoint'] + 237.3)))

    df_merged.drop(columns=['dewpoint'], inplace=True)

    df_merged['DAY'] = df_merged.index.date

    df_merged = df_merged[['TMIN', 'TMAX', 'TEMP', 'IRRAD', 'RAIN', 'WIND', 'VAP', 'DAY']]

    return df_merged


def extract_weather_data(response, params) -> dict:
    """
    Extracts daily and hourly weather data from the response object.

    Returns a dictionary with two keys, "daily" and "hourly", each of which is a dict.
    The expected keys for daily are:
        - "time"
        - "temperature_2m_min"
        - "temperature_2m_max"
        - "precipitation_sum"
        - "shortwave_radiation_sum"
    The expected keys for hourly are:
        - "time"
        - "temperature_2m"
        - "windspeed_10m"
        - "dewpoint_2m"

    Adjust the method names as required by your client.
    """
    # Extract daily data
    daily = response.Daily()
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        ),
    }
    daily_data["date"] = daily_data["date"].date

    for i, name in enumerate(params["daily"]):
        daily_data[name] = daily.Variables(i).ValuesAsNumpy()

    # Extract hourly data.
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
    }
    for i, name in enumerate(params["hourly"]):
        hourly_data[name] = hourly.Variables(i).ValuesAsNumpy()
    return {"daily": daily_data, "hourly": hourly_data}



class OpenMeteoWeatherProvider:
    """
    A weather provider that only needs a location (latitude and longitude)
    at initialization. When you call it with a single date or a date range,
    it fetches and returns the corresponding weather data.
    """

    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    def __init__(
            self,
            latitude: float,
            longitude: float,
            timezone: str ='UTC'):
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone

    def _fetch_data(self, start_date, end_date):
        """
        Internal method to fetch and prepare weather data for a given date range.
        Returns a DataFrame indexed by date.
        """
        url = "https://previous-runs-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": format_date(start_date),
            "end_date": format_date(end_date),
            "daily": ["temperature_2m_max","temperature_2m_min","precipitation_sum","shortwave_radiation_sum"],
            "hourly": ["temperature_2m","windspeed_10m","dewpoint_2m"],
            "timezone": self.timezone,
            "models": "bom_access_global",
        }

        # Use the client that has caching and retry support.
        response = OpenMeteoWeatherProvider.openmeteo.weather_api(url=url, params=params)
        # response = requests.get(url, params=params)
        # TODO: add loop for multiple locations. Now we have one.
        weather_api_object = response[0]

        raw_data = extract_weather_data(weather_api_object, params)

        df = prepare_weather_dataframe(raw_data)

        df['LAT'] = weather_api_object.Latitude()
        df['LON'] = weather_api_object.Longitude()
        df['ELEV'] = weather_api_object.Elevation()
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