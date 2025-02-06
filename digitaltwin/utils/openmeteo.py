import os
import datetime

from typing import Union

import pandas as pd
import numpy as np
from math import log10

from pcse.base import WeatherDataProvider, WeatherDataContainer
from pcse.util import ea_from_tdew, reference_ET, check_angstromAB, wind10to2
from pcse.exceptions import PCSEError
from pcse.settings import settings


def format_date(date: Union[str, datetime.date]):
    """
    Converts a date or datetime object to a string in 'YYYY-MM-DD' format.
    If d is already a string, it is returned unchanged.
    """
    if isinstance(date, (datetime.date, datetime)):
        return date.strftime('%Y-%m-%d')
    return date


class OpenMeteoWeatherProvider(WeatherDataProvider):
    """
    A weather provider that only needs a location (latitude and longitude)
    at initialization. When you call it with a single date or a date range,
    it fetches and returns the corresponding weather data.
    """
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    daily_variables = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "shortwave_radiation_sum"]
    hourly_variables = ["temperature_2m", "windspeed_10m", "dewpoint_2m"]

    angstA = 0.29
    angstB = 0.49

    #  List of forecast and historical weather models for OpenMeteo
    #  Comments show coverage and spatial resolution
    #  TODO: make dict of model name and earliest start date
    dict_forecast_models = {
        'arpae_cosmo_5m': datetime.date(2024, 2, 2),  # europe, 5m
        'bom_access_global': datetime.date(2024, 1, 19),  # global, 0.15deg
        'gem_seamless': datetime.date(2022, 11, 24),  # global, 0.15deg
        'jma_gsm': datetime.date(2016, 1, 1),  # global, 0.5deg
        'icon_seamless': datetime.date(2022, 11, 25),  # global, 11km
        'ecmwf_ifs025': datetime.date(2024, 2, 4),  # global, 0.25deg
        'knmi_seamless': datetime.date(2024, 7, 2),  # europe, 2.5km
        'meteofrance_seamless': datetime.date(2024, 1, 3),  # global 0.25deg
        'gfs_seamless': datetime.date(2021, 3, 24),  # global 0.11deg
        'ukmo_seamless': datetime.date(2022, 3, 2),  # global, 0.09deg/10km
    }

    dict_historical_models = {
        'era5': datetime.date(1941, 1, 1),  # global, 0.25deg
        'era5_land': datetime.date(1951, 1, 1),  # global, 0.1deg
        'ecmwf_ifs': datetime.date(2017, 1, 1),  # global, 9km
        'cerra': datetime.date(1986, 1, 1),  # global, 5km
    }

    delay_historical_models = {
        'era5': 5,  # global, 0.25deg
        'era5_land': 5,  # global, 0.1deg
        'ecmwf_ifs': 2, # global, 9km
        'cerra': 0,
    }

    def __init__(
            self,
            latitude: float,
            longitude: float,
            timezone: str = 'UTC',
            openmeteo_model: str = 'gfs_seamless',
            start_date: Union[str, datetime.date] = None,
            ETmodel: str = "PM",
            force_update: bool = False,
    ):
        WeatherDataProvider.__init__(self)

        self.model = openmeteo_model
        self.ETmodel = ETmodel
        self.start_date = start_date
        if self.start_date is None and self.model in self.dict_forecast_models:
            self.start_date = self.dict_forecast_models[self.model]
        elif self.start_date is None and self.model in self.dict_historical_models:
            self.start_date = self.dict_historical_models[self.model]

        if latitude < -90 or latitude > 90:
            msg = "Latitude should be between -90 and 90 degrees."
            raise ValueError(msg)
        if longitude < -180 or longitude > 180:
            msg = "Longitude should be between -180 and 180 degrees."
            raise ValueError(msg)

        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.timezone = timezone

        self._check_cache(force_update)

    def _check_cache(self, force_update: bool = False):
        # Check for existence of a cache file
        cache_file = self._find_cache_file(self.latitude, self.longitude)
        if cache_file is None or force_update is True:
            msg = "No cache file or forced update, getting data from OpenMeteo Power."
            self.logger.debug(msg)
            # No cache file, we really have to get the data from the open-meteo server
            self._fetch_data(self.start_date)
            return

        # get age of cache file, if age < 90 days then try to load it. If loading fails retrieve data
        # from the OpenMeteo server .
        r = os.stat(cache_file)
        cache_file_date = datetime.date.fromtimestamp(r.st_mtime)
        age = (datetime.date.today() - cache_file_date).days
        if age < 90:
            msg = "Start loading weather data from cache file: %s" % cache_file
            self.logger.debug(msg)

            status = self._load_cache_file()
            if status is not True:
                msg = "Loading cache file failed, reloading data from OpenMeteo."
                self.logger.debug(msg)
                # Loading cache file failed!
                self._fetch_data(self.start_date)
        else:
            # Cache file is too old. Try loading new data from OpenMeteo
            try:
                msg = "Cache file older then 90 days, reloading data from OpenMeteo."
                self.logger.debug(msg)
                self._fetch_data(self.start_date)
            except Exception as e:
                msg = ("Reloading data from OpenMeteo failed, reverting to (outdated) " +
                       "cache file")
                self.logger.debug(msg)
                status = self._load_cache_file()
                if status is not True:
                    msg = "Outdated cache file failed loading."
                    raise PCSEError(msg)

    def _fetch_data(self, start_date):
        """
        Internal method to fetch and prepare weather data for a given date range.
        Returns a DataFrame indexed by date.
        """
        url = self._get_url(previous_runs=True)
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": format_date(start_date),
            "end_date": format_date(self._get_end_date()),
            "daily": self.daily_variables,
            "hourly": self.hourly_variables,
            "timezone": self.timezone,
            "models": self.model,
        }

        # Use the client that has caching and retry support.
        response = OpenMeteoWeatherProvider.openmeteo.weather_api(url=url, params=params)
        # response = requests.get(url, params=params)
        # TODO: add loop for multiple locations. Now we have one.
        weather_api_object = response[0]

        self.elevation = weather_api_object.Elevation()

        raw_data = self._extract_weather_data(weather_api_object, params)

        df = self._prepare_weather_dataframe(raw_data)

        self._make_WeatherDataContainers(df.to_dict(orient='records'))

        cache_filename = self._get_cache_filename(self.latitude, self.longitude)
        self._dump(cache_filename)

    def _find_cache_file(self, latitude, longitude):
        """Try to find a cache file for given latitude/longitude.

        Returns None if the cache file does not exist, else it returns the full path
        to the cache file.
        """
        cache_filename = self._get_cache_filename(latitude, longitude)
        if os.path.exists(cache_filename):
            return cache_filename
        else:
            return None

    def _get_cache_filename(self, latitude, longitude):
        """Constructs the filename used for cache files given latitude and longitude

        The latitude and longitude is coded into the filename by truncating on
        0.1 degree. So the cache filename for a point with lat/lon 52.56/-124.78 will be:
        OpenMeteoWeatherDataProvider_LAT00525_LON-1247.cache
        """

        fname = "%s_LAT%05i_LON%05i_%s.cache" % (self.__class__.__name__,
                                                 int(latitude * 10), int(longitude * 10),
                                                 self.model[:5])
        cache_filename = os.path.join(settings.METEO_CACHE_DIR, fname)
        return cache_filename

    def _write_cache_file(self):
        """Writes the meteo data from OpenMeteo Power to a cache file.
        """
        cache_filename = self._get_cache_filename(self.latitude, self.longitude)
        try:
            self._dump(cache_filename)
        except (IOError, EnvironmentError) as e:
            msg = "Failed to write cache to file '%s' due to: %s" % (cache_filename, e)
            self.logger.warning(msg)

    def _load_cache_file(self):
        """Loads the data from the cache file. Return True if successful.
        """
        cache_filename = self._get_cache_filename(self.latitude, self.longitude)
        try:
            self._load(cache_filename)
            msg = "Cache file successfully loaded."
            self.logger.debug(msg)
            return True
        except (IOError, EnvironmentError, EOFError) as e:
            msg = "Failed to load cache from file '%s' due to: %s" % (cache_filename, e)
            self.logger.warning(msg)
            return False

    def _make_WeatherDataContainers(self, recs):
        """Create a WeatherDataContainers from recs, compute ET and store the WDC's.
        """

        for rec in recs:
            # Build weather data container from dict 't'
            wdc = WeatherDataContainer(**rec)

            # add wdc to dictionary for this date
            self._store_WeatherDataContainer(wdc, wdc.DAY)

    def _prepare_weather_dataframe(self, weather_data):
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

        # Convert wind from 10m to 2m
        df_merged['WIND'] = wind10to2(df_merged['WIND'])

        # Calculate vapor pressure (in hPa) from dewpoint (°C) using the formula:
        # e = 6.108 * exp((17.27 * T_d) / (T_d + 237.3))
        df_merged['VAP'] = (6.108 * np.exp((17.27 * df_merged['dewpoint']) / (df_merged['dewpoint'] + 237.3)))

        df_merged.drop(columns=['dewpoint'], inplace=True)

        df_merged['DAY'] = df_merged.index.date

        df_merged['LAT'] = self.latitude
        df_merged['LON'] = self.longitude
        df_merged['ELEV'] = self.elevation

        df_merged = df_merged[['TMIN', 'TMAX', 'TEMP', 'IRRAD', 'RAIN', 'WIND', 'VAP', 'DAY', 'LAT', 'LON', 'ELEV']]

        E0_list = []
        ES0_list = []
        ET0_list = []

        for row in df_merged.itertuples():
            E0, ES0, ET0 = reference_ET(row.DAY, row.LAT, row.ELEV, row.TMIN,
                                        row.TMAX, row.IRRAD,
                                        row.VAP, row.WIND,
                                        self.angstA, self.angstB, self.ETmodel)

            #  convert to cm/day
            E0_list.append(E0 / 10.)
            ES0_list.append(ES0 / 10.)
            ET0_list.append(ET0 / 10.)

        df_merged["E0"] = E0_list
        df_merged["ES0"] = ES0_list
        df_merged["ET0"] = ET0_list

        df_merged = df_merged[
            [
                'TMIN',
                'TMAX',
                'TEMP',
                'IRRAD',
                'RAIN',
                'WIND',
                'VAP',
                'DAY',
                'LAT',
                'LON',
                'ELEV',
                'E0',
                'ES0',
                'ET0'
            ]
        ]

        return df_merged

    def _extract_weather_data(self, response, params) -> dict:
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


    def _get_url(self, previous_runs: bool = False) -> str:
        if self.model in self.dict_forecast_models and previous_runs is True:
            return "https://previous-runs-api.open-meteo.com/v1/forecast"
        elif self.model in self.dict_forecast_models and previous_runs is False:
            return "https://api.open-meteo.com/v1/forecast"
        elif self.model in self.dict_historical_models:
            return "https://archive-api.open-meteo.com/v1/archive"
        else:
            raise ValueError("Model not found. Check model availability.")


    def _get_end_date(self):
        if self.model in self.dict_forecast_models:
            return datetime.date.today() + datetime.timedelta(days=7)
        elif self.model in self.dict_historical_models:
            return datetime.date.today() - datetime.timedelta(days=self.delay_historical_models[self.model])
        else:
            raise ValueError("Model not found. Check model availability.")


if __name__ == '__main__':
    # Example of grabbing weather from Wageningen
    omwp = OpenMeteoWeatherProvider(51.98, 5.65)

    # Get weather for a single day.
    single_date = datetime.date(2024, 5, 15)
    weather_single = omwp(single_date)
    print(f"Weather on {single_date}:", weather_single)
