from datetime import date
from ascab.env.env import AScabEnv
from typing import Optional, List
from sd_data_adapter.models.smartDataModel import Relationship
import sd_data_adapter.models.agri_food as agri_food_model
from sd_data_adapter.api import get_by_id


class AscabModel(AScabEnv):
    def __init__(
        self,
        parcel_id: Optional[str] = None,
        crop_id: Optional[str] = None,
        pest_id: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Initialize attributes only if they are not already set
        self._locatedAtParcel: Optional[Relationship] = (
            parcel_id
            if parcel_id is not None
            else getattr(self, "_locatedAtParcel", None)
        )
        self._isAgriCrop: Optional[Relationship] = (
            crop_id if crop_id is not None else getattr(self, "_isAgriCrop", None)
        )
        self._isAgriPest: Optional[Relationship] = (
            pest_id if pest_id is not None else getattr(self, "_isAgriPest", None)
        )

    def __str__(self):
        return (
            f"AscabModel for {self._isAgriPest}, "
            f"located at parcel: {self._locatedAtParcel}, "
            f"agri crop: {self._isAgriCrop}, "
            f"dates: {self.dates}, "
        )


def get_weather_location(
    parcel: agri_food_model.AgriParcel,
):
    location = None
    for feature in parcel.location["features"]:
        if (
            feature["properties"]["name"] == "weather location"
            and feature["geometry"]["type"] == "Point"
        ):
            location = feature["geometry"]["coordinates"]
    return location


def create_digital_twins(
    parcels: List[agri_food_model.AgriParcel],
    end_date: date = None,
) -> List[AscabModel]:
    end_date = end_date or date(2022, 10, 1)
    year = end_date.year

    start_date = date(year, 1, 1)
    dates = (start_date.isoformat(), end_date.isoformat())

    results = []
    for parcel in parcels:
        crop = get_by_id(parcel.hasAgriCrop["object"])
        ascab = AscabModel(
            parcel_id=parcel.id,
            crop_id=crop.id,
            pest_id=crop.hasAgriPest["object"],
            location=(get_weather_location(parcel)),
            dates=dates,
            biofix_date="March 10",
            budbreak_date="March 10",
        )
        results.append(ascab)
    return results
