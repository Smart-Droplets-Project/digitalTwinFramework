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
        *args,
        **kwargs
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
) -> List[AscabModel]:
    results = []
    for parcel in parcels:
        crop = get_by_id(parcel.hasAgriCrop["object"])
        ascab = AscabModel(
            parcel_id=parcel.id,
            crop_id=crop.id,
            location=(get_weather_location(parcel)),  # (42.1620, 3.0924),
            dates=("2022-01-01", "2022-10-01"),
        )
        results.append(ascab)
    return results
