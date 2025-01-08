import datetime
from geojson import (
    Polygon,
    MultiLineString,
    Point,
    Feature,
    FeatureCollection,
)
from typing import Union, Optional, List

from ..cropmodel.crop_model import get_default_variables, get_dummy_measurements
from sd_data_adapter.api import upload
import sd_data_adapter.models.agri_food as agri_food_model
import sd_data_adapter.models.device as device_model
import sd_data_adapter.models.autonomous_mobile_robot as autonomous_mobile_robot


def create_agripest(do_upload=True):
    model = agri_food_model.AgriPest(description="ascab")
    if do_upload:
        upload(model)
    return model


def create_crop(
    crop_type: str,
    pest: Optional[agri_food_model.AgriPest] = None,
    do_upload=True,
) -> agri_food_model.AgriCrop:
    """
    Function to create SDM crop entity

    :param do_upload: upload to registry
    :param crop_type: String of generic crop type
    :param pest: AgriPest
    :return: AgriCrop entity
    """
    model = agri_food_model.AgriCrop(
        alternateName="Arminda" if crop_type == "wheat" else "",
        description=crop_type,
        dateCreated=str(datetime.datetime.now()),
        dateModified=str(datetime.datetime.now()),
        # TODO grab from somewhere
        plantingFrom=[
            "20221003",
            "20230820",
        ],  # List of planting and harvest date in YYYYMMDD in str
        **({"hasAgriPest": pest.id} if pest else {}),
    )
    if do_upload:
        upload(model)
    return model


def create_fertilizer(do_upload=True) -> agri_food_model.AgriProductType:
    model = agri_food_model.AgriProductType(type="fertilizer", name="Nitrogen")
    if do_upload:
        upload(model)
    return model


def create_agrisoil(do_upload=True):
    model = agri_food_model.AgriSoil(description="layered_soil")
    if do_upload:
        upload(model)
    return model


def create_device(controlled_asset: str, variable: str, do_upload=True):
    model = device_model.Device(
        controlledProperty=variable, controlledAsset=controlled_asset
    )
    if do_upload:
        upload(model)
    return model


def create_device_measurement(
    device: device_model.Device, date_observed: str, value: float, do_upload=True
):
    model = device_model.DeviceMeasurement(
        controlledProperty=device.controlledProperty,
        dateObserved=date_observed,
        numValue=value,
        refDevice=device.id,
    )
    if do_upload:
        upload(model)
    return model


def create_command_message(
    message_id, command, command_time, waypoints, do_upload=True
):
    model = autonomous_mobile_robot.CommandMessage(
        id=message_id,
        command=command,
        commandTime=command_time,
        waypoints=waypoints,
    )
    if do_upload:
        upload(model)
    return model


def create_fertilizer_application(
    parcel: agri_food_model.AgriParcel,
    product: agri_food_model.AgriProductType,
    quantity=60,
    date: str = "20230310",
    do_upload=True,
) -> agri_food_model.AgriParcelOperation:
    model = agri_food_model.AgriParcelOperation(
        operationType="fertiliser",
        hasAgriParcel=parcel.id,
        hasAgriProductType=product.id,
        plannedStartAt=date,
        quantity=quantity,
    )
    if do_upload:
        upload(model)
    return model


def create_parcel(
    location: Union[FeatureCollection, Point, MultiLineString, Polygon],
    area_parcel: float,
    crop: agri_food_model.AgriCrop,
    soil: Optional[agri_food_model.AgriSoil] = None,
    pest: Optional[agri_food_model.AgriPest] = None,
    do_upload=True,
    name: str = "Wheat Parcel Lithuania",
    address: str = "Uzumiskes, Kaunas, Lithuania",  # replaced e accent with normal e
    desciption: str = "Lithuania",
) -> agri_food_model.AgriParcel:
    """
    Function to initialize a parcel Entity.

    :param location: A geoJSON object describing multiple things:
                    - Point: A point to reference for weather/meteo provider
                    - MultiLineString: Multi lines describing tractor rows for the retrofitted tractor
                    - Polygon: Describes parcel area
                    - FeatureCollection: a collection of the above features
    :param area_parcel: A float of parcel area in hectares (ha)
    :param crop: A SmartDataModel crop entity
    :param soil: A SmartDataModel soil entity
    :param pest: A SmartDataModel pest entity
    :param do_upload: Bool to upload entity to Data Management Platform
    :param name: Name of the parcel
    :param address: address of the parcel
    :param description: description of the parcel
    :return: AgriParcel SmartDataModel entity
    """
    model = agri_food_model.AgriParcel(
        name=name,
        address=address,
        location=location,
        area=area_parcel,
        hasAgriCrop=crop.id,
        **({"hasAgriSoil": soil.id} if soil else {}),
        description=desciption,
    )
    if do_upload:
        upload(model)
    return model


def generate_feature_collections(
    point: Point = None,
    point_name: str = "weather location",
    multilinestring: MultiLineString = None,
    multilinestring_name: str = "rows",
    polygon: Polygon = None,
    polygon_name: str = "parcel area",
) -> FeatureCollection:
    ...
    point = point if point is not None else Point()
    point_feature = Feature(
        geometry=point, properties={"name": point_name}
    )  # check if any properties are needed

    multilinestring = (
        multilinestring if multilinestring is not None else MultiLineString()
    )
    multilinestring_feature = Feature(
        geometry=multilinestring, properties={"name": multilinestring_name}
    )

    polygon = polygon if polygon is not None else Polygon()
    polygon_feature = Feature(geometry=polygon, properties={"name": polygon_name})

    return FeatureCollection([point_feature, multilinestring_feature, polygon_feature])


def get_coordinates(
    parcel_loc: Union[FeatureCollection, agri_food_model.AgriParcel.location],
    feature_type: str = "MultiLineString",
):
    multi_line_string_coords = []
    for feature in parcel_loc["features"]:
        if feature["geometry"]["type"] == feature_type:
            coords = feature["geometry"]["coordinates"]
            multi_line_string_coords.append(coords)
    return multi_line_string_coords


def fill_database(variables: list[str] = get_default_variables()):
    wheat_crop = create_crop("wheat")
    soil = create_agrisoil()
    geo_feature_collection = generate_feature_collections(
        point=Point((52.0, 5.5)),  # for weather data (latitude, longitude)
        multilinestring=(MultiLineString()),  # for rows
        polygon=Polygon(
            [
                [
                    (23.9100961, 55.126702),
                    (23.9081435, 55.1224445),
                    (23.914731, 55.1215733),
                    (23.9158039, 55.1243586),
                    (23.9135079, 55.1248494),
                    (23.9100961, 55.126702),
                ]
            ]
        ),
    )
    parcel = create_parcel(
        location=geo_feature_collection, area_parcel=20, crop=wheat_crop, soil=soil
    )
    for variable in variables:
        device = create_device(
            controlled_asset=wheat_crop.id, variable=f"sim-{variable}"
        )

    obs_dict = {}
    for variable in variables:
        device = create_device(
            controlled_asset=wheat_crop.id, variable=f"obs-{variable}"
        )
        obs_dict[variable] = device

    dvs_measurements = get_dummy_measurements()
    for date, row in dvs_measurements.iterrows():
        dvs_measurement = create_device_measurement(
            device=obs_dict["DVS"],
            date_observed=date.isoformat(),
            value=row["DVS"],
        )


def fill_database_ascab():
    apple_pest = create_agripest()
    apple_crop = create_crop(crop_type="apple", pest=apple_pest)
    geo_feature_collection = generate_feature_collections(
        point=Point((42.16, 3.09)),  # for weather data (latitude, longitude)
        multilinestring=(MultiLineString()),  # for rows
        polygon=Polygon(
            [
                [
                    (3.0928589, 42.1628388),
                    (3.0927731, 42.1615902),
                    (3.0961419, 42.1613676),
                    (3.0962492, 42.1625684),
                    (3.0928589, 42.1628388),
                ]
            ]
        ),
    )

    parcel = create_parcel(
        location=geo_feature_collection,
        area_parcel=20,
        crop=apple_crop,
        name="Apple Orchard",
        address="Carrer Major, 1, 17143 Jafre, Girona, Spain",
        desciption="Serrater",
    )
    variables = ["LeafWetness"]
    for variable in variables:
        device = create_device(
            controlled_asset=apple_crop.id, variable=f"sim-{variable}"
        )


def generate_rec_message_id(day: str, parcel_id: str):
    return f"urn:ngsi-ld:CommandMessage:rec-{day}-'{parcel_id}'"


def get_recommendation_message(type: str, amount: float, day: str, parcel_id: str):
    return f"rec-{type}:{amount}:day:{day}:parcel_id:{parcel_id}"


def create_geojson_from_feature_collection(
    parcel_loc: Union[FeatureCollection, agri_food_model.AgriParcel.location],
    target_rate_value=0.5,
):
    features = []
    for feature in parcel_loc["features"]:
        if feature["geometry"]["type"] == "Polygon":
            # Modify the properties of the feature
            feature["properties"]["Target Rate"] = target_rate_value
            features.append(feature)

    feature_collection = FeatureCollection(features, name="output")
    return feature_collection


def split_device_dicts(
    devices: list[device_model.Device],
) -> tuple[
    dict[str, list[tuple[device_model.Device, device_model.DeviceMeasurement]]],
    dict[str, list[tuple[device_model.Device, device_model.DeviceMeasurement]]],
]:
    sim_dict, obs_dict = {}, {}

    for device in devices:
        key = device.controlledProperty
        value = (
            device,
            device_model.DeviceMeasurement(
                refDevice=device.id, controlledProperty=device.controlledProperty
            ),
        )
        # Add to the appropriate dictionary and handle multiple measurements
        if key.startswith("sim-"):
            sim_key = key.split("-", 1)[1]
            sim_dict.setdefault(sim_key, []).append(value)
        elif key.startswith("obs-"):
            obs_key = key.split("-", 1)[1]
            obs_dict.setdefault(obs_key, []).append(value)

    return sim_dict, obs_dict
