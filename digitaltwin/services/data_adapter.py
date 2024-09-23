import datetime
from geojson import (
    Polygon,
    MultiLineString,
    Point,
    Feature,
    FeatureCollection,
)
from typing import Union

from sd_data_adapter.api import upload
import sd_data_adapter.models.agri_food as agri_food_model
import sd_data_adapter.models.device as device_model
import sd_data_adapter.models.autonomous_mobile_robot as autonomous_mobile_robot


def create_agripest(do_upload=True):
    model = agri_food_model.AgriPest(description="ascab")
    if do_upload:
        upload(model)
    return model


def create_crop(crop_type: str, do_upload=True) -> agri_food_model.AgriCrop:
    """
    Function to create SDM crop entity

    :param do_upload: upload to registry
    :param crop_type: String of generic crop type
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


def create_device(crop: agri_food_model.agriCrop, variable: str, do_upload=True):
    model = device_model.Device(controlledProperty=variable, controlledAsset=crop.id)
    if do_upload:
        upload(model)
    return model


def create_device_measurement(
    device: device_model.Device, date_observed: str, value: float, do_upload=True
):
    model = device_model.DeviceMeasurement(
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
    date: str = "20230401",
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
    soil: agri_food_model.AgriSoil,
    do_upload=True,
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
    :param do_upload: Bool to upload entity to Data Management Platform
    :return: AgriParcel SmartDataModel entity
    """
    model = agri_food_model.AgriParcel(
        location=location,
        area=area_parcel,
        hasAgriCrop=crop.id,
        hasAgriSoil=soil.id,
        description="initial_site",  # TODO placeholder description
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


def get_row_coordinates(
    parcel_loc: Union[FeatureCollection, agri_food_model.AgriParcel.location],
):
    multi_line_string_coords = []
    for feature in parcel_loc["features"]:
        if feature["geometry"]["type"] == "MultiLineString":
            coords = feature["geometry"]["coordinates"]
            multi_line_string_coords.append(coords)
    return multi_line_string_coords


def fill_database():
    wheat_crop = create_crop("wheat")
    soil = create_agrisoil()
    geo_feature_collection = generate_feature_collections(
        point=Point((52.0, 5.5)),  # for weather data (latitude, longitude)
        multilinestring=(MultiLineString()),  # for rows
        polygon=Polygon(),  # for parcel area
    )
    parcel = create_parcel(
        location=geo_feature_collection, area_parcel=20, crop=wheat_crop, soil=soil
    )
    fertilizer = create_fertilizer()
    fertilizer_application = create_fertilizer_application(
        parcel=parcel, product=fertilizer
    )
    device = create_device(crop=wheat_crop, variable="LAI")