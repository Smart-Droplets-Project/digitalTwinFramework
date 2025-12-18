import argparse
import datetime
import pickle

from sd_data_adapter.client import DAClient
from sd_data_adapter.api import search, upsert
from sd_data_adapter.models import AgriFood
import sd_data_adapter.models.device as device_model
from geojson import (
    Polygon,
    MultiLineString,
    Point,
    MultiPoint,
    Feature,
    FeatureCollection,
)
from digitaltwin.utils.data_adapter import (
    fill_database,
    generate_feature_collections,
)
from digitaltwin.utils.database import (
    clear_database,
    get_demo_parcels,
    find_device,
    get_by_id,
    find_command_messages,
)
from digitaltwin.utils.riskmap_helpers import (
    generate_points_in_polygon,
    get_daily_actions,
    get_polygon_bounds,
    create_prescription_maps
)

from digitaltwin.ascabmodel.ascab_model import create_digital_twins
from digitaltwin.ascabmodel.recommendation import minimize_risk


def get_polygon_bounds_from_feature_collection(parcel):
    """
    Extract outer ring coordinates from the last feature's geometry
    in the parcel's FeatureCollection.
    Returns a list of (lon, lat) tuples.
    """
    fc = parcel

    geom = fc["features"][-1]["geometry"]

    if geom["type"] == "Polygon":
        # outer ring
        return geom["coordinates"][0]
    elif geom["type"] == "MultiPolygon":
        # first polygon, outer ring
        return geom["coordinates"][0][0]
    else:
        raise ValueError(f"Unsupported geometry type: {geom['type']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="localhost", help="Hostname orion context broker"
    )
    args = parser.parse_args()

    DAClient.get_instance(host=args.host, port=1026)

    clear_database()
    fill_database()

    parcels = search(get_demo_parcels("Lithuania"), ctx=AgriFood.ctx)
    # digital_twins = create_digital_twins(parcels)

    action_dict = {}
    points_dict = {}
    polygon_dict = {}
    for parcel in parcels:
        # polygon_coordinates = get_polygon_bounds_from_feature_collection(parcel)
        polygon_coordinates = get_polygon_bounds(parcel)
        print(polygon_coordinates)
        polygon_dict[parcel.id]: dict[str, list[tuple[float, float]]] = polygon_coordinates
        action_dict[parcel.id]: dict[str, dict[datetime.date, float]] = get_daily_actions(
            (44.5, 1.9),
            start_date=datetime.date(2025, 3, 1),
            end_date=datetime.date(2025, 8, 18),
        )
        # Emulate detections from the camera
        points: dict = generate_points_in_polygon(
            polygon_coordinates,
            action_dict[parcel.id],
            points_per_day=40
        )
        points_dict[parcel.id]: dict[str, dict] = points

    # note sigma value
    create_prescription_maps(
        action_dict,
        polygon_dict,
        points_dict,
        sigma = 0.0008,  # parameter for spread of gaussian
        cutoff_minimum=0.5  # parameter of cutoff for minimum value from agent's action
    )


if __name__ == "__main__":
    main()
