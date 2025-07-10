import argparse
import datetime
import pickle

from sd_data_adapter.client import DAClient
from sd_data_adapter.api import search, upsert
from sd_data_adapter.models import AgriFood
import sd_data_adapter.models.device as device_model
from geojson import Polygon

from digitaltwin.utils.data_adapter import (
    fill_database_ascab,
    create_device_measurement,
    create_command_message,
    generate_rec_message_id,
    create_geojson_from_feature_collection,
    get_recommendation_message,
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
    get_risks,
    get_polygon_bounds,
    visualize_riskmap,
    prescription_from_points,
    create_prescription_geojson
)

from digitaltwin.ascabmodel.ascab_model import create_digital_twins
from digitaltwin.ascabmodel.recommendation import minimize_risk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="localhost", help="Hostname orion context broker"
    )
    args = parser.parse_args()

    DAClient.get_instance(host=args.host, port=1026)

    clear_database()
    fill_database_ascab()

    parcels = search(get_demo_parcels("Serrater"), ctx=AgriFood.ctx)
    # digital_twins = create_digital_twins(parcels)

    risk_dict = {}
    points_dict = {}
    polygon_dict = {}
    for parcel in parcels:
        polygon_coordinates = get_polygon_bounds(parcel)
        print(polygon_coordinates)
        points_list = generate_points_in_polygon(polygon_coordinates)
        polygon_dict[parcel.id] = polygon_coordinates
        risk_dict[parcel.id] = get_risks(points_list)
        points_dict[parcel.id] = points_list

    from_points = False
    if from_points:
        prescription_from_points(risk_dict, points_dict)
    else:
        create_prescription_geojson(risk_dict, polygon_dict)

    visualize_riskmap()


if __name__ == "__main__":
    main()
