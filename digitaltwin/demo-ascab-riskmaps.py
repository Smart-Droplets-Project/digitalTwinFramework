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
    get_daily_actions,
    get_polygon_bounds,
    create_prescription_maps
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

    action_dict = {}
    points_dict = {}
    polygon_dict = {}
    for parcel in parcels:
        polygon_coordinates = get_polygon_bounds(parcel)
        print(polygon_coordinates)
        polygon_dict[parcel.id]: dict[str, list[tuple[float, float]]] = polygon_coordinates
        action_dict[parcel.id]: dict[str, dict[datetime.date, float]] = get_daily_actions(
            polygon_coordinates[0],
            start_date=datetime.date(2025, 3, 1),
            end_date=datetime.date(2025, 5, 31),
        )
        # Emulate detections from the camera
        points: dict = generate_points_in_polygon(
            polygon_coordinates,
            action_dict[parcel.id],
            points_per_day=15
        )
        points_dict[parcel.id]: dict[str, dict] = points

    create_prescription_maps(
        action_dict,
        polygon_dict,
        points_dict,
    )


if __name__ == "__main__":
    main()
