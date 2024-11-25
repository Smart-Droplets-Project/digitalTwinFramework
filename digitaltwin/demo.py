import argparse

from sd_data_adapter.client import DAClient
from sd_data_adapter.api import search, upsert
from sd_data_adapter.models import AgriFood
import sd_data_adapter.models.device as device_model

from digitaltwin.cropmodel.crop_model import create_digital_twins, get_default_variables
from digitaltwin.cropmodel.recommendation import standard_practice
from digitaltwin.utils.data_adapter import (
    create_command_message,
    fill_database,
    get_row_coordinates,
    generate_rec_message_id,
    get_recommendation_message,
)
from digitaltwin.utils.database import (
    get_by_id,
    get_demo_parcels,
    has_demodata,
    find_parcel_operations,
    find_device,
    find_device_measurement,
    find_command_messages,
    clear_database,
    get_parcel_operation_by_date,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="localhost", help="Hostname orion context broker"
    )
    args = parser.parse_args()

    DAClient.get_instance(host=args.host, port=1026)

    clear_database()

    # fill database with demo data
    if not has_demodata():
        fill_database(variables=get_default_variables())

    run_cropmodel(debug=True)


if __name__ == "__main__":
    main()
