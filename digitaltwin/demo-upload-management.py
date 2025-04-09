import argparse

from sd_data_adapter.client import DAClient
from sd_data_adapter.api import search

from digitaltwin.utils.data_adapter import (
    fill_database,
    create_device_measurement,
    create_fertilizer,
    create_agriparcel_operation,
)
from digitaltwin.utils.database import clear_database, get_demo_parcels
from sd_data_adapter.models import AgriFood


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orionhost",
        type=str,
        default="localhost",
        help="Hostname orion context broker",
    )
    args = parser.parse_args()

    DAClient.get_instance(host=args.orionhost, port=1026)

    clear_database()
    fill_database()

    parcels = search(get_demo_parcels(), ctx=AgriFood.ctx)
    fertilizer = create_fertilizer()
    for parcel in parcels:
        fertilizer_application = create_agriparcel_operation(
            parcel=parcel, product=fertilizer
        )


if __name__ == "__main__":
    main()
