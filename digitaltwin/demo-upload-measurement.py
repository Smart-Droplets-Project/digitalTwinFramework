import argparse
from datetime import datetime

from sd_data_adapter.client import DAClient
from sd_data_adapter.models import Devices
from sd_data_adapter.api import search

from digitaltwin.utils.data_adapter import (fill_database, create_device, create_device_measurement)
from digitaltwin.utils.database import (clear_database)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orionhost", type=str, default="localhost", help="Hostname orion context broker"
    )
    args = parser.parse_args()

    DAClient.get_instance(host=args.orionhost, port=1026)
    clear_database()
    fill_database()

    devices = search({"type": "Device"}, ctx=Devices.ctx)
    create_device_measurement(
        device=devices[0],
        date_observed=datetime.now().isoformat(),
        value=0.5,
    )


if __name__ == "__main__":
    main()
