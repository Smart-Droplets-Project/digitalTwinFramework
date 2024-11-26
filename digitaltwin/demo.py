import argparse

from sd_data_adapter.client import DAClient

from digitaltwin.cropmodel.crop_model import get_default_variables
from digitaltwin.utils.data_adapter import fill_database
from digitaltwin.utils.database import has_demodata, clear_database
from digitaltwin.utils.simulator import run_cropmodel


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
