import argparse

from sd_data_adapter.client import DAClient
from sd_data_adapter.api import search
from sd_data_adapter.models import AgriFood

from digitaltwin.utils.data_adapter import create_agripest, fill_database_ascab
from digitaltwin.utils.database import clear_database, get_demo_parcels
from digitaltwin.ascabmodel.ascab_model import AscabModel, create_digital_twins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="localhost", help="Hostname orion context broker"
    )
    args = parser.parse_args()

    DAClient.get_instance(host=args.host, port=1026)

    clear_database()
    fill_database_ascab()
    create_agripest()

    parcels = search(get_demo_parcels("Serrater"), ctx=AgriFood.ctx)
    print(f"parcels: {parcels}")

    digital_twins = create_digital_twins(parcels)

    # run digital twins
    for digital_twin in digital_twins:
        terminated = False
        digital_twin.reset()
        action = 0.0
        while not terminated:
            _, _, terminated, _, _ = digital_twin.step(action)

        # get output
        summary_output = digital_twin.get_info(to_dataframe=True)
        print(summary_output)


if __name__ == "__main__":
    main()
