import argparse

from ascab.env.env import AScabEnv

from sd_data_adapter.client import DAClient
from digitaltwin.services.data_adapter import create_agripest
from digitaltwin.services.database import clear_database


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="localhost", help="Hostname orion context broker"
    )
    args = parser.parse_args()

    DAClient.get_instance(host=args.host, port=1026)

    clear_database()

    create_agripest()

    ascab = AScabEnv(location=(42.1620, 3.0924), dates=("2022-01-01", "2022-10-01"))

    # run digital twin
    terminated = False
    ascab.reset()
    action = 0.0
    while not terminated:
        _, _, terminated, _, _ = ascab.step(action)

    # get output
    summary_output = ascab.get_info(to_dataframe=True)
    print(summary_output)


if __name__ == "__main__":
    main()
