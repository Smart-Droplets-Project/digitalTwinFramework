import argparse

from ascab.env.env import AScabEnv

from sd_data_adapter.client import DAClient
import sd_data_adapter.models.agri_food as agri_food_model
from sd_data_adapter.api import upload


def clear_database():
    # clear database with error handling
    try:
        with DAClient.get_instance() as client:
            client.purge()
        print("Purge completed successfully.")
    except Exception as e:
        # Catch any exceptions that occur and print the error message
        print(f"An error occurred while purging the database: {e}")


def create_agripest(do_upload=True):
    model = agri_food_model.AgriPest(description="ascab")
    if do_upload:
        upload(model)
    return model


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
