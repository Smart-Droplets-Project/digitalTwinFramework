import argparse
import datetime
from sd_data_adapter.client import DAClient

from digitaltwin.cropmodel.crop_model import get_default_variables
from digitaltwin.utils.data_adapter import fill_database, fill_database_ascab
from digitaltwin.utils.database import has_demodata, clear_database
from digitaltwin.utils.simulator import run_cropmodel, run_ascabmodel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="localhost", help="Hostname orion context broker"
    )
    args = parser.parse_args()

    DAClient.get_instance(host=args.host, port=1026)

    clear_database(verbose=False)

    # fill database with demo data
    if not has_demodata():
        fill_database(variables=get_default_variables())

    # Define start and end dates
    start_date = datetime.datetime.strptime("20221003", "%Y%m%d").date()
    end_date = datetime.datetime.strptime("20230820", "%Y%m%d").date()

    # List to hold all dates
    start_end = []

    # Add 1st April and 1st May manually
    start_end.append(datetime.date(2023, 4, 1))  # Add 1st April
    start_end.append(datetime.date(2023, 5, 1))  # Add 1st May

    # Generate the dates, incrementing by 75 days, ensuring start_date is included
    start_end += [
        start_date + datetime.timedelta(days=i)
        for i in range(0, (end_date - start_date).days + 1, 75)
    ]

    # Ensure that end_date is included
    if end_date not in start_end:
        start_end.append(end_date)

    # Remove duplicates and sort the list
    start_end = sorted(set(start_end))

    for day in start_end:
        print(f"start simulation at {day}")
        run_cropmodel(
            debug=True,
            end_date=day,
            use_cropgym_agent=False,
        )


if __name__ == "__main__":
    main()
