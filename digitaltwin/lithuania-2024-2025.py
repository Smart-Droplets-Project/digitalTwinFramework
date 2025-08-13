import argparse
import datetime
import pandas as pd
import os

import warnings
from sd_data_adapter.client import DAClient

from digitaltwin.cropmodel.crop_model import (
    get_default_variables,
    print_calibration_parameters,
)
from digitaltwin.utils.data_adapter import fill_database, fill_database_ascab
from digitaltwin.utils.database import has_demodata, clear_database
from digitaltwin.utils.simulator import run_cropmodel
from utils.plotter import plot, plot_calibration_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="localhost", help="Hostname orion context broker"
    )
    parser.add_argument(
        "--load-from-disk",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=True,
        help="Whether to load results from disk (default: True). Set to False to generate new results.",
    )
    args = parser.parse_args()

    repodir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(repodir, "model_results_13_aug")
    os.makedirs(outdir, exist_ok=True)

    # Expected files
    required_files = [
        "observations.csv",
        "standard_practice.csv",
        "rl.csv",
        "uncalibrated.csv",
        "calibration.csv",
    ]
    missing_files = [
        f for f in required_files if not os.path.exists(os.path.join(outdir, f))
    ]

    if args.load_from_disk and missing_files:
        warnings.warn(
            f"Some required result files are missing: {', '.join(missing_files)}. "
            f"Generating new results instead."
        )

        print("Start Uncalibrated")
        clear_database(verbose=False)
        fill_database(
            variables=get_default_variables(), fertilization="standard_practice"
        )
        start_date = datetime.datetime.strptime("20240918", "%Y%m%d").date()
        end_date = datetime.datetime.strptime("20250811", "%Y%m%d").date()

        start_end = []
        start_end.append(end_date)
        start_end = sorted(set(start_end))
        for day in start_end:
            print(f"start simulation at {day}")
            digitaltwins_uncalibrated, _ = run_cropmodel(
                debug=False,
                end_date=day,
                use_cropgym_agent=False,
                calibrate_flag=False,
            )
        pd.DataFrame(digitaltwins_uncalibrated[0].get_output()).to_csv(
            os.path.join(outdir, "uncalibrated.csv"), index=False
        )

        print("Start Calibration")
        clear_database(verbose=False)
        fill_database(
            variables=get_default_variables(), fertilization="standard_practice"
        )
        df_list_calibrate = []
        start_end = []
        start_end.append(start_date)
        start_end.append(end_date)
        start_end += [
            start_date + datetime.timedelta(days=i)
            for i in range(0, (end_date - start_date).days + 1, 14)
        ]
        for day in start_end:
            print(f"start simulation at {day}")
            digitaltwins, observations = run_cropmodel(
                debug=False,
                end_date=day,
                use_cropgym_agent=False,
                calibrate_flag=True,
            )
            if digitaltwins[0].parameterprovider._override:
                calibrated_parameters = pd.DataFrame(
                    digitaltwins[0].parameterprovider._override, index=[day]
                )
                df_list_calibrate.append(calibrated_parameters)
        df_calibrate = pd.concat(df_list_calibrate)
        df_calibrate.to_csv(os.path.join(outdir, "calibration.csv"), index=True)
        pd.DataFrame(digitaltwins[0].get_output()).to_csv(
            os.path.join(outdir, "standard_practice.csv"), index=False
        )
        observations[0].to_csv(os.path.join(outdir, "observations.csv"), index=True)

        print("Start RL agent")
        clear_database(verbose=False)
        fill_database(variables=get_default_variables(), fertilization="rl_agent")
        start_end = []
        start_end.append(start_date)
        start_end.append(end_date)
        start_end += [
            start_date + datetime.timedelta(days=i)
            for i in range(0, (end_date - start_date).days + 1, 14)
        ]
        # Remove duplicates and sort the list
        if end_date not in start_end:
            start_end.append(end_date)
        start_end = sorted(set(start_end))
        for day in start_end:
            print(f"start simulation at {day}")
            digitaltwins_rl, _ = run_cropmodel(
                debug=False,
                end_date=day,
                use_cropgym_agent=True,
                calibrate_flag=True,
            )
        pd.DataFrame(digitaltwins_rl[0].get_output()).to_csv(
            os.path.join(outdir, "rl.csv"), index=False
        )

    print("Start plotting")
    df_assimilate = pd.read_csv(
        os.path.join(outdir, "observations.csv"), index_col=0, parse_dates=True
    )
    standard_practice = pd.read_csv(os.path.join(outdir, "standard_practice.csv"))
    rl_agent = pd.read_csv(os.path.join(outdir, "rl.csv"))
    uncalibrated = pd.read_csv(os.path.join(outdir, "uncalibrated.csv"))

    plot(
        standard_practice=standard_practice,
        rl_agent=rl_agent,
        uncalibrated=uncalibrated,
        df_assimilate=df_assimilate,
        calibrate_flag=True,
        dotted_start_date=None,
    )

    calibration_df = pd.read_csv(os.path.join(outdir, "calibration.csv"))
    calibration_df.rename(columns={calibration_df.columns[0]: "day"}, inplace=True)
    calibration_df["day"] = pd.to_datetime(calibration_df["day"])
    calibration_df["TSUM2"] = 750

    plot_calibration_params(calibration_df, standard_practice)


if __name__ == "__main__":
    main()
