import os
import datetime
import tempfile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from digitaltwin.cropmodel.crop_model import get_default_variables, get_titles


def plot(
    standard_practice,
    rl_agent,
    uncalibrated,
    df_assimilate,
    calibrate_flag=True,
    dotted_start_date=None,
):
    # Convert 'day' columns to datetime and set as index
    for df in [standard_practice, rl_agent]:
        df["day"] = pd.to_datetime(df["day"])
        df.set_index("day", inplace=True)

    # Uncalibrated might already have datetime index or 'day' column
    if "day" in uncalibrated.columns:
        uncalibrated["day"] = pd.to_datetime(uncalibrated["day"])
        uncalibrated.set_index("day", inplace=True)
    else:
        uncalibrated.index = pd.to_datetime(uncalibrated.index)

    # Assimilate index to datetime
    df_assimilate.index = pd.to_datetime(df_assimilate.index)

    # Normalize dotted_start_date to datetime if given as date
    if (
        dotted_start_date is not None
        and isinstance(dotted_start_date, datetime.date)
        and not isinstance(dotted_start_date, datetime.datetime)
    ):
        dotted_start_date = datetime.datetime.combine(
            dotted_start_date, datetime.time.min
        )

    # Filter assimilated data before dotted_start_date if provided
    if dotted_start_date is not None:
        df_assimilate = df_assimilate[df_assimilate.index < dotted_start_date]

    plot_variables = get_default_variables()
    titles = get_titles()

    fig, axes = plt.subplots(len(plot_variables), 1, sharex=True, figsize=(12, 10))

    for i, var in enumerate(plot_variables):
        ax = axes if len(plot_variables) == 1 else axes[i]

        # Plot standard practice - red solid line
        if dotted_start_date is None:
            ax.plot_date(
                standard_practice.index,
                standard_practice[var],
                "r-",
                label="Standard Practice",
            )
        else:
            solid = standard_practice[standard_practice.index < dotted_start_date]
            first_10 = standard_practice[
                (standard_practice.index >= dotted_start_date)
                & (
                    standard_practice.index
                    < dotted_start_date + datetime.timedelta(days=10)
                )
            ]
            after_10 = standard_practice[
                standard_practice.index
                >= dotted_start_date + datetime.timedelta(days=10)
            ]
            if not solid.empty:
                ax.plot_date(solid.index, solid[var], "r-", label="Standard Practice")
            if not first_10.empty:
                ax.plot_date(first_10.index, first_10[var], "r--")
            if not after_10.empty:
                ax.plot_date(after_10.index, after_10[var], "r:")

        # Plot RL agent - green line solid/dashed/dotted
        if dotted_start_date is None:
            ax.plot_date(
                rl_agent.index,
                rl_agent[var],
                "g-",
                label="RL Agent",
            )
        else:
            solid = rl_agent[rl_agent.index < dotted_start_date]
            first_10 = rl_agent[
                (rl_agent.index >= dotted_start_date)
                & (rl_agent.index < dotted_start_date + datetime.timedelta(days=10))
            ]
            after_10 = rl_agent[
                rl_agent.index >= dotted_start_date + datetime.timedelta(days=10)
            ]
            if not solid.empty:
                ax.plot_date(solid.index, solid[var], "g-", label="RL Agent")
            if not first_10.empty:
                ax.plot_date(first_10.index, first_10[var], "g--")
            if not after_10.empty:
                ax.plot_date(after_10.index, after_10[var], "g:")

        # Plot uncalibrated - orange dashed line WITHOUT markers
        if dotted_start_date is None:
            ax.plot(
                uncalibrated.index,
                uncalibrated[var],
                color="orange",
                linestyle="--",
                label="Uncalibrated",
            )
        else:
            solid = uncalibrated[uncalibrated.index < dotted_start_date]
            first_10 = uncalibrated[
                (uncalibrated.index >= dotted_start_date)
                & (uncalibrated.index < dotted_start_date + datetime.timedelta(days=10))
            ]
            after_10 = uncalibrated[
                uncalibrated.index >= dotted_start_date + datetime.timedelta(days=10)
            ]
            if not solid.empty:
                ax.plot(
                    solid.index,
                    solid[var],
                    color="orange",
                    linestyle="--",
                    label="Uncalibrated",
                )
            if not first_10.empty:
                ax.plot(first_10.index, first_10[var], color="orange", linestyle=":")
            if not after_10.empty:
                ax.plot(after_10.index, after_10[var], color="orange", linestyle=":")

        # Plot observation data before dotted_start_date
        if calibrate_flag and var in df_assimilate.columns:
            ax.plot_date(
                df_assimilate.index,
                df_assimilate[var],
                "bo",
                label="Observations",
            )

        # Axis formatting
        name, unit = titles[var]
        ax.set_ylabel(f"[{unit}]")
        ax.set_title(f"{var} - {name}", fontsize=8.5)
        if i == 0:
            ax.legend(fontsize=6.5)
        ax.grid(True)

        # Draw vertical lines at dotted_start_date if given
        if dotted_start_date is not None:
            ax.axvline(dotted_start_date, color="gray", linestyle="-", linewidth=1)
            ax.axvline(
                dotted_start_date + datetime.timedelta(days=10),
                color="gray",
                linestyle="--",
                linewidth=1,
            )

        # Set x-axis range based on uncalibrated first valid index
        first_valid_index = uncalibrated.dropna(how="any").first_valid_index()
        ax.set_xlim(
            [first_valid_index, uncalibrated.index[-1] + datetime.timedelta(days=7)]
        )

    # Format x-axis labels with abbreviated month names
    axes_iter = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    for ax in axes_iter:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        for label in ax.get_xticklabels():
            label.set_rotation(0)
            label.set_horizontalalignment("center")

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(tempfile.gettempdir(), f"model_output_{now}.png")
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_calibration_params(calibration_df, reference_df):
    # --- Ensure both dataframes have a 'day' column ---
    def ensure_day_column(df):
        if "day" not in df.columns:
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "day"}, inplace=True)
        df["day"] = pd.to_datetime(df["day"])
        return df

    calibration_df = calibration_df.sort_values("day")
    reference_df = reference_df.sort_values("day")
    calibration_df = ensure_day_column(calibration_df)
    reference_df = ensure_day_column(reference_df)

    # --- Determine parameters to plot (exclude 'day') ---
    param_cols = [c for c in calibration_df.columns if c != "day"]

    # --- Create subplots with smaller vertical height ---
    fig, axes = plt.subplots(
        len(param_cols), 1, figsize=(10, 1.8 * len(param_cols)), sharex=True
    )

    if len(param_cols) == 1:
        axes = [axes]  # ensure iterable

    for ax, param in zip(axes, param_cols):
        ax.plot(calibration_df["day"], calibration_df[param], linestyle="-", color="C0")
        ax.set_title(param, fontsize=8.5)
        ax.grid(True)

    # Match x-axis to reference_df's date range
    axes[-1].set_xlim(reference_df["day"].min(), reference_df["day"].max())

    # Format x-axis ticks to month abbreviations
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Reduce space between subplots
    plt.subplots_adjust(hspace=0.4)

    plt.tight_layout()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(tempfile.gettempdir(), f"calibration_{now}.png")
    plt.savefig(filename, dpi=300)
    plt.show()
