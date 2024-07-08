#!/usr/bin/env python
# coding: utf-8

"""
This script generates a scatter plot of tropical vs. extratropical temperature anomalies.

Author: Matthias Stocker [matthias.stocker(at)uni-graz.at]

"""

import argparse
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main():

    parser = argparse.ArgumentParser(
        description="Scatter plot of tropical vs. extratropical temperature anomalies."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory where the input data file is located.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        required=True,
        help="Directory where the output plot will be saved.",
    )
    parser.add_argument(
        "--alt_start",
        type=int,
        default=30,
        help="Start of the altitude range in km (default: 30).",
    )
    parser.add_argument(
        "--alt_end",
        type=int,
        default=35,
        help="End of the altitude range in km (default: 35).",
    )

    args = parser.parse_args()

    plot_dir = args.plot_dir
    data_dir = args.data_dir
    alt_start = args.alt_start
    alt_end = args.alt_end

    alt_slice = [alt_start, alt_end]

    ds_in = xr.open_dataset(
        f"{data_dir}/regression_results_RO_dry_temperature_anomalies_monthly_2002-01-2023-12.nc"
    )
    ds_in["altitude"] = ds_in.altitude / 1000.0
    da_resid = ds_in.dry_temperature - ds_in[
        "dry_temperature_reconstr_vector_GLSAR"
    ].sel(k=ds_in.k[0:2]).sum(dim="k")
    ds = (
        da_resid.to_dataset(name="dry_temperature")
        .drop(["month", "longitude"])
        .squeeze()
    )

    da_shs = (
        ds.dry_temperature.sel(latitude_bins=slice(-90, -30))
        .mean("latitude_bins")
        .sel(altitude=slice(alt_slice[0], alt_slice[1]))
        .mean("altitude")
    )
    da_nhs = (
        ds.dry_temperature.sel(latitude_bins=slice(30, 90))
        .mean("latitude_bins")
        .sel(altitude=slice(alt_slice[0], alt_slice[1]))
        .mean("altitude")
    )

    da_extra = (da_shs + da_nhs) / 2
    da_trop = (
        ds.dry_temperature.sel(latitude_bins=slice(-30, 30))
        .mean("latitude_bins")
        .sel(altitude=slice(alt_slice[0], alt_slice[1]))
        .mean("altitude")
    )

    # Remove extreme outlier
    da_extra.loc[{"time": "2019-09"}] = 0
    da_trop.loc[{"time": "2019-09"}] = 0

    plot_temperature_scatter(da_trop, da_extra, alt_slice, plot_dir)


def plot_temperature_scatter(da_trop, da_extra, alt_slice, plot_dir):
    """
    Plot a scatter plot of tropical vs. extratropical temperature anomalies.
    Includes a linear regression line across all data points and displays
    the correlation coefficient.

    Parameters:
    da_trop (xarray.DataArray): DataArray containing tropical temperatures
    with a 'time' dimension.
    da_extra (xarray.DataArray): DataArray containing extratropical temperatures
    with a 'time' dimension.
    alt_slice (list): List of two integers representing the altitude range (in meters).
    plot_dir (str): Directory where the output plot will be saved.
    """

    # Group data by year
    trop_grouped = da_trop.groupby("time.year")
    extra_grouped = da_extra.groupby("time.year")

    # Prepare a color map that has as many distinct colors as there are groups
    years = list(trop_grouped.groups.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))

    # Figure setup
    plt.rcParams.update({"font.size": 16})

    plt.figure(figsize=(10, 8))
    all_trop_data = []
    all_extra_data = []

    # Iterate over the groups and plot, with colors aligned to the number of years
    for (trop_year, trop_data), (extra_year, extra_data), color in zip(
        trop_grouped, extra_grouped, colors
    ):
        if trop_year != extra_year:
            continue  # Ensuring consistency in the years for safety
        if trop_year == 2022:
            plt.scatter(trop_data, extra_data, color="red", label=str(trop_year))
        else:
            plt.scatter(trop_data, extra_data, color=color, label=str(trop_year))
        all_trop_data.extend(trop_data.values)
        all_extra_data.extend(extra_data.values)

        # Annotate the months for 2022 only
        if trop_year == 2022:
            for i, (x, y) in enumerate(zip(trop_data, extra_data)):
                month = trop_data.time.dt.month.values[i]  # Extracting the month
                plt.text(x, y, f"{month}", color="black", fontsize=10)

    # Linear regression and correlation coefficient
    slope, intercept, r_value, _, _ = stats.linregress(all_trop_data, all_extra_data)
    plt.plot(
        all_trop_data,
        np.array(all_trop_data) * slope + intercept,
        "k-",
        label=f"Linear fit: r = {r_value:.2f}",
    )

    plt.title(f"Altitude mean ({alt_slice[0]} km to {alt_slice[1]} km)")
    plt.xlabel("Tropical temp. anom. (K)")
    plt.ylabel("Extratropical temp. anom. (K)")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.legend(title="Years", loc="best", fontsize=8)
    plt.grid(True)
    output_str = (
        f"{plot_dir}/Trop_vs_extratrop_temp_anom_{alt_slice[0]}_to_{alt_slice[1]}.png"
    )
    plt.savefig(output_str, bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":

    main()
