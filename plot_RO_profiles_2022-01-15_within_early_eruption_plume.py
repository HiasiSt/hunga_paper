#!/usr/bin/env python
# coding: utf-8
"""
This script plots RO temperature/bending angle anomaly profiles
(with respect to the IFS forecast) recorded within the early Hunga
water vapor plume.

Author: Matthias Stocker [matthias.stocker(at)uni-graz.at]

"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib.ticker as mticker
import datetime
import rioxarray as rxr
import cartopy.crs as ccrs

import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing input data files",
    )
    parser.add_argument(
        "--plot_dir", type=str, required=True, help="Directory to save the plot."
    )
    parser.add_argument(
        "--variable",
        type=str,
        required=True,
        choices=["temperature_anom", "bending_angle_anom"],
        default="temperature",
        help="Variable to plot (bending angle or temperature).",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    plot_dir = args.plot_dir
    variable = args.variable

    # Fixed date
    dates = ["2022-015"]

    if len(dates[0]) == 8:
        start_date = datetime.datetime.strptime(dates[0], "%Y-%j")
    elif len(dates[0]) == 10:
        start_date = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
    else:
        raise ValueError("Unknown date format in dates.")

    latmin = -25
    latmax = -15
    lonmin = -180
    lonmax = -170

    time_matters = True

    ds_profs = xr.open_dataset(
        f"/{data_dir}/RO_profiles_2022-01-15_within_water_vapor_plume.nc"
    )

    plot_vis_profs(
        ds_profs,
        start_date,
        latmin,
        latmax,
        lonmin,
        lonmax,
        variable,
        data_dir,
        plot_dir,
    )


def plot_vis_profs(
    ds_profs,
    start_date,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    variable,
    data_dir,
    plot_dir,
):
    """
    Plots the GOES sattelite image as well as the RO temperature anomaly profiles.

    """

    if variable == "temperature_anom":
        var = "dry_temperature_anom_ifs"
        x_lab = "RO dry temperature anom. (K)"
        scaler = 1
        alt_range = [0, 50]

    elif variable == "bending_angle_anom":
        var = "bending_angle_anom_ifs"
        x_lab = "Bending angle anom. (%)"
        scaler = 0.01
        alt_range = [0, 50]

    else:
        raise ValueError("Variable name {} not known!".format(variable))

    time_matters = True

    da = rxr.open_rasterio(
        f"{data_dir}/snapshot-2022-01-15T05_10_00Z_geocolor_multispectral.tiff"
    )

    # Sorting ds_profs by time
    times = pd.to_datetime(ds_profs.time.values)

    # Get sorted indices based on time
    sorted_indices = times.argsort()
    # Sort the dataset by the sorted indices
    ds_profs = ds_profs.isel(time=sorted_indices)

    # Define the projection
    crs = ccrs.PlateCarree()

    # Figure setup
    plt.rcParams.update({"font.size": 15})
    plt.rc("legend", fontsize=7)

    # Define a custom color list
    custom_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    color_index = 0

    fig = plt.figure(figsize=(15, 9))

    gsouter = gridspec.GridSpec(1, 2, width_ratios=[1, 0.6])
    spec0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gsouter[0], wspace=0.2)
    spec1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gsouter[1], wspace=0)
    ax1 = fig.add_subplot(spec0[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(spec1[0, 0])

    # Plot the true color image for the selected region
    da.plot.imshow(ax=ax1, rgb="band", transform=crs)
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)

    gl = ax1.gridlines(
        crs=crs, draw_labels=True, linewidth=1, color="grey", alpha=0.5, linestyle="--"
    )
    gl.right_labels = False
    gl.bottom_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 185, 2.5))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 95, 2.5))

    for x in range(ds_profs.sizes["time"]):
        # Select the current profile
        ds = ds_profs.isel(time=x)

        # Select altitude range
        ds["altitude"] = ds.altitude / 1000.0
        ds = ds.sel(altitude=slice(alt_range[0], alt_range[1]))

        # Get time
        time = ds.time.min().data
        time = pd.to_datetime(time)

        if time_matters:
            if time.hour < 4 or time.hour > 13:
                continue
            else:
                ax1.scatter(
                    ds.longitude,
                    ds.latitude,
                    marker="o",
                    s=80,
                    color=custom_colors[color_index % len(custom_colors)],
                    transform=ccrs.PlateCarree(),
                    zorder=10,
                    label="Time: {:02d}:{:02d} UTC".format(time.hour, time.minute),
                )
                line = ax2.plot(
                    ds[var].squeeze().data / scaler,
                    ds.altitude,
                    linewidth=2,
                    color=custom_colors[color_index % len(custom_colors)],
                    zorder=10,
                    label="Time: {:02d}:{:02d} UTC; {}".format(
                        time.hour, time.minute, ds.ReceiverId.data
                    ),
                )

        else:
            ax1.scatter(
                ds.longitude,
                ds.latitude,
                marker="o",
                s=80,
                color=custom_colors[color_index % len(custom_colors)],
                transform=ccrs.PlateCarree(),
                zorder=10,
                label="Time: {:02d}:{:02d} UTC".format(time.hour, time.minute),
            )
        color_index += 1

    ax1.text(
        0.05,
        0.95,
        "GOES-West, Time: 05:10 UTC",
        fontsize=15,
        color="white",
        transform=ax1.transAxes,
    )

    if ds_profs.sizes["time"] < 1000:
        ax1.legend()
        ax2.legend()

    ax2.set_yticks(np.arange(0, alt_range[1] + 2, 2))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.set_ylim(alt_range[0], alt_range[1])

    ax2.set_xlabel(x_lab)
    ax2.set_ylabel("Altitude (km)")

    ax2.grid(linestyle="dotted")

    ax1.set_title("")

    fig.suptitle(
        "Hunga Tonga-Hunga Ha'apai, {}-{}-{}".format(
            start_date.year, start_date.month, start_date.day
        )
    )

    plot_name = f"{plot_dir}/RO_profiles_within_early_water_vapor_plume_2022-01-15.png"
    print("Plot saved to: {}".format(plot_name))
    plt.savefig(plot_name, bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":

    main()
