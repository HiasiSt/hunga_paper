#!/usr/bin/env python
# coding: utf-8

"""
This script plots temperature anomaly profiles/gridpoints from RO/MLS data
within the early Hunga water vapor plume.

Autohr: Matthias Stocker [matthias.stocker(at)uni-graz.at]
"""

import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates


def main():

    parser = argparse.ArgumentParser(
        description="Plot temperature profiles within a water vapor plume from RO and MLS data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the data files",
    )
    parser.add_argument(
        "--plot_dir", type=str, required=True, help="Directory to save the plots"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2022-01-16",
        help="Start date for the time range (format: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2022-03-01",
        help="End date for the time range (format: YYYY-MM-DD)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    plot_dir = args.plot_dir
    start_date = args.start_date
    end_date = args.end_date

    ds_temp_profs_ro = xr.open_dataset(
        f"{data_dir}/RO_dry_temperature_anomaly_profiles_within_water_vapor_plume_2022-01-16_to_2022-04-24.nc"
    )
    ds_temp_profs_mls = xr.open_dataset(
        f"{data_dir}/MLS_temperature_anomaly_gridpoints_within_water_vapor_plume_2022-01-16_to_2022-04-24.nc"
    )

    plot_profs_within_wv_plume(
        ds_temp_profs_ro, ds_temp_profs_mls, data_dir, plot_dir, start_date, end_date
    )


def plot_profs_within_wv_plume(
    ds_temp_profs_ro, ds_temp_profs_mls, data_dir, plot_dir, start_date, end_date
):


    ds_temp_profs_mls["altitude"] = ds_temp_profs_mls.altitude / 1000.0
    ds_temp_profs_ro["altitude"] = ds_temp_profs_ro.altitude / 1000.0

    ticks = np.arange(-5, 6, 1)
    levels = np.linspace(-5.25, 5.25, 22)

    da_temp_mls = (
        ds_temp_profs_mls["temperature_anom"]
        .sel(altitude=slice(18, 35))
        .resample(time="d")
        .interpolate()
        .sel(time=slice(start_date, end_date))
    )
    da_temp_ro = (
        ds_temp_profs_ro["dry_temperature_anom"]
        .sel(altitude=slice(18, 35))
        .resample(time="d")
        .interpolate()
        .sel(time=slice(start_date, end_date))
    )

    ds_MLS_wv = xr.open_dataset(
        f"{data_dir}/MLS_water_vapor_max_alt-of-max_lon-coverage_2022-01-16_2022-04-24.nc"
    )
    ds_MLS_wv = ds_MLS_wv.sel(time=slice(start_date, end_date))

    max_alt_h2o = ds_MLS_wv.altitude_of_MLS_water_vapor_maximum
    max_wv = ds_MLS_wv.MLS_water_vapor_maximum
    cover_wv = ds_MLS_wv.MLS_water_vapor_lon_coverage

    plt.rcParams.update({"font.size": 16})
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(15, 19), gridspec_kw={"height_ratios": [3, 3, 1, 1]}, sharex=True
    )

    # MLS ######
    CS = ax1.contourf(
        da_temp_mls.time,
        da_temp_mls.altitude,
        da_temp_mls.T,
        vmax=5,
        vmin=-5,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )
    ax1.plot(
        max_alt_h2o.time,
        max_alt_h2o / 1000.0,
        linestyle="dashed",
        color="black",
        linewidth=3,
        label="Altitude of H$_{2}$O max.",
    )
    ax1.set_ylim(18, 34)
    ax1.set_ylabel("Altitude (km)")
    ax1.grid(linestyle="dotted", linewidth=1.5)
    ax1.set_title("MLS")
    ax1.legend()

    # RO ######
    CS1 = ax2.contourf(
        da_temp_ro.time,
        da_temp_ro.altitude,
        da_temp_ro.T,
        vmax=5,
        vmin=-5,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )
    ax2.plot(
        max_alt_h2o.time,
        max_alt_h2o / 1000.0,
        linestyle="dashed",
        color="black",
        linewidth=3,
        label="Altitude of H$_{2}$O max.",
    )
    ax2.axvspan(
        "2022-01-16",
        "2022-02-01",
        hatch="X",
        facecolor="none",
        edgecolor="dimgray",
        linewidth=1,
    )
    cbar_ax = plt.gcf().add_axes([0.92, 0.415, 0.015, 0.4])
    cbar = fig.colorbar(
        CS, cax=cbar_ax, ticks=ticks, label="Temperature anom. (K)", extend="both"
    )
    ax2.set_ylim(18, 34)
    ax2.set_ylabel("Altitude (km)")
    ax2.grid(linestyle="dotted", linewidth=1.5)
    ax2.set_title("RO")
    ax2.legend()

    ax3.plot(max_wv.time, max_wv.data, color="black", linewidth=1.5)
    ax3.grid(linestyle="dotted", linewidth=1.5)
    ax3.set_ylabel("H$_{2}$O max. (ppmv)")

    ax4.plot(cover_wv.time, cover_wv.data, color="black", linewidth=1.5)
    ax4.grid(linestyle="dotted", linewidth=1.5)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Lon. coverage (%)")
    days = mdates.DayLocator(interval=7)
    ax4.xaxis.set_major_locator(days)

    plt.savefig(
        f"{plot_dir}/RO_profiles_MLS_gridboxes_within_water_vapor_plume.png",
        bbox_inches="tight",
        dpi=300,
    )


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \ppmv" if plt.rcParams["text.usetex"] else f"{s} ppmv"


if __name__ == "__main__":

    main()
