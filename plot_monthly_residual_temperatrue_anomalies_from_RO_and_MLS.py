#!/usr/bin/env python
# coding: utf-8

"""
This script plots the monthly residual temperature anomalies from MLS
and RO after removing natural variability (ENSO, QBO, high latitude
variability, solar) using multiple linear regression analysis
(GLSAR from statsmodels).

Author: Matthias Stocker [matthias.stocker(at)uni-graz.at]
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib.dates as mdates
from matplotlib import gridspec
import datetime
import argparse


def main():

    parser = argparse.ArgumentParser(
        description="Plot residual temperature anomalies from MLS and RO data."
    )
    parser.add_argument(
        "--input_data_dir",
        type=str,
        required=True,
        help="Directory containing input data files",
    )
    parser.add_argument(
        "--plot_dir", type=str, required=True, help="Directory to save the plot."
    )
    parser.add_argument(
        "--meas_str",
        type=str,
        required=True,
        choices=["RO", "MLS"],
        help='Measurement string indicating the data source (e.g., "RO" or "MLS").',
    )
    parser.add_argument(
        "--altitudes",
        nargs=3,
        type=int,
        default=[19, 27, 32],
        help="Three different altitudes (in km) to plot (default: 19, 27, 32).",
    )
    parser.add_argument(
        "--dates",
        type=str,
        nargs=3,
        default=["2022-02", "2022-05", "2022-12"],
        help="Three dates to be plotted in the right panels (format (str): YYYY-MM). Default is 2022-02, 2022-05, 2022-12",
    )
    args = parser.parse_args()

    input_data_dir = args.input_data_dir
    plot_alts = args.altitudes
    sig_dates = args.dates

    if args.meas_str == "RO":
        ds_in = xr.open_dataset(
            f"{input_data_dir}/regression_results_RO_dry_temperature_anomalies_monthly_2002-01-2023-12.nc"
        )
        da_resid = ds_in.dry_temperature - ds_in[
            "dry_temperature_reconstr_vector_GLSAR"
        ].sel(k=ds_in.k[0:]).sum(dim="k")
        ds = da_resid.to_dataset(name="temperature_resid_GLSAR")
        ds = ds.rename({"latitude_bins": "latitude"})
        cbar_label = "Temp. anom. (K)"

    elif args.meas_str == "MLS":
        ds_in = xr.open_dataset(
            f"{input_data_dir}/regression_results_MLS_temperature_anomalies_monthly_2005-01-2023-12.nc"
        )
        da_resid = ds_in.temperature - ds_in["temperature_reconstr_vector_GLSAR"].sel(
            k=ds_in.k[0:]
        ).sum(dim="k")
        ds = da_resid.to_dataset(name="temperature_resid_GLSAR")
        cbar_label = "Temp. anom. (K)"

    plot_resids_with_sig(
        ds, args.meas_str, plot_alts, sig_dates, cbar_label, args.plot_dir
    )


def plot_resids_with_sig(ds, data_str, plot_alts, sig_dates, cbar_label, plot_dir):

    ticks_temp_v = np.arange(-4, 5, 1)
    levels_temp_v = np.linspace(-4.25, 4.25, 18)

    ds.load()
    ds["altitude"] = ds.altitude / 1000.0
    ds = ds.squeeze()

    # Figure setup
    plt.rcParams.update({"font.size": 16})

    # Create figure
    fig = plt.figure(figsize=(16, 13))

    gs = gridspec.GridSpec(3, 5, figure=fig, wspace=-0.15, hspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0:4])
    ax2 = fig.add_subplot(gs[0, 4])

    ax3 = fig.add_subplot(gs[1, 0:4])
    ax4 = fig.add_subplot(gs[1, 4])

    ax5 = fig.add_subplot(gs[2, 0:4])
    ax6 = fig.add_subplot(gs[2, 4])

    da_full = (
        ds.temperature_resid_GLSAR.sel(latitude=slice(-62.5, 62.5))
        .sel(altitude=slice(17.5, 38))
        .squeeze()
    )

    # First line #####################################################

    da1 = (
        ds.temperature_resid_GLSAR.sel(altitude=plot_alts[0])
        .sel(latitude=slice(-62.5, 62.5))
        .sel(time=slice("2005-01-01", "2024-01-01"))
    )
    im1 = ax1.contourf(
        da1.time,
        da1.latitude,
        da1.data,
        cmap="RdBu_r",
        levels=levels_temp_v,
        extend="both",
    )

    fig.colorbar(im1, orientation="vertical", label=cbar_label, ticks=ticks_temp_v)

    da_line1 = (
        ds.temperature_resid_GLSAR.sel(altitude=plot_alts[0])
        .sel(latitude=slice(-62.5, 62.5))
        .sel(latitude=slice(-20, 10))
        .mean("latitude")
    )

    ax_double1 = ax1.twinx()
    da_line1.plot(
        ax=ax_double1,
        linewidth=2,
        color="black",
        linestyle="dotted",
        label="Mean 20 $^\circ$S to 10 $^\circ$N",
    )
    ax_double1.set_title("")
    ax_double1.set_ylim(-5, 5)
    ax_double1.set_ylabel("")

    da_sig_line1 = filter_to_maximum_anomaly(da_line1)
    ax_double1.scatter(
        da_sig_line1.time.values, da_sig_line1.data, color="black", marker="*", s=100
    )

    # Lines
    ax1.axhline(linewidth=0.7, color="black")

    ax1.axvline(
        x=datetime.datetime(2022, 1, 15), ymin=0, ymax=1, linewidth=1.5, color="black"
    )
    ax1.plot(datetime.datetime(2022, 1, 15), -20.32, "k^", markersize=8)
    ax1.text(datetime.datetime(2020, 5, 1), -27.32, "Hunga", color="black", fontsize=14)

    # Set the locator
    min_locator = mdates.MonthLocator()
    max_locator = mdates.YearLocator()
    max_ylocator = MultipleLocator(20)

    ax1.set_ylabel("Latitude")
    ax1.set_ylim(-45, 45)
    ax1.set_xlabel("Time")
    ax1.grid(linestyle="dotted", linewidth=1.5)

    if data_str == "RO":
        ax1.set_title(f"RO temperature resid. @ {plot_alts[0]} km")
    if data_str == "MLS":
        ax1.set_title(f"MLS temperature resid. @ {plot_alts[0]} km")

    ax_double1.legend(fontsize=14)

    da2 = (
        ds.temperature_resid_GLSAR.sel(time=sig_dates[0])
        .sel(latitude=slice(-62.5, 62.5))
        .sel(altitude=slice(17.5, 38))
        .squeeze()
    )
    im2 = ax2.contourf(
        da2.latitude,
        da2.altitude,
        da2.T.data,
        cmap="RdBu_r",
        levels=levels_temp_v,
        extend="both",
    )

    sig2 = significance_test_date_three_sigma(da_full, sig_dates[0])
    lat_flat, alt_flat, sig_flat = prepare_plot_data(sig2)
    ax2.scatter(lat_flat, alt_flat, s=sig_flat, marker="x")

    ax2.plot(-20.32, 17.5, "k^", markersize=16)
    ax2.axvline(-20.32, ymin=0, ymax=1, linewidth=1.5, color="black")
    ax2.text(-40, 19, "Hunga", color="black", fontsize=14)
    ax2.set_title(f"{sig_dates[0]}")
    ax2.yaxis.set_ticks_position("right")
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Altitude (km)")
    ax2.set_yticks(np.arange(18, 40, 2))
    ax2.set_xlabel("Latitude")
    ax2.set_xticks(np.arange(-60, 80, 20))
    ax2.set_xlim(-45, 45)
    ax2.grid(linestyle="dotted", linewidth=1.5)

    # SECOND LINE ####################################################

    da3 = (
        ds.temperature_resid_GLSAR.sel(altitude=plot_alts[1])
        .sel(latitude=slice(-62.5, 62.5))
        .sel(time=slice("2005-01-01", "2024-01-01"))
    )
    im3 = ax3.contourf(
        da3.time,
        da3.latitude,
        da3.data,
        cmap="RdBu_r",
        levels=levels_temp_v,
        extend="both",
    )

    fig.colorbar(im3, orientation="vertical", label=cbar_label, ticks=ticks_temp_v)

    da_line3 = (
        ds.temperature_resid_GLSAR.sel(altitude=plot_alts[1])
        .sel(latitude=slice(-62.5, 62.5))
        .sel(latitude=slice(-27.5, -17.5))
        .mean("latitude")
    )

    ax_double3 = ax3.twinx()
    da_line3.plot(
        ax=ax_double3,
        linewidth=2,
        color="black",
        linestyle="dotted",
        label="Mean 30 $^\circ$S to 15 $^\circ$S",
    )
    ax_double3.set_title("")
    ax_double3.set_ylim(-5, 5)
    ax_double3.set_ylabel("")

    da_sig_line3 = filter_to_maximum_anomaly(da_line3)

    ax_double3.scatter(
        da_sig_line3.time.values, da_sig_line3.data, color="black", marker="*", s=100
    )

    # Lines
    ax3.axhline(linewidth=0.7, color="black")

    ax3.axvline(
        x=datetime.datetime(2022, 1, 15), ymin=0, ymax=1, linewidth=1.5, color="black"
    )
    ax3.plot(datetime.datetime(2022, 1, 15), -20.32, "k^", markersize=8)
    ax3.text(datetime.datetime(2020, 5, 1), -27.32, "Hunga", color="black", fontsize=14)

    # Set the locator
    min_locator = mdates.MonthLocator()
    max_locator = mdates.YearLocator()
    max_ylocator = MultipleLocator(20)

    ax3.set_ylabel("Latitude")
    ax3.set_ylim(-45, 45)
    ax3.set_xlabel("Time")
    ax3.grid(linestyle="dotted", linewidth=1.5)

    if data_str == "RO":
        ax3.set_title(f"RO temperature resid. @ {plot_alts[1]} km")
    if data_str == "MLS":
        ax3.set_title(f"MLS temperature resid. @ {plot_alts[1]} km")

    ax_double3.legend(fontsize=14)

    da4 = (
        ds.temperature_resid_GLSAR.sel(time=sig_dates[1])
        .sel(latitude=slice(-62.5, 62.5))
        .sel(altitude=slice(17.5, 38))
        .squeeze()
    )
    im4 = ax4.contourf(
        da4.latitude,
        da4.altitude,
        da4.T.data,
        cmap="RdBu_r",
        levels=levels_temp_v,
        extend="both",
    )

    sig4 = significance_test_date_three_sigma(da_full, sig_dates[1])
    lat_flat, alt_flat, sig_flat = prepare_plot_data(sig4)
    ax4.scatter(lat_flat, alt_flat, s=sig_flat * 20, marker="x", c="black")

    ax4.plot(-20.32, 17.5, "k^", markersize=16)
    ax4.axvline(-20.32, ymin=0, ymax=1, linewidth=1.5, color="black")
    ax4.text(-40, 19, "Hunga", color="black", fontsize=14)
    ax4.set_title(f"{sig_dates[1]}")
    ax4.set_ylabel("Altitude (km)")
    ax4.yaxis.set_ticks_position("right")
    ax4.yaxis.set_label_position("right")
    ax4.set_yticks(np.arange(18, 40, 2))
    ax4.set_xlabel("Latitude")
    ax4.set_xticks(np.arange(-60, 80, 20))
    ax4.set_xlim(-45, 45)
    ax4.grid(linestyle="dotted", linewidth=1.5)

    # THIRD LINE ######################################################

    da5 = (
        ds.temperature_resid_GLSAR.sel(altitude=plot_alts[2])
        .sel(latitude=slice(-62.5, 62.5))
        .sel(time=slice("2005-01-01", "2024-01-01"))
    )
    im5 = ax5.contourf(
        da5.time,
        da5.latitude,
        da5.data,
        cmap="RdBu_r",
        levels=levels_temp_v,
        extend="both",
    )

    fig.colorbar(im5, orientation="vertical", label=cbar_label, ticks=ticks_temp_v)

    da_line5 = (
        ds.temperature_resid_GLSAR.sel(altitude=plot_alts[2])
        .sel(latitude=slice(-62.5, 62.5))
        .sel(latitude=slice(-10, 10))
        .mean("latitude")
    )

    ax_double5 = ax5.twinx()
    da_line5.plot(
        ax=ax_double5,
        linewidth=2,
        color="black",
        linestyle="dotted",
        label="Mean 10 $^\circ$S to 10 $^\circ$N",
    )
    ax_double5.set_title("")
    ax_double5.set_ylim(-5, 5)
    ax_double5.set_ylabel("")

    da_sig_line5 = filter_to_maximum_anomaly(da_line5)
    ax_double5.scatter(
        da_sig_line5.time.values, da_sig_line5.data, color="black", marker="*", s=100
    )

    # Lines
    ax5.axhline(linewidth=0.7, color="black")

    ax5.axvline(
        x=datetime.datetime(2022, 1, 15), ymin=0, ymax=1, linewidth=1.5, color="black"
    )
    ax5.plot(datetime.datetime(2022, 1, 15), -20.32, "k^", markersize=8)
    ax5.text(datetime.datetime(2020, 5, 1), -27.32, "Hunga", color="black", fontsize=14)

    # Set the locator
    min_locator = mdates.MonthLocator()
    max_locator = mdates.YearLocator()
    max_ylocator = MultipleLocator(20)

    ax5.set_ylabel("Latitude")
    ax5.set_ylim(-45, 45)
    ax5.set_xlabel("Time")
    ax5.grid(linestyle="dotted", linewidth=1.5)

    if data_str == "RO":
        ax5.set_title(f"RO temperature resid. @ {plot_alts[2]} km")
    if data_str == "MLS":
        ax5.set_title(f"MLS temperature resid. @ {plot_alts[2]} km")

    ax_double5.legend(fontsize=14)

    da6 = (
        ds.temperature_resid_GLSAR.sel(time=sig_dates[2])
        .sel(latitude=slice(-62.5, 62.5))
        .sel(altitude=slice(17.5, 38))
        .squeeze()
    )
    im6 = ax6.contourf(
        da6.latitude,
        da6.altitude,
        da6.T.data,
        cmap="RdBu_r",
        levels=levels_temp_v,
        extend="both",
    )

    sig6 = significance_test_date_three_sigma(da_full, sig_dates[2])
    lat_flat, alt_flat, sig_flat = prepare_plot_data(sig6)
    ax6.scatter(lat_flat, alt_flat, s=sig_flat * 20, marker="x", color="black")

    ax6.plot(-20.32, 17.5, "k^", markersize=16)
    ax6.axvline(-20.32, ymin=0, ymax=1, linewidth=1.5, color="black")
    ax6.text(-40, 19, "Hunga", color="black", fontsize=14)
    ax6.set_title(f"{sig_dates[2]}")
    ax6.set_ylabel("Altitude (km)")
    ax6.yaxis.set_ticks_position("right")
    ax6.yaxis.set_label_position("right")
    ax6.set_yticks(np.arange(18, 40, 2))
    ax6.set_xlabel("Latitude")
    ax6.set_xticks(np.arange(-60, 80, 20))
    ax6.set_xlim(-45, 45)
    ax6.grid(linestyle="dotted", linewidth=1.5)

    output_str = f"{plot_dir}/{data_str}_residual_temperature_anomalies_monthly_{plot_alts[0]}km_{plot_alts[1]}km_{plot_alts[2]}km_2005-01_2023-12.png"
    plt.savefig(
        f"{output_str}",
        bbox_inches="tight",
        dpi=300,
    )
    print(f"Plot saved to: {output_str}")
    # plt.show()


def filter_to_maximum_anomaly(ts):
    """
    Set all values in the time series to NaN except the one with the maximum absolute anomaly from the mean.
    This function assumes `ts` is an xarray DataArray with a 'time' dimension.

    Parameters:
    ts (xr.DataArray): The input time series data.

    Returns:
    xr.DataArray: The modified time series with all values set to NaN except for the one with the maximum anomaly.
    """
    # Calculate the mean of the time series
    ts_mean = ts.mean(dim="time")

    # Calculate absolute anomalies
    anomalies = abs(ts - ts_mean)

    # Find the maximum anomaly
    max_anomaly = anomalies.max(dim="time")

    # Create a mask for the maximum anomaly
    max_anomaly_mask = anomalies == max_anomaly

    # Apply the mask to retain only the maximum anomaly and set other values to NaN
    ts_filtered = ts.where(max_anomaly_mask, np.nan)

    return ts_filtered


def significance_test_date_three_sigma(da, date):
    """
    Tests the significance of values at a specific date in an xr dataset
    across all latitudes and altitudes, based on a 3-sigma rule.

    Parameters:
    - ds: Xarray dataset with dimensions time, latitude, and altitude.
    - date: String or datetime-like object specifying the date to test.

    Returns:
    - A subset of the original Xarray dataset for the specific date, with non-significant
      values set to np.nan.
    """
    # Calculate mean and standard deviation across the 'time' dimension
    ts_mean = da.mean("time")
    ts_std = da.std("time")

    # Select values for the specific date
    values_at_date = da.sel(time=date)

    # Identify values outside the mean ± 3*std
    significant_mask = (values_at_date > ts_mean + 3 * ts_std) | (
        values_at_date < ts_mean - 3 * ts_std
    )

    # Replace non-significant values with np.nan
    significant_values = values_at_date.where(significant_mask, np.nan)
    significant_values = significant_values.where(significant_values.isnull(), 1)

    return significant_values


def prepare_plot_data(da):
    """
    Prepare data for plotting, indicating significant values.

    Parameters:
    - ds: Xarray data_array with dimensions [latitude, time, altitude]

    Returns:
    - lat_flat: Flattened array of latitude values.
    - alt_flat: Flattened array of altitude values.
    - temp_flat: Flattened array of temperature data where non-NaN values are
                 replaced with 1 (indicating significance) and NaNs remain unchanged.
    """

    if len(da.time) > 1:
        raise ValueError("Dataset contains more than one time point.")

    temp_data = da

    lat, alt = np.meshgrid(da.latitude, da.altitude, indexing="ij")

    # Flatten the latitude, altitude, and temperature arrays
    lat_flat = lat.flatten()
    alt_flat = alt.flatten()
    temp_flat = temp_data.values.flatten()

    # Replace non-NaN values with 1
    temp_flat[~np.isnan(temp_flat)] = 1

    return lat_flat, alt_flat, temp_flat


if __name__ == "__main__":

    main()
