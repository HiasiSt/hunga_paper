#!/usr/bin/env python
# coding: utf-8

"""
This script processes and visualizes temperature anomalies, residual
temperature anomalies after natural variability removal, as well as
reconstructed natural variability following the Hunga eruption.
Additionally, it allows for various overlays (e.g., water vapor, wind)
to be chosen.

The script supports selecting different measurement sources
(e.g., RO or MLS) and overlay options (e.g., wind, ozone, aerosol,
water vapor).

Author: Matthias Stocker [matthias.stocker(at)uni-graz.at]
"""

import numpy as np
import xarray as xr
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.dates as mdates
from matplotlib import gridspec
import argparse


def main():

    args = parse_arguments()

    data_dir = args.data_dir
    plot_dir = args.plot_dir
    meas_str = args.meas_str

    variable = args.variable
    overlay = args.overlay

    plot_alts = args.altitudes
    add_tropopause = args.add_tropopause

    if meas_str == "RO":
        dataset_path = f"{data_dir}/RO_dry_temperature_results_daily_lat_band_2021-12-01_to_2023-12-31.nc"
    elif meas_str == "MLS":
        dataset_path = f"{data_dir}/MLS_temperature_results_daily_lat_band_2021-12-01_to_2023-12-31.nc"

    ds = xr.open_dataset(dataset_path)

    plot_temperature(
        ds,
        meas_str,
        data_dir,
        variable,
        plot_alts,
        overlay,
        plot_dir,
        add_tropopause=add_tropopause,
    )


def plot_temperature(
    ds, meas_str, data_dir, variable, plot_alts, overlay, plot_dir, add_tropopause=False
):

    ds = ds.sel(time=slice('2021-12-01', '2023-12-16'))

    if meas_str == "RO":
        dry_str = "dry_"
    else:
        dry_str = ''

    if variable == "temperature_anomalies":
        variable = f"{meas_str}_{dry_str}temperature_anomalies"
        plot_str = f"{meas_str} temperature anomalies"
        file_str = "temperature_anom_daily"

    if variable == "residual_temperature_anomalies":
        variable = f"{meas_str}_residual_{dry_str}temperature_anomalies"
        plot_str = f"Residual {meas_str} temperature anomalies"
        file_str = "residual_temperature_anom_daily"

    if variable == "reconstr_natural_variability":
        variable = f"{meas_str}_reconstr_natural_variability"
        plot_str = f"Reconstructed natural variability"
        file_str = "reconstr_natural_variability"

    ticks_temp = np.arange(-4, 5, 1)
    levels_temp = np.linspace(-4.25, 4.25, 18)

    if add_tropopause:
        ds_trp = xr.open_dataset(f"{data_dir}/RO_climatological_lrt.nc")
        da_trp = ds_trp.lrt_dry_temperature_climatology_altitude
        da_trp = da_trp / 1000.0

    lat_mean = [-30, 10]

    if overlay == "wind":
        ds_wind = xr.open_dataset(f"{data_dir}/GSFC_singapore_wind_anomalies.nc")
        ds_wind = ds_wind.sel(altitude=slice(16, 38))
        ds_wind = ds_wind.sel(time=slice(ds.time.min(), ds.time.max()))

    if overlay == "ozone":
        ds_o3 = xr.open_dataset(f"{data_dir}/MLS_O3_anomalies_lat_band.nc")
        ds_o3["altitude"] = ds_o3.altitude / 1000.0
        ds_o3 = ds_o3.sel(altitude=slice(17.5, 38)).sel(latitude=slice(-67.5, 67.5))
        ds_o3 = ds_o3.sel(time=slice(ds.time.min(), ds.time.max()))

    if overlay == "aerosol":
        ds_aer = xr.open_dataset(f"{data_dir}/OMPS-LP_aerosol_anomalies_lat_band.nc")
        ds_aer["altitude"] = ds_aer.altitude / 1000.0
        ds_aer = ds_aer.sel(altitude=slice(16, 38))
        ds_aer = ds_aer.sel(altitude=slice(16, 38)).sel(
            time=slice(ds.time.min(), ds.time.max()))

    if overlay == "water_vapor":
        ds_wv = xr.open_dataset(f"{data_dir}/MLS_water_vapor_anomalies_lat_band.nc")
        ds_wv["altitude"] = ds_wv.altitude / 1000.0
        ds_wv = ds_wv.sel(altitude=slice(16, 38)).sel(
            time=slice(ds.time.min(), ds.time.max()))

    ds["altitude"] = ds.altitude / 1000.0
    ds = ds.sel(altitude=slice(16, 38))

    # Figure setup
    fmt1 = matplotlib.ticker.LogFormatterSciNotation()
    fmt1.create_dummy_axis()
    plt.rcParams.update({"font.size": 16})

    fig = plt.figure(figsize=(15, 21))

    gsouter = gridspec.GridSpec(2, 1, height_ratios=[0.4, 1])
    spec0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gsouter[0])
    spec1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gsouter[1], hspace=0.1)
    ax1 = fig.add_subplot(spec0[0, 0])
    ax2 = fig.add_subplot(spec1[0, 0])
    ax3 = fig.add_subplot(spec1[1, 0])
    ax4 = fig.add_subplot(spec1[2, 0])

    ds[variable].sel(latitude=slice(lat_mean[0], lat_mean[1])).mean(
        "latitude"
    ).plot.contourf(
        x="time",
        y="altitude",
        ax=ax1,
        vmin=-4,
        vamx=4,
        cmap="RdBu_r",
        levels=levels_temp,
        cbar_kwargs={
            "label": "Temperature anom. (K)",
            "ticks": ticks_temp,
            "orientation": "vertical",
        },
        extend="both",
    )
    if add_tropopause:
        ax1.plot(
            da_trp.time, da_trp.data, linestyle="dashed", color="black", linewidth=2
        )

    if overlay == "wind":
        CS1 = ds_wind.singapore_wind_anomalies.plot.contour(
            x="time",
            y="altitude",
            ax=ax1, levels=15, cmap="BrBG"
        )
        ax1.clabel(CS1, CS1.levels, inline=True, fmt=fmt, fontsize=10)
    elif overlay == "water_vapor":
        da = ds_wv.water_vapor_anomalies.sel(
            latitude=slice(lat_mean[0], lat_mean[1])
        ).mean("latitude")
        da = da * 1000000.0
        CS1 = da.T.plot.contour(
            x="time", y="altitude", ax=ax1, levels=20, cmap="YlGnBu", linewidths=1.5
        )
        ax1.clabel(CS1, CS1.levels, inline=2, fontsize=12, fmt=fmt_o3)
    elif overlay == "ozone":
        da = ds_o3.O3.sel(latitude=slice(lat_mean[0], lat_mean[1])).mean("latitude")
        da = da * 1000000.0
        CS1 = da.T.plot.contour(
            x="time", y="altitude", ax=ax1, levels=15, cmap="BrBG_r", linewidths=2
        )
        ax1.clabel(CS1, CS1.levels, inline=True, fontsize=12, fmt=fmt_o3)
    elif overlay == "aerosol":
        da = ds_aer.aerosol_anomalies.sel(
            latitude=slice(lat_mean[0], lat_mean[1])
        ).mean("latitude")
        CS1 = da.T.plot.contour(
            x="time",
            y="altitude",
            ax=ax1,
            levels=12,
            cmap="inferno",
            linewidths=2,
            vmin=0,
        )
        ax1.clabel(CS1, CS1.levels, inline=True, fontsize=12, fmt=fmt_aer)
    elif overlay == None:
        print("No overlay.")
    else:
        raise ValueError(f"Overlay {overlay} not implemented.")

    ax1.axvline(
        x=datetime.datetime(2022, 1, 15), ymin=0, ymax=1, linewidth=1.5, color="black"
    )
    ax1.text(datetime.datetime(2022, 1, 15), 32, "Hunga", color="black")

    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter("%Y-%m")
    days = mdates.DayLocator()

    ax1.xaxis.set_major_locator(months)
    ax1.xaxis.set_major_formatter(monthsFmt)
    ax1.xaxis.set_minor_locator(days)

    ax1.set_ylabel("Altitude (km)")
    ax1.set_xlabel("Time")
    ax1.set_yticks(np.arange(16, 40, 2))
    ax1.set_xlim(ds.time.min(), ds.time.max())
    ax1.grid(linestyle="dotted", linewidth=1.5)
    ax1.set_title(
        f"{plot_str} (mean {abs(lat_mean[0])}$^\circ$S to {abs(lat_mean[1])}$^\circ$N) "
    )

    ds[variable].sel(latitude=slice(-67.5, 67.5)).sel(
        altitude=plot_alts[0]
    ).plot.contourf(
        x="time",
        y="latitude",
        ax=ax2,
        vmin=-6,
        vamx=6,
        cmap="RdBu_r",
        levels=levels_temp,
        cbar_kwargs={
            "label": "Temperature anom. (K)",
            "ticks": ticks_temp,
            "orientation": "vertical",
        },
        extend="both",
    )
    if overlay == "water_vapor":
        da = ds_wv.water_vapor_anomalies.sel(altitude=plot_alts[0])
        da = da * 1000000.0
        CS2 = da.plot.contour(
            x="time", y="latitude", ax=ax2, levels=25, cmap="YlGnBu", linewidths=1.5
        )
        ax2.clabel(CS2, CS2.levels, inline=2, fontsize=12, fmt=fmt_o3)
    elif overlay == "ozone":
        da = ds_o3.O3.sel(altitude=plot_alts[0])
        da = da * 1000000.0
        CS2 = da.plot.contour(
            x="time",
            y="latitude",
            ax=ax2,
            levels=np.linspace(-0.3, 0.16, 12),
            cmap="BrBG_r",
            linewidths=2,
        )
        ax2.clabel(CS2, CS2.levels, inline=True, fontsize=12, fmt=fmt_o3)
    elif overlay == "aerosol":
        da = ds_aer.aerosol_anomalies.sel(altitude=plot_alts[0])
        CS2 = da.plot.contour(
            x="time",
            y="latitude",
            ax=ax2,
            levels=12,
            cmap="inferno",
            linewidths=2,
            vmin=0,
        )
        ax2.clabel(CS2, CS2.levels, inline=True, fontsize=12, fmt=fmt_aer)

    ax2.axvline(
        x=datetime.datetime(2022, 1, 15), ymin=0, ymax=1, linewidth=1.5, color="black"
    )
    ax2.plot(datetime.datetime(2022, 1, 15), -20.32, "k^", markersize=10)
    ax2.text(datetime.datetime(2022, 1, 15), -17.5, "Hunga", color="black")
    ax2.axhline(0, linewidth=1.5, color="black", linestyle="dashed")

    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter("")

    ax2.xaxis.set_major_locator(months)
    ax2.xaxis.set_major_formatter(monthsFmt)
    ax2.xaxis.set_minor_locator(days)

    minorLocator = MultipleLocator(2.5)
    ax2.yaxis.set_minor_locator(minorLocator)

    ax2.set_ylabel("Latitude")
    ax2.set_ylim(-45, 45)
    ax2.set_xlabel("")
    ax2.grid(linestyle="dotted", linewidth=1.5)
    ax2.set_title(f"{plot_str} @ {plot_alts[0]} km")

    ds[variable].sel(latitude=slice(-67.5, 67.5)).sel(
        altitude=plot_alts[1]
    ).plot.contourf(
        x="time",
        y="latitude",
        ax=ax3,
        vmin=-6,
        vamx=6,
        cmap="RdBu_r",
        levels=levels_temp,
        cbar_kwargs={
            "label": "Temperature anom. (K)",
            "ticks": ticks_temp,
            "orientation": "vertical",
        },
        extend="both",
    )
    if overlay == "water_vapor":
        da = ds_wv.water_vapor_anomalies.sel(altitude=plot_alts[1])
        da = da * 1000000.0
        CS3 = da.plot.contour(
            x="time", y="latitude", ax=ax3, levels=25, cmap="YlGnBu", linewidths=1.5
        )
        ax3.clabel(CS3, CS3.levels, inline=2, fontsize=12, fmt=fmt_o3)
    elif overlay == "ozone":
        da = ds_o3.O3.sel(altitude=plot_alts[1])
        da = da * 1000000.0
        CS3 = da.plot.contour(
            x="time",
            y="latitude",
            ax=ax3,
            levels=np.linspace(-1, 0.8, 12),
            cmap="BrBG_r",
            linewidths=2,
        )
        ax3.clabel(CS3, CS3.levels, inline=True, fontsize=12, fmt=fmt_o3)
    elif overlay == "aerosol":
        da = ds_aer.aerosol_anomalies.sel(altitude=plot_alts[1])
        CS3 = da.plot.contour(
            x="time",
            y="latitude",
            ax=ax3,
            levels=12,
            cmap="inferno",
            linewidths=2,
            vmin=0,
        )
        ax3.clabel(CS3, CS3.levels, inline=True, fontsize=12, fmt=fmt_aer)

    ax3.axvline(
        x=datetime.datetime(2022, 1, 15), ymin=0, ymax=1, linewidth=1.5, color="black"
    )
    ax3.axhline(0, linewidth=1.5, color="black", linestyle="dashed")
    ax3.plot(datetime.datetime(2022, 1, 15), -20.32, "k^", markersize=10)
    ax3.text(datetime.datetime(2022, 1, 15), -17.5, "Hunga", color="black")

    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter("")

    ax3.xaxis.set_major_locator(months)
    ax3.xaxis.set_major_formatter(monthsFmt)
    ax3.xaxis.set_minor_locator(days)

    minorLocator = MultipleLocator(2.5)
    ax3.yaxis.set_minor_locator(minorLocator)
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax3.set_ylabel("Latitude")
    ax3.set_ylim(-45, 45)
    ax3.set_xlabel("")
    ax3.grid(linestyle="dotted", linewidth=1.5)
    ax3.set_title(f"{plot_str} @ {plot_alts[1]} km")

    ds[variable].sel(latitude=slice(-67.5, 67.5)).sel(
        altitude=plot_alts[2]
    ).plot.contourf(
        x="time",
        y="latitude",
        ax=ax4,
        vmin=-6,
        vamx=6,
        cmap="RdBu_r",
        levels=levels_temp,
        cbar_kwargs={
            "label": "Temperature anom. (K)",
            "ticks": ticks_temp,
            "orientation": "vertical",
        },
        extend="both",
    )

    if overlay == "water_vapor":
        da = ds_wv.water_vapor_anomalies.sel(altitude=plot_alts[2])
        da = da * 1000000.0
        CS4 = da.plot.contour(
            x="time", y="latitude", ax=ax4, levels=25, cmap="YlGnBu", linewidths=1.5
        )
        ax4.clabel(CS4, CS4.levels, inline=2, fontsize=12, fmt=fmt_o3)
    elif overlay == "ozone":
        da = ds_o3.O3.sel(altitude=plot_alts[2])
        da = da * 1000000.0
        CS4 = da.plot.contour(
            x="time",
            y="latitude",
            ax=ax4,
            levels=np.linspace(-1, 0.8, 12),
            cmap="BrBG_r",
            linewidths=2,
        )
        ax4.clabel(CS4, CS4.levels, inline=True, fontsize=12, fmt=fmt_o3)
    elif overlay == "aerosol":
        da = ds_aer.aerosol_anomalies.sel(altitude=plot_alts[2])
        CS4 = da.plot.contour(
            x="time",
            y="latitude",
            ax=ax4,
            levels=12,
            cmap="inferno",
            linewidths=2,
            vmin=0,
        )
        ax4.clabel(CS4, CS4.levels, inline=True, fontsize=12, fmt=fmt_aer)

    ax4.axvline(
        x=datetime.datetime(2022, 1, 15), ymin=0, ymax=1, linewidth=1.5, color="black"
    )
    ax4.axhline(0, linewidth=1.5, color="black", linestyle="dashed")
    ax4.plot(datetime.datetime(2022, 1, 15), -20.32, "k^", markersize=10)
    ax4.text(datetime.datetime(2022, 1, 15), -17.5, "Hunga", color="black")

    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter("%Y-%m")

    ax4.xaxis.set_major_locator(months)
    ax4.xaxis.set_major_formatter(monthsFmt)
    ax4.xaxis.set_minor_locator(days)

    minorLocator = MultipleLocator(2.5)
    ax4.yaxis.set_minor_locator(minorLocator)

    ax4.set_ylabel("Latitude")
    ax4.set_ylim(-45, 45)
    ax4.set_xlabel("Time")
    ax4.grid(linestyle="dotted", linewidth=1.5)
    ax4.set_title(f"{plot_str} @ {plot_alts[2]} km")

    plt.tight_layout()

    output_str = f"{plot_dir}/Hunga_{meas_str}_{file_str}_mean_{lat_mean[0]}-{lat_mean[1]}_{plot_alts[0]}km_{plot_alts[1]}km_{plot_alts[2]}km_{overlay}.png"
    plt.savefig(output_str, bbox_inches="tight", dpi=300)
    print(f"Plot saved to: {output_str}")

    return


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \m/s" if plt.rcParams["text.usetex"] else f"{s} m/s"


def fmt_o3(x):
    s = f"{x:.2f}"
    if s.endswith("0"):
        s = f"{x:.1f}"
    return rf"{s} \ppmv" if plt.rcParams["text.usetex"] else f"{s} ppmv"


def fmt_aer(x):
    # Formatting the number in standard scientific notation
    s = f"{x:.2e}"

    # Splitting the string to convert 'e+02' to '×10²'
    num, exp = s.split("e")
    exp = int(
        exp
    )  # Convert exponent to integer to remove leading '+' and format properly
    if exp == 0:
        # If the exponent is 0, just show the number without exponentiation
        formatted_number = num
    else:
        # Formatting with '×10' and superscript for the exponent
        exp_str = f"{exp}".translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
        formatted_number = f"{num}×10{exp_str}"

    return f"{formatted_number} km⁻¹"


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText
        )

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r"$\mathdefault{%s}$" % self.format


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to the input dataset file.")
    parser.add_argument(
        "plot_dir", type=str, help="Directory where the plots shall be saved."
    )
    parser.add_argument(
        "--meas_str",
        type=str,
        default="RO",
        choices=["RO", "MLS"],
        help='Measurement string indicating the source (e.g., "RO" or "MLS").',
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="residual_temperature_anomalies",
        choices=[
            "temperature_anomalies",
            "residual_temperature_anomalies",
            "reconstr_natural_variability",
        ],
        help='Variable to plot (e.g.,"temperature_anomalies").',
    )
    parser.add_argument(
        "--overlay",
        type=str,
        default="ozone",
        choices=["wind", "ozone", "aerosol", "water_vapor"],
        help='Overlay to add to the plot (e.g., "wind", "ozone", "aerosol", "water_vapor").',
    )
    parser.add_argument(
        "--altitudes",
        nargs=3,
        type=int,
        default=[19, 27, 32],
        help="Three different altitudes to plot (default: 19, 27, 32).",
    )
    parser.add_argument(
        "--add_tropopause",
        action="store_true",
        help="Add climatological laps rate tropopause to the plot.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    main()
