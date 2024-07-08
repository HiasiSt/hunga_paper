# Scripts for the paper "Strong Persistent Cooling of the Stratosphere after the Hunga Eruption" by Stocker et al. (2024)

This repository contains scripts used to reproduce the figures presented in the paper by Stocker et al. (2024) titled "Strong persistent cooling of the stratosphere after the Hunga eruption". These scripts process and visualize data to analyze the impact of the Hunga eruption on the stratosphere.

## Repository Structure

- `plot_RO_profiles_and_MLS_gridpints_within_water_vapor_plume.py`
- `plot_monthly_residual_temperatrue_anomalies_from_RO_and_MLS.py`
- `plot_Hunga_daily_RO_and_MLS_temperature.py`
- `plot_tropical_vs_extratropical_temperature_anomalies.py`
- `plot_RO_profiles_2022-01-15_within_early_eruption_plume.py`

## Prerequisites

- Python 3.6 or higher
- Required Python packages (can be installed via `requirements.txt`):
  - `numpy`
  - `xarray`
  - `matplotlib`
  - `scipy`
  - `pandas`
  - `argparse`
  - `rioxarray`
  - `cartopy`
  
## Data Availability

The data required to run these scripts is available from the following Zenodo repository: [10.5281/zenodo.12682814](https://doi.org/10.5281/zenodo.12682814)

## Scripts Overview and Usage

### 1. plot_RO_profiles_and_MLS_gridpints_within_water_vapor_plume.py

This script plots temperature anomaly profiles/gridpoints from RO/MLS data within the early Hunga water vapor plume.

**Usage:**
```
python plot_RO_profiles_and_MLS_gridpints_within_water_vapor_plume.py --data_dir <path_to_data> --plot_dir <path_to_save_plots> --start_date <YYYY-MM-DD> --end_date <YYYY-MM-DD>
```

### 2. plot_monthly_residual_temperatrue_anomalies_from_RO_and_MLS.py

This script plots the monthly residual temperature anomalies from MLS and RO after removing natural variability using multiple linear regression analysis.

**Usage:**
```
python plot_monthly_residual_temperatrue_anomalies_from_RO_and_MLS.py --input_data_dir <path_to_data> --plot_dir <path_to_save_plots> --meas_str <RO_or_MLS> --altitudes <alt1> <alt2> <alt3> --dates <date1> <date2> <date3>
```

### 3. plot_Hunga_daily_RO_and_MLS_temperature.py

This script processes and visualizes daily temperature anomalies, residual temperature anomalies after removing natural variability, and reconstructed natural temperature variability following the Hunga eruption.

**Usage:**
```
python plot_monthly_residual_temperatrue_anomalies_from_RO_and_MLS.py --input_data_dir <path_to_data> --plot_dir <path_to_save_plots> --meas_str <RO_or_MLS> --altitudes <alt1> <alt2> <alt3> --dates <date1> <date2> <date3>
```

### 4. plot_tropical_vs_extratropical_temperature_anomalies.py

This script generates a scatter plot of tropical vs. extratropical temperature anomalies.
**Usage:**
```
python plot_tropical_vs_extratropical_temperature_anomalies.py --data_dir <path_to_data> --plot_dir <path_to_save_plots> --alt_start <start_altitude_km> --alt_end <end_altitude_km>
```

### 5. plot_RO_profiles_2022-01-15_within_early_eruption_plume.py

This script plots RO temperature/bending angle anomaly profiles within the early Hunga water vapor plume.

**Usage:**
```
python plot_RO_profiles_2022-01-15_within_early_eruption_plume.py --data_dir <path_to_data> --plot_dir <path_to_save_plots> --variable <temperature_anom_or_bending_angle_anom>

```

## Example calls for creating the figures from Stocker et al. (2024)

#### Figure 1
```
python plot_RO_profiles_and_MLS_gridpints_within_water_vapor_plume.py --data_dir ./data --plot_dir ./plots --start_date 2022-01-16 --end_date 2022-03-01
```

#### Figure 2
```
python plot_Hunga_daily_RO_and_MLS_temperature.py ./data ./plots --meas_str RO --variable temperature_anomalies --overlay wind --add_tropopause
```

#### Figure 3
```
python plot_Hunga_daily_RO_and_MLS_temperature.py ./data ./plots --meas_str RO --variable reconstr_natural_variability --overlay wind --add_tropopause
```

#### Figure 4
```
python plot_monthly_residual_temperatrue_anomalies_from_RO_and_MLS.py --input_data_dir ./data --plot_dir ./plots --meas_str RO

```
#### Figure 5
```
python plot_Hunga_daily_RO_and_MLS_temperature.py ./data ./plots --meas_str RO --variable residual_temperature_anomalies --overlay water_vapor --add_tropopause

```
#### Figure S1
```
python plot_RO_profiles_2022-01-15_within_early_eruption_plume.py --data_dir ./data --plot_dir ./plots --variable temperature_anom

```
#### Figure S2
```
python plot_Hunga_daily_RO_and_MLS_temperature.py ./data ./plots --meas_str RO --variable residual_temperature_anomalies --overlay ozone --add_tropopause

```
#### Figure S3
```
python plot_tropical_vs_extratropical_temperature_anomalies.py --data_dir ./data --plot_dir ./plots --alt_start 30 --alt_end 35

```
#### Figure S4
```
python plot_Hunga_daily_RO_and_MLS_temperature.py ./data ./plots --meas_str RO --variable residual_temperature_anomalies --overlay aerosol --add_tropopause

```
#### Figure S5
```
python plot_Hunga_daily_RO_and_MLS_temperature.py ./data ./plots --meas_str MLS --variable residual_temperature_anomalies --overlay water_vapor --add_tropopause

```

## Contact

For any questions or issues, please contact matthias.stocker(at)uni-graz.at.




