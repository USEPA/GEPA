"""
Name:                   1B1a_energy_abandoned_coal.py
Date Last Modified:     2024-06-07
Authors Name:           N. Kruskamp, H. Lohman (RTI International), Erin McDuffie
                        (EPA/OAP)
Purpose:                Spatially allocates methane emissions for source category 1B1a,
                        sector energy, source Abandoned Underground Coal Mines.
Input Files:            - 
Output Files:           - 
Notes:                  - 
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------

# for testing/development
# %load_ext autoreload
# %autoreload 2
from zipfile import ZipFile
import calendar
import datetime

import osgeo  # noqa
import duckdb
import geopandas as gpd
import pandas as pd
import seaborn as sns
from geopy.geocoders import Nominatim
from IPython.display import display
import numpy as np

from gch4i.config import (
    V3_DATA_PATH,
    ghgi_data_dir_path,
    proxy_data_dir_path,
    global_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
    emi_data_dir_path,
)
from gch4i.utils import (
    QC_emi_raster_sums,
    QC_proxy_allocation,
    grid_allocated_emissions,
    name_formatter,
    plot_annual_raster_data,
    plot_raster_data_difference,
    allocate_emissions_to_proxy,
    tg_to_kt,
    write_ncdf_output,
    write_tif_output,
    calculate_flux,
    combine_gridded_emissions,
)

# from pytask import Product, task

gpd.options.io_engine = "pyogrio"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)


# %%
# https://www.epa.gov/system/files/documents/2024-02/us-ghg-inventory-2024-main-text.pdf
IPCC_ID = "1B1a"
SECTOR_NAME = "Energy"
SOURCE_NAME = "Abandoned Coal Mines"
FULL_NAME = "_".join([IPCC_ID, SECTOR_NAME, SOURCE_NAME]).replace(" ", "_")

netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
netcdf_description = (
    f"Gridded EPA Inventory - {SECTOR_NAME} - "
    f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID}"
)

# OUTPUT FILES
ch4_kt_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_kt_per_year"
ch4_flux_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_emi_flux"

# %% STEP 1. Load GHGI-Proxy Mapping Files
proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"

proxy_mapping = pd.read_excel(proxy_file_path, sheet_name="proxy_emi_mapping").query(
    f"Category == '{SOURCE_NAME}'"
)
proxy_mapping


# %% STEP 2: Read In EPA State GHGI Emissions by Year ----------------------------------


emi_dict = {}
for mapping_row in proxy_mapping.itertuples():
    mapping_row
    emi_name = mapping_row.emi
    proxy_name = mapping_row.proxy

    emi_dict[emi_name] = {}
    emi_path = list(emi_data_dir_path.glob(f"{emi_name}*"))[0]
    emi_dict[emi_name]["emi"] = pd.read_csv(emi_path)
    # STEP 3: GET AND FORMAT PROXY DATA ---------------------------------------------
    proxy_path = list(proxy_data_dir_path.glob(f"{proxy_name}*.parquet"))[0]
    emi_dict[emi_name]["proxy"] = gpd.read_parquet(proxy_path)
    # STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES -----------------------
    emi_dict[emi_name]["allocated"] = allocate_emissions_to_proxy(
        emi_dict[emi_name]["proxy"],
        emi_dict[emi_name]["emi"],
        proxy_has_year=True,
        use_proportional=True,
        proportional_col_name="emis_mmcfd",
    )
    # STEP X: QC ALLOCATION ---------------------------------------------------------
    emi_dict[emi_name]["allocation_qc"] = QC_proxy_allocation(
        emi_dict[emi_name]["allocated"], emi_dict[emi_name]["emi"]
    )
    # STEP X: GRID EMISSIONS --------------------------------------------------------
    emi_dict[emi_name]["rasters"] = grid_allocated_emissions(
        emi_dict[emi_name]["allocated"]
    )
    # STEP X: QC GRIDDED EMISSIONS --------------------------------------------------
    emi_dict[emi_name]["raster_qc"] = QC_emi_raster_sums(
        emi_dict[emi_name]["rasters"], emi_dict[emi_name]["emi"]
    )


# %% STEP 5.2: COMBINE SUBSOURCE RASTERS TOGETHER --------------------------------------
raster_list = [emi_dict[emi_name]["rasters"] for emi_name in emi_dict.keys()]
ch4_kt_result_rasters = combine_gridded_emissions(raster_list)
ch4_flux_result_rasters = calculate_flux(ch4_kt_result_rasters)


# %% STEP 5.2: QC FLUX AGAINST V2 ------------------------------------------------------
import osgeo
import rioxarray
import rasterio
import xarray as xr


def qc_flux_emis(data: dict, v2_name: str):

    v2_data_paths = V3_DATA_PATH.glob("Gridded_GHGI_Methane_v2_*.nc")
    v2_data_dict = {}
    for in_path in v2_data_paths:
        v2_year = int(in_path.stem.split("_")[-1])
        v2_data = rioxarray.open_rasterio(in_path, variable=v2_name)[
            v2_name
        ].values.squeeze(axis=0)
        v2_data = np.where(v2_data == 0, np.nan, v2_data)
        v2_data_dict[v2_year] = v2_data

    result_list = []
    for year in data.keys():
        if year in v2_data_dict.keys():
            v3_data = np.where(data[year] == 0, np.nan, data[year])
            yearly_dif = data[year] - v2_data_dict[year]
            v2_sum = np.nansum(v2_data)
            v3_sum = np.nansum(v3_data)
            print(f"v2 sum: {v2_sum}, v3 sum: {v3_sum}")
            result_list.append(
                pd.DataFrame(yearly_dif.ravel())
                .dropna(how="all", axis=1)
                .describe()
                .rename(columns={0: year})
            )

        # TODO: save out a map if differences
        # compare sums of arrays by year

    result_df = pd.concat(result_list, axis=1)
    return result_df


flux_emi_qc_df = qc_flux_emis(
    ch4_flux_result_rasters, v2_name="emi_ch4_1B1a_Abandoned_Coal"
)
flux_emi_qc_df
# %% STEP 6: SAVE THE FILES ------------------------------------------------------------
write_tif_output(ch4_kt_result_rasters, ch4_kt_dst_path)
write_tif_output(ch4_flux_result_rasters, ch4_flux_dst_path)
write_ncdf_output(
    ch4_flux_result_rasters,
    ch4_flux_dst_path,
    netcdf_title,
    netcdf_description,
)

# %% STEP 7: PLOT THE RESULTS AND DIFFERENCE, SAVE FIGURES TO FILES --------------------
plot_annual_raster_data(ch4_flux_result_rasters, SOURCE_NAME)
plot_raster_data_difference(ch4_flux_result_rasters, SOURCE_NAME)

# %%
fig, (ax1, ax2) = plt.subplots(2)
fig.tight_layout()
ax1.hist(v2_data.ravel(), bins=100)
ax2.hist(v3_data.ravel() / 1000, bins=100)
plt.show()
# %%
np.nanmax(v3_data)
# %%
np.nanmax(v2_data)

# %%
proxy_path
# %%
