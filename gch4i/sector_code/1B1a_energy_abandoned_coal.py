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
SOURCE_NAME = "Abandoned Underground Coal Mines"
FULL_NAME = "_".join([IPCC_ID, SECTOR_NAME, SOURCE_NAME]).replace(" ", "_")

netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
netcdf_description = (
    f"Gridded EPA Inventory - {SECTOR_NAME} - "
    f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID}"
)

# PATHS
sector_dir = V3_DATA_PATH / "sector/abandoned_mines"

# OUTPUT FILES
ch4_kt_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_kt_per_year"
ch4_flux_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_emi_flux"

# INVENTORY INPUT FILE
inventory_workbook_path = (
    ghgi_data_dir_path / "coal/AbandonedCoalMines1990-2022_FRv1.xlsx"
)

# PROXY INPUT FILES
# state vector refence with state_code
state_geo_path = global_data_dir_path / "tl_2020_us_state.zip"
county_url = "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip"
# input 1: NOTE: mine list comes from inventory workbook
# input 2: abandoned mines address list
# downloaded manually from:
# https://www.msha.gov/data-and-reports/mine-data-retrieval-system
# scroll down to section "Explore MSHA Datasets"
# select from drop down: "13 Mines Data Set"
msha_path = sector_dir / "Mines.zip"


# %% STEP 1. Load GHGI-Proxy Mapping Files
# TODO: make this work...

proxy_file_path = Path(
    "C:/Users/nkruskamp/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/v3_ghgrp_data_from_sector_leads.xlsx"
)

proxy_mapping = pd.read_excel(proxy_file_path, sheet_name="proxy_emi_mapping").query(
    "Category == 'Abandoned Coal Mines'"
)
proxy_mapping


# %%


# %% STEP 2: Read In EPA State GHGI Emissions by Year ----------------------------------


EPA_state_liberated_emi_df = pd.read_csv(emi_data_dir_path / "abd_coal_lib_emi.csv")
EPA_state_randu_emi_df = pd.read_csv(
    emi_data_dir_path / "abd_coal_rec_and_used_emi.csv"
)

# plot state inventory data
sns.relplot(
    kind="line",
    data=EPA_state_liberated_emi_df,
    x="year",
    y="ghgi_ch4_kt",
    hue="state_code",
    palette="tab20",
    # legend=False,
)

# plot state inventory data
sns.relplot(
    kind="line",
    data=EPA_state_randu_emi_df,
    x="year",
    y="ghgi_ch4_kt",
    hue="state_code",
    palette="tab20",
    # legend=False,
)

# %% STEP 3: GET AND FORMAT PROXY DATA -------------------------------------------------
ab_coal_proxy_gdf = get_ab_coal_mine_proxy_data(
    inventory_workbook_path, msha_path, county_url, state_geo_path
)

# %% STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES ---------------------------
allocated_liberated_emis_gdf = allocate_emissions_to_proxy(
    ab_coal_proxy_gdf,
    EPA_state_liberated_emi_df,
    proxy_has_year=True,
    use_proportional=True,
    proportional_col_name="emis_mmcfd",
)

allocated_randu_emis_gdf = allocate_emissions_to_proxy(
    ab_coal_proxy_gdf,
    EPA_state_randu_emi_df,
    proxy_has_year=True,
    use_proportional=True,
    proportional_col_name="emis_mmcfd",
)

# %% STEP 4.1: QC PROXY ALLOCATED EMISSIONS BY STATE AND YEAR --------------------------
proxy_qc_liberated_result = QC_proxy_allocation(
    allocated_liberated_emis_gdf, EPA_state_liberated_emi_df
)
proxy_qc_liberated_result

# %%
proxy_qc_randu_result = QC_proxy_allocation(
    allocated_randu_emis_gdf, EPA_state_randu_emi_df
)
proxy_qc_randu_result

# %% STEP 5: RASTERIZE THE CH4 KT AND FLUX ---------------------------------------------
ch4_kt_liberated_result_rasters = grid_allocated_emissions(allocated_liberated_emis_gdf)

ch4_kt_randu_result_rasters = grid_allocated_emissions(allocated_randu_emis_gdf)

# %% STEP 5.1: QC GRIDDED EMISSIONS BY YEAR --------------------------------------------
# TODO: report QC metrics for flux values compared to V2: descriptive statistics
qc_kt_liberated_rasters = QC_emi_raster_sums(
    ch4_kt_liberated_result_rasters, EPA_state_liberated_emi_df
)
qc_kt_liberated_rasters
# %%
qc_kt_randu_rasters = QC_emi_raster_sums(
    ch4_kt_randu_result_rasters, EPA_state_randu_emi_df
)
qc_kt_randu_rasters

# %% STEP 5.2: COMBINE SUBSOURCE RASTERS TOGETHER --------------------------------------
ch4_kt_result_rasters = combine_gridded_emissions(
    [ch4_kt_liberated_result_rasters, ch4_kt_randu_result_rasters]
)
ch4_flux_result_rasters = calculate_flux(ch4_kt_result_rasters)

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
