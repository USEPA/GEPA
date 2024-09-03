"""
Name:                   2B8_petrochemical_production.py
Date Last Modified:     2024-08-28
Authors Name:           N. Kruskamp, H. Lohman (RTI International), Erin McDuffie
                        (EPA/OAP)
Purpose:                Spatially allocates methane emissions for source category 5A,
                        sector Waste, 
                        source Landfills.
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
    QC_flux_emis
)

# from pytask import Product, task

gpd.options.io_engine = "pyogrio"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)


# %%
# https://www.epa.gov/system/files/documents/2024-02/us-ghg-inventory-2024-main-text.pdf
IPCC_ID = "5A"
SECTOR_NAME = "Waste"
# SOURCE_NAME = "Landfills"
SOURCE_NAME = "Industrial Waste Landfills Emissions"
FULL_NAME = "_".join([IPCC_ID, SECTOR_NAME, SOURCE_NAME]).replace(" ", "_")
GCH4I_NAME = "5A_industrial_landfills"

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

# proxy_mapping = pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping").query(
#     # f"Category == '{SOURCE_NAME}'"
#     f"Subcategory1 == '{SOURCE_NAME}'"
# )
proxy_mapping = pd.read_excel(proxy_file_path, sheet_name="gridding_flag_info").query(
    f"gch4i_name == '{GCH4I_NAME}'"
)
proxy_mapping


# %% READ IN EPA DATA, ALLOCATE, GRID, AND QC ----------------------------------


emi_dict = {}
for mapping_row in proxy_mapping.itertuples():
    mapping_row
    emi_name = mapping_row.emi_id
    proxy_name = mapping_row.proxy_id
    yearly_flag = mapping_row.yearly_flag
    proxy_use_prop = mapping_row.proxy_use_prop
    prop_col_name = mapping_row.prop_col_name
    v2_name = mapping_row.v2_name


    # STEP 2: Read In EPA State GHGI Emissions by Year
    # EEM: question -- can we create a script that we can run separately to run all the 
    #  get_emi and get_proxy functions? Then we can include a comment in this
    # script that states that those functions need to be run first
    # Also see comments on the emissions script. The emission values need to be corrected
    emi_dict[emi_name] = {}
    emi_path = list(emi_data_dir_path.glob(f"{emi_name}*"))[0]
    emi_dict[emi_name]["emi_id"] = pd.read_csv(emi_path)
    
    # STEP 3: GET AND FORMAT PROXY DATA ---------------------------------------------
    proxy_path = list(proxy_data_dir_path.glob(f"{proxy_name}*.parquet"))[0]
    emi_dict[emi_name]["proxy_id"] = gpd.read_parquet(proxy_path)
    
    # STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES -----------------------
    emi_dict[emi_name]["allocated"] = allocate_emissions_to_proxy(
        emi_dict[emi_name]["proxy_id"],
        emi_dict[emi_name]["emi_id"],
        proxy_has_year=yearly_flag,
        use_proportional=proxy_use_prop,
        proportional_col_name=prop_col_name
        #proportional_col_name="capacity_kt",
    )

    # STEP X: QC ALLOCATION ---------------------------------------------------------
    emi_dict[emi_name]["allocation_qc"] = QC_proxy_allocation(
        emi_dict[emi_name]["allocated"], emi_dict[emi_name]["emi_id"]
    )
    # STEP X: GRID EMISSIONS --------------------------------------------------------
    # emi_dict[emi_name]["rasters"] = grid_allocated_emissions(
    #     emi_dict[emi_name]["allocated"]
    # )
    # STEP X: QC GRIDDED EMISSIONS --------------------------------------------------
    # emi_dict[emi_name]["raster_qc"] = QC_emi_raster_sums(
    #     emi_dict[emi_name]["rasters"], emi_dict[emi_name]["emi_id"]
    # )


# %% STEP 5.2: COMBINE SUBSOURCE RASTERS TOGETHER --------------------------------------
raster_list = [emi_dict[emi_name]["rasters"] for emi_name in emi_dict.keys()]
ch4_kt_result_rasters = combine_gridded_emissions(raster_list)
ch4_flux_result_rasters = calculate_flux(ch4_kt_result_rasters)


# %% STEP 5.2: QC FLUX AGAINST V2 ------------------------------------------------------
flux_emi_qc_df = QC_flux_emis(ch4_flux_result_rasters, SOURCE_NAME, v2_name=v2_name)
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
