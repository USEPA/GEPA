"""
Name:                   1B1a_energy_abandoned_coal.py
Date Last Modified:     2024-06-07
Authors Name:           
Purpose:                Spatially allocates methane emissions for source category {xxxx},
                        sector {xxxx}, source {xxxx}.
Input Files:            - 
Output Files:           - 
Notes:                  - 
"""

# %% STEP 0. Load packages, configuration files, and local parameters

import calendar
import warnings
from pathlib import Path

import pandas as pd
import osgeo  # noqa
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.enums
import seaborn as sns
from IPython.display import display
import duckdb

# from pytask import Product, task
from rasterio.features import rasterize

from gch4i.config import (
    ghgi_data_dir_path,
    global_data_dir_path,
    max_year,
    min_year,
    gridded_data_path,
    data_dir_path,
)
from gch4i.gridding import ARR_SHAPE, GEPA_PROFILE
from gch4i.utils import (
    calc_conversion_factor,
    load_area_matrix,
    write_tif_output,
    write_ncdf_output,
    tg_to_kt,
    name_formatter,
)

# https://www.epa.gov/system/files/documents/2024-02/us-ghg-inventory-2024-main-text.pdf
IPCC_ID = " "
SECTOR_NAME = " "
SOURCE_NAME = " "
FULL_NAME = "_".join([IPCC_ID, SECTOR_NAME, SOURCE_NAME]).replace(" ", "_")

netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
netcdf_description = (
    f"Gridded EPA Inventory - {SECTOR_NAME} - "
    f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID}"
)
# OUTPUT FILES (TODO: include netCDF flux files)
ch4_kt_dst_path = gridded_data_path / f"{FULL_NAME}_ch4_kt_per_year.tif"
ch4_flux_dst_path = gridded_data_path / f"{FULL_NAME}_ch4_emi_flux.tif"

area_matrix = load_area_matrix()

# %% STEP 1. Load GHGI-Proxy Mapping Files

# %% STEP 2: READ IN EPA INVENTORY EMISSIONS BY STATE / YEAR

# %% STEP 3: GET AND FORMAT PROXY DATA

# %% STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES

# %% STEP 4.1: QC PROXY EMISSIONS ALLOCATION BY STATE / YEAR

# %% STEP 5: RASTERIZE THE CH4 KT AND FLUX

# %% STEP 5.1: QC GRIDDED EMISSIONS ALLOCATION BY STATE / YEAR

# %% STEP 6: SAVE TIF AND NETCDF FILES

# %% STEP 7: PLOT THE DATA FOR AND SAVE VISUALS
