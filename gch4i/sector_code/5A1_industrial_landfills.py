# %%
# Name: 5A1_landfills.py

# Authors Name: N. Kruskamp, H. Lohman (RTI International), Erin McDuffie (EPA/OAP)
# Date Last Modified: 5/21/2024
# Purpose: Spatially allocates methane emissions for source category 2C2 Ferroalloy production
#
# Input Files:
#      - State_Ferroalloys_1990-2021.xlsx, SubpartK_Ferroalloy_Facilities.csv,
#           all_ghgi_mappings.csv, all_proxy_mappings.csv
# Output Files:
#      - f"{INDUSTRY_NAME}_ch4_kt_per_year.tif, f"{INDUSTRY_NAME}_ch4_emi_flux.tif"
# Notes:
# TODO: update to use facility locations from 2024 GHGI state inventory files
# TODO: include plotting functionaility
# TODO: include netCDF writting functionality

# ---------------------------------------------------------------------
# %% STEP 0. Load packages, configuration files, and local parameters

import calendar
import warnings
from pathlib import Path

import osgeo  # noqa
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.enums
import seaborn as sns
from IPython.display import display

# from pytask import Product, task
from rasterio.features import rasterize

from gch4i.config import (
    ghgi_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
    data_dir_path,
)
from gch4i.gridding import ARR_SHAPE, GEPA_PROFILE
from gch4i.utils import (
    calc_conversion_factor,
    load_area_matrix,
    write_tif_output,
    write_ncdf_output,
    tg_to_kt,
)
# %%