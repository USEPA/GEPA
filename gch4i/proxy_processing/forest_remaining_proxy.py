# %%
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile
import calendar
import datetime

from pyarrow import parquet
from io import StringIO
import pandas as pd
import duckdb
import osgeo
import geopandas as gpd
import numpy as np
import seaborn as sns
from pytask import Product, task, mark

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    emi_data_dir_path,
    global_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year,
    years
)

# %%

# Read in emi data
forest_land_emi = pd.read_csv(emi_data_dir_path / "forest_land_emi.csv")

# Read in proxy data
forest_land_proxy = pd.read_csv(proxy_data_dir_path / "fire/MTBS_byEventFuelFuelbed_09Sep2024.csv")

# Read in fuelbed crosswalk data
fccs_fuelbed = pd.read_csv(proxy_data_dir_path / "fire/fccs_fuelbed_Aug2023_jesModified.csv")
nawfd_fuelbed = pd.read_csv(proxy_data_dir_path / "fire/nawfd_fuelbed_Aug2023_jesModified.csv")

# %% Functions


