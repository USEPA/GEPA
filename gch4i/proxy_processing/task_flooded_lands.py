"""
Author:     Nick Kruskamp
Date:       2021-07-26
exsum:      This script is used to process the flooded lands data into the proxy data.
"""


# %%
%load_ext autoreload
%autoreload 2

import multiprocessing
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import osgeo  # noqa
import rasterio
import rioxarray
import xarray as xr
from geocube.api.core import make_geocube
from pyarrow import parquet  # noqa
from pytask import Product, mark, task

from gch4i.config import (
    global_data_dir_path,
    proxy_data_dir_path,
    sector_data_dir_path,
    years,
)
from gch4i.gridding import make_raster_binary, mask_raster_parallel, warp_to_gepa_grid

NUM_WORKERS = multiprocessing.cpu_count()


# define a function to normalize the population data by state and year
def normalize(x):
    return x / x.sum()
# %%

fl_data_path = sector_data_dir_path / "flooded_lands"
list(fl_data_path.rglob("*"))
# %%
