# a collection of functions that do standard gridding
from pathlib import Path

import osgeo  # noqa f401
import numpy as np
import rasterio
import xarray as xr
from rasterio.profiles import default_gtiff_profile

# specs for raster outputs. These are the default specs for which all input and output
# raster files should match.
res01 = 0.1  # deg
lon_left = -130  # deg
lon_right = -60  # deg
lat_up = 55  # deg
lat_low = 20  # deg
x = np.arange(lon_left, lon_right, res01)
y = np.arange(lat_low, lat_up, res01)
HEIGHT, WIDTH = ARR_SHAPE = (len(y), len(x))

GEPA_PROFILE = default_gtiff_profile.copy()
GEPA_PROFILE.update(
    transform=rasterio.Affine(res01, 0.0, lon_left, 0.0, -res01, lat_up),
    height=HEIGHT,
    width=WIDTH,
    crs=4326,
    dtype=np.float64,
)

# take any input raster file and warp it to match the GEPA_PROFILE
def warp_to_gepa_grid():
    pass


# take any vector data input and grid it to the standard GEPA grid
def vector_to_gepa_grid():
    pass


# function to create an empty x/array of the standard GEPA grid
def create_empty_grid():
    pass
