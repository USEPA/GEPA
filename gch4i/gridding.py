# a collection of functions that do standard gridding
from pathlib import Path

import osgeo  # noqa f401
import numpy as np
import rasterio
import xarray as xr
from rasterio.profiles import default_gtiff_profile


# # specs for raster outputs. These are the default specs for which all input and output
# # raster files should match.
# res01 = 0.1  # deg
# lon_left = -130  # deg
# lon_right = -60  # deg
# lat_up = 55  # deg
# lat_low = 20  # deg
# x = np.arange(lon_left, lon_right, res01)
# y = np.arange(lat_low, lat_up, res01)
# height, width = arr_shape = (len(y), len(x))

# GEPA_PROFILE = default_gtiff_profile.copy()
# GEPA_PROFILE.update(
#     transform=rasterio.Affine(res01, 0.0, lon_left, 0.0, -res01, lat_up),
#     height=height,
#     width=width,
#     crs=4326,
#     dtype=np.float64,
# )


# %%
class GEPA_spatial_profile:
    lon_left = -130  # deg
    lon_right = -60  # deg
    lat_up = 55  # deg
    lat_low = 20  # deg
    valid_resolutions = [0.1, 0.01]

    def __init__(self, resolution:float = 0.1):
        self.resolution = resolution
        self.check_resolution(self.resolution)
        self.x = np.arange(self.lon_left, self.lon_right, self.resolution)
        self.y = np.arange(self.lat_low, self.lat_up, self.resolution)
        self.height, self.width = self.arr_shape = (len(self.y), len(self.x))
        self.profile = self.get_profile(self.resolution)

    def check_resolution(self, resolution):
        if resolution not in self.valid_resolutions:
            raise ValueError(
                f"resolution must be one of {', '.join(self.valid_resolutions)}"
            )

    def get_profile(self, resolution):
        base_profile = default_gtiff_profile.copy()
        base_profile.update(
            transform=rasterio.Affine(
                resolution, 0.0, self.lon_left, 0.0, -resolution, self.lat_up
            ),
            height=self.height,
            width=self.width,
            crs=4326,
            dtype=np.float64,
        )
        return base_profile


# %%


# take any input raster file and warp it to match the GEPA_PROFILE
def warp_to_gepa_grid():
    pass


# take any vector data input and grid it to the standard GEPA grid
def vector_to_gepa_grid():
    pass


# function to create an empty x/array of the standard GEPA grid
def create_empty_grid():
    pass
