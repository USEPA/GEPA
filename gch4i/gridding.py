# a collection of functions that do standard gridding
import concurrent
import threading
from pathlib import Path

import numpy as np
import osgeo  # noqa f401
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.profiles import default_gtiff_profile


# import xarray as xr

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

    def __init__(self, resolution: float = 0.1):
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
            dtype="float32",
        )
        return base_profile


# %%
def make_raster_binary(
    input_path: Path, output_path: Path, true_vals: np.array, num_workers: int = 1
):
    """take a raster file and convert it to a binary raster file based on the true_vals
    array"""
    with rasterio.open(input_path) as src:

        # Create a destination dataset based on source params. The
        # destination will be tiled, and we'll process the tiles
        # concurrently.
        profile = src.profile
        profile.update(nodata=-99, dtype="int8")

        with rasterio.open(output_path, "w", **profile) as dst:
            windows = [window for ij, window in dst.block_windows()]

            # We cannot write to the same file from multiple threads
            # without causing race conditions. To safely read/write
            # from multiple threads, we use a lock to protect the
            # DatasetReader/Writer
            read_lock = threading.Lock()
            write_lock = threading.Lock()

            def process(window):
                with read_lock:
                    src_array = src.read(window=window)

                # result = np.where((src_array >= 1) & (src_array <= 60), 1, 0)
                result = np.isin(src_array, true_vals)

                with write_lock:
                    dst.write(result, window=window)

            # We map the process() function over the list of
            # windows.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                executor.map(process, windows)


def mask_raster_parallel(
    input_path: Path, output_path: Path, mask_path: Path, num_workers: int = 1
):
    """take a raster file and convert it to a binary raster file based on the true_vals
    array

    alpha version of the function. Not fully tested yet. Use with caution.

    This does not check if the mask raster is the same size as the input raster. It
    assumes that the mask raster is the same size as the input raster. If not, you will
    need to run the warp function with the input_path as the target_path to get the mask
    aligned correctly. Otherwise this will throw and error.
    """
    with rasterio.open(input_path) as src:
        with rasterio.open(mask_path) as msk:

            # Create a destination dataset based on source params. The
            # destination will be tiled, and we'll process the tiles
            # concurrently.
            # profile = src.profile
            # profile.update(
            #     blockxsize=128, blockysize=128, tiled=True, nodata=0, dtype="float32"
            # )

            with rasterio.open(output_path, "w", **src.profile) as dst:
                windows = [window for ij, window in dst.block_windows()]

                # We cannot write to the same file from multiple threads
                # without causing race conditions. To safely read/write
                # from multiple threads, we use a lock to protect the
                # DatasetReader/Writer
                read_lock = threading.Lock()
                write_lock = threading.Lock()

                def process(window):
                    with read_lock:
                        src_array = src.read(window=window)
                        msk_array = msk.read(window=window)

                    # result = np.where((src_array >= 1) & (src_array <= 60), 1, 0)
                    result = np.where(msk_array == 1, src_array, 0)

                    with write_lock:
                        dst.write(result, window=window)

                # We map the process() function over the list of
                # windows.
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                ) as executor:
                    executor.map(process, windows)


# take any input raster file and warp it to match the GEPA_PROFILE
def warp_to_gepa_grid(
    input_path: Path,
    output_path: Path,
    target_path: Path = None,
    resampling: str = "average",
    num_threads: int = 1,
):

    if target_path is None:
        # print("warping to GEPA grid")
        profile = GEPA_spatial_profile().profile
    else:
        # print("warping to other raster")
        with rasterio.open(target_path) as src:
            profile = src.profile

    try:
        resamp_method = getattr(Resampling, resampling)
    except AttributeError as ex:
        raise ValueError(
            f"resampling method {resampling} not found in rasterio.Resampling"
        ) from ex

    with rasterio.open(input_path) as src:
        profile.update(count=src.count, dtype="float32")
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=profile["transform"],
                    dst_crs=profile["crs"],
                    dst_nodata=0.0,
                    resampling=resamp_method,
                    num_threads=num_threads,
                )


# take any vector data input and grid it to the standard GEPA grid
def vector_to_gepa_grid():
    pass


# function to create an empty x/array of the standard GEPA grid
def create_empty_grid():
    pass


# %%
