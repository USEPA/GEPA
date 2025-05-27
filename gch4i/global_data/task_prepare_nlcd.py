"""
Name:                   task_prepare_nlcd.py
Date Last Modified:     2025-05-27
Authors Name:           Nick Kruskamp (RTI International)
Purpose:                This script is used to process the NLCD land cover data into
                        forest and grass binary layers aligned  the GEPA grid. The NLCD
                        dat is manually downloaded from the MRLC website and placed in
                        the `gch4i/sector_data/nlcd` directory. The script creates binary
                        rasters for forest and grassland, warps them to the GEPA grid, and
                        vectorizes the rasters into GeoDataFrames. The resulting data is saved
                        as GeoDataFrames in Parquet format for further analysis.
"""

# %%
# %load_ext autoreload
# %autoreload 2
# %%

import multiprocessing
from pathlib import Path

import rasterio
from rasterio import features
from shapely.geometry import shape
import geopandas as gpd

from gch4i.config import sector_data_dir_path
from gch4i.utils import make_raster_binary, warp_to_gepa_grid

NUM_WORKERS = multiprocessing.cpu_count()
# %%

nlcd_dir = sector_data_dir_path / "nlcd"
nlcd_2012_data_path = nlcd_dir / "Annual_NLCD_LndCov_2012_CU_C1V0.tif"

# https://www.mrlc.gov/sites/default/files/docs/LSDS-2103%20Annual%20National%20Land%20Cover%20Database%20(NLCD)%20Collection%201%20Science%20Product%20User%20Guide%20-v1.0%202024_10_15.pdf
forest_vals = [41, 42, 43]

# NOTE: shrubland (52) may also be a valid classification for grassland?
grass_vals = [52, 71]


class BinaryAndWarpRaster:
    """
    A class to process NLCD land cover data into binary rasters for forest and grassland
    """

    def __init__(
        self,
        input_path: Path,
        binary_output_path: Path,
        gepa_output_path: Path,
        true_vals: list[int],
    ):
        self.input_path = input_path
        self.binary_output_path = binary_output_path
        self.gepa_output_path = gepa_output_path
        self.true_vals = true_vals
        self.process()

    def process(self):
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file {self.input_path} not found.")
        if not self.binary_output_path.exists():
            make_raster_binary(
                input_path=self.input_path,
                output_path=self.binary_output_path,
                true_vals=self.true_vals,
                num_workers=NUM_WORKERS,
            )
        if not self.gepa_output_path.exists():
            warp_to_gepa_grid(
                input_path=self.binary_output_path,
                output_path=self.gepa_output_path,
                resampling="nearest",
                num_threads=NUM_WORKERS,
            )

    def vectorize_raster(self, plot=False, save=True):
        with rasterio.open(self.gepa_output_path) as src:
            data = src.read(1)  # Read the first band
            mask = data == 1  # Assuming binary raster where 1 is the true value
            transform = src.transform
            crs = src.crs
        raster_shapes = features.shapes(data, mask=mask, transform=transform)
        shapes, vals = zip(*raster_shapes)
        shapes = [shape(s) for s in shapes]
        vals = [int(x) for x in vals]  # Convert values to integers

        self.feature_gdf = gpd.GeoDataFrame(
            {"geometry": shapes, "values": vals}, crs=crs
        )
        if plot:
            self.feature_gdf.plot("values")
        if save:
            self.feature_gdf.to_parquet(self.gepa_output_path.with_suffix(".parquet"))


# %%
forest_processor = BinaryAndWarpRaster(
    input_path=nlcd_2012_data_path,
    binary_output_path=nlcd_dir / "NLCD_2012_forest_binary.tif",
    gepa_output_path=nlcd_dir / "NLCD_2012_forest_binary_gepa.tif",
    true_vals=forest_vals,
)
forest_processor.vectorize_raster(plot=True, save=True)

grass_processor = BinaryAndWarpRaster(
    input_path=nlcd_2012_data_path,
    binary_output_path=nlcd_dir / "NLCD_2012_grass_binary.tif",
    gepa_output_path=nlcd_dir / "NLCD_2012_grass_binary_gepa.tif",
    true_vals=grass_vals,
)
grass_processor.vectorize_raster(plot=True, save=True)

# %%
