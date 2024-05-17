from pathlib import Path
from typing import Annotated

import osgeo
import geopandas as gpd
import numpy as np
import rasterio
from pytask import Product, mark, task
from rasterio.features import shapes

from gch4i.config import global_data_dir_path
from gch4i.gridding import ARR_SHAPE, GEPA_PROFILE


# @mark.persist
@task
def task_create_area_raster(
    output_path: Annotated[Path, Product] = global_data_dir_path / "gridded_area_m2.tif"
):
    """create a raster where each cell value is its' area in square meters

    This uses a equal area projection to calculate the cell areas.
    use ESRI:102003 USA_Contiguous_Albers_Equal_Area_Conic for accurate area
    measurements
    https://gis.stackexchange.com/questions/141580/which-projection-is-best-for-mapping-the-contiguous-united-states
    """

    # get the number of cells
    ncells = np.multiply(*ARR_SHAPE)
    # create an empty array of the right shape, assign each cell a unique value
    tmp_arr = np.arange(ncells, dtype=np.int32).reshape(ARR_SHAPE)
    # get the cells as individual items in a dictionary holding their id and geom
    results = [
        {"properties": {"raster_val": v}, "geometry": s}
        for i, (s, v) in enumerate(shapes(tmp_arr, transform=GEPA_PROFILE["transform"]))
    ]

    # turn geom dictionary into a geodataframe, reproject to equal area, calculate the
    # cell area in square meters
    area_gdf = (
        gpd.GeoDataFrame.from_features(results, crs=4326)
        .to_crs("ESRI:102003")
        .assign(
            cell_area_sq_m=lambda df: df.area,
            # cell_area_sq_mi=lambda df: df["cell_area_sq_m"] / 2.59e6,
        )
    )

    # We have to resort the dataframe on the id value to get it in the right order for
    # turning into a matrix
    area_matrix = area_gdf.sort_values("raster_val", ascending=False)[
        "cell_area_sq_m"
    ].values.reshape(ARR_SHAPE)

    # get the GEPA profile, make the count 1
    dst_profile = GEPA_PROFILE.copy()
    dst_profile.update(count=1)

    # save the file for all other tasks to use
    with rasterio.open(output_path, "w", **dst_profile) as dst:
        dst.write(area_matrix, 1)
