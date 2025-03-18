# %%
from pathlib import Path
from typing import Annotated

import osgeo  # noqa f401
import geopandas as gpd
import numpy as np
import rasterio
from pytask import Product, mark, task
from rasterio.features import shapes

from gch4i.config import global_data_dir_path
from gch4i.utils import GEPA_spatial_profile


def _create_params(resolution_list):
    _id_to_kwargs = {}
    for resolution in resolution_list:
        res_text = str(resolution).replace(".", "")
        _id_to_kwargs[res_text] = {
            "resolution": resolution,
            "output_path": global_data_dir_path / f"gridded_area_{res_text}_cm2.tif",
        }
    return _id_to_kwargs


_ID_TO_KWARGS = _create_params([0.1, 0.01])



resolution, output_path = _ID_TO_KWARGS["001"].values()
# resolution, output_path = _ID_TO_KWARGS["01"].values()
# %%
for _id, kwargs in _ID_TO_KWARGS.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_create_area_raster(
        resolution: float, output_path: Annotated[Path, Product]
    ):
        """create a raster where each cell value is its' area in square meters

        This uses a equal area projection to calculate the cell areas.
        use ESRI:102003 USA_Contiguous_Albers_Equal_Area_Conic for accurate area
        measurements
        https://gis.stackexchange.com/questions/141580/which-projection-is-best-for-mapping-the-contiguous-united-states
        """

        profile = GEPA_spatial_profile(resolution)
        # get the GEPA profile, make the count 1
        profile.profile.update(count=1)

        # get the number of cells
        ncells = np.multiply(*profile.arr_shape)
        # create an empty array of the right shape, assign each cell a unique value
        tmp_arr = np.arange(ncells, dtype=np.int32).reshape(profile.arr_shape)
        # get the cells as individual items in a dictionary holding their id and geom
        # NOTE: previously we calculated the area for the entire raster, but we only
        # need calculate the area for the first column (N/S), as the area is the same
        # for all rows (E/W). So we calculate only the first column and replicate those
        # values across the rest of the raster. This significantly speeds up calcs.
        results = [
            {"properties": {"raster_val": v}, "geometry": s}
            for i, (s, v) in enumerate(
                shapes(tmp_arr[:, [0]], transform=profile.profile["transform"])
                # shapes(tmp_arr, transform=profile.profile["transform"])
            )
        ]

        # turn geom dictionary into a geodataframe, reproject to equal area, calculate
        # the cell area in square meters
        area_gdf = (
            gpd.GeoDataFrame.from_features(results, crs=profile.profile["crs"])
            .to_crs("ESRI:102003")
            .assign(
                cell_area_sq_cm=lambda df: df.area * 10_000,
            )
        )

        # replicate the first row across the width of the array/raster.
        area_matrix = np.repeat(
            area_gdf.cell_area_sq_cm.values, repeats=profile.arr_shape[1]
        ).reshape(profile.arr_shape)

        # save the file for all other tasks to use
        with rasterio.open(output_path, "w", **profile.profile) as dst:
            dst.write(area_matrix, 1)


# %%
