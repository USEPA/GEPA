"""
Author:     Nick Kruskamp
Date:       2021-07-26
exsum:      This script is used to process the flooded lands data into the proxy data.
"""

# %%
# %load_ext autoreload
# %autoreload 2


from pathlib import Path
from typing import Annotated

import geopandas as gpd
import rasterio
from pytask import Product, mark, task

from gch4i.config import (
    EQ_AREA_CRS,
    global_data_dir_path,
    proxy_data_dir_path,
    sector_data_dir_path,
    years,
)
from gch4i.utils import (
    GEPA_spatial_profile,
    get_cell_gdf,
    proxy_from_stack,
    stack_rasters,
)

# %%

fl_data_path = sector_data_dir_path / "flooded_lands"
intermediate_dir = fl_data_path / "intermediate"
intermediate_dir.mkdir(exist_ok=True, parents=True)
# list(fl_data_path.rglob("*"))

fl_gpkg_path = fl_data_path / "flooded_lands_.gpkg"

query_dict = dict(
    fl_rem_res="lu == 'Flooded Land Remaining Flooded Land' & type == 'reservoir'",
    fl_rem_other="lu == 'Flooded Land Remaining Flooded Land' & type == 'other constructed waterbodies'",  # noqa
    fl_conv_res="lu == 'Land Converted to Flooded Land' & type == 'reservoir'",
)

# %%
EMTPY_GRID_GDF = get_cell_gdf().to_crs(EQ_AREA_CRS)
EMTPY_GRID_GDF


# %%


def get_data_params(years, q_dict):
    arg_dict = dict()

    for the_year in years:

        # since we don't have data for 2021 and 2022, we'll use the 2020 data later in
        # the process
        if the_year > 2020:
            continue

        arg_dict[the_year] = dict()
        arg_dict[the_year]["the_year"] = the_year
        arg_dict[the_year]["input_path"] = fl_gpkg_path
        arg_dict[the_year]["query_dict"] = q_dict

        output_paths = {
            x: intermediate_dir / f"{x}_{the_year}.tif" for x in q_dict.keys()
        }

        arg_dict[the_year]["output_path_dict"] = output_paths
    return arg_dict


_ID_TO_KWARGS_FL_DATA = get_data_params(years, query_dict)

for _id, kwargs in _ID_TO_KWARGS_FL_DATA.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_process_fl_data(
        the_year: int,
        input_path: Path,
        query_dict: dict,
        output_path_dict: Annotated[dict[str, Path], Product],
    ) -> None:

        profile = GEPA_spatial_profile()
        profile.profile.update(count=1)

        year_gdf = gpd.read_file(input_path, layer=str(the_year))

        # for each of the flooded land types we need
        for key, query in query_dict.items():

            # create the output path
            out_path = output_path_dict[key]
            # pytask will run this even if the data exist for just one product, so we're
            # going to skip if the file exists here to avoid unnecessary processing.
            if out_path.exists():
                continue
            # filter the data from that year with with query, calculate the area of the
            # flooded land polygons
            print(f"filtering for {key} in {the_year}")
            data_gdf = (
                year_gdf.query(query)
                .to_crs(EQ_AREA_CRS)
                .assign(fl_area=lambda df: df.area)
            )
            print(the_year, key, data_gdf.shape[0])
            # data_gdf.to_parquet(out_path)

            # the bulk of the work: overlay the data with the grid cells, calculated the
            # fractional area of that cell that is flooded, and multiply that by the
            # total ch4 emissions for that cell to get the fraction of emission. This
            # accounts for polygons that span multiple cells so that the emissions are
            # allocated via the fractional area. Then groupby the cell id and sum the
            # emissions.
            print(f"calculating grid cell emissions for {key} in {the_year}")
            ch4_sum = (
                data_gdf.overlay(
                    EMTPY_GRID_GDF.reset_index(drop=False), how="intersection"
                )
                .assign(
                    fractional_area=lambda df: df.area / df["fl_area"],
                    frac_emi=lambda df: df["fractional_area"]
                    * df["ch4.total.tonnes.y"],
                )
                .groupby("index")["frac_emi"]
                .sum()
                .rename("ch4_sum")
            )
            # join this data back to the empty grid, reshape the data to the original
            # grid.
            res_gdf = EMTPY_GRID_GDF.join(ch4_sum)

            print(f"saving data for {key} in {the_year}")
            out_arr = res_gdf.ch4_sum.values.reshape(profile.arr_shape)
            # save the file
            with rasterio.open(out_path, "w", **profile.profile) as dst:
                dst.write(out_arr, 1)
            print()


# %%
def get_stack_params(years, q_dict):
    arg_dict = {}
    for key in q_dict.keys():
        output_path = intermediate_dir / f"{key}_stack.tif"
        input_paths = []
        for the_year in years:
            # if the the year is 2021 or 2022, use 2020 data
            if the_year > 2020:
                input_paths.append(intermediate_dir / f"{key}_2020.tif")
            else:
                input_paths.append(intermediate_dir / f"{key}_{the_year}.tif")
        arg_dict[f"{key}_stack"] = {
            "input_paths": input_paths,
            "output_path": output_path,
        }
    return arg_dict


_ID_TO_KWARGS_STACK = get_stack_params(years, query_dict)

for _id, kwargs in _ID_TO_KWARGS_STACK.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_stack_fl_data(input_paths: Path, output_path: Annotated[Path, Product]):
        stack_rasters(input_paths, output_path)


# %%


def get_proxy_params(q_dict):
    arg_dict = dict()
    for key in q_dict.keys():
        arg_dict[key] = {
            "input_path": intermediate_dir / f"{key}_stack.tif",
            "state_geo_path": global_data_dir_path / "tl_2020_us_state.zip",
            "output_path": proxy_data_dir_path / f"{key}_proxy.nc",
        }
    return arg_dict


_ID_PROXY_PARAMS = get_proxy_params(query_dict)

for _id, kwargs in _ID_PROXY_PARAMS.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_fl_proxy(
        input_path: Path,
        state_geo_path: Path,
        output_path: Annotated[Path, Product],
    ) -> None:
        proxy_from_stack(input_path, state_geo_path, output_path)


# %%
