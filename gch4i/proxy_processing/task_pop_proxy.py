"""
Name:                  task_pop_proxy.py
Date Last Modified:    2025-01-27
Authors Name:          Nick Kruskamp (RTI International)
Purpose:               Generate population proxy data for emissions.
Input Files:           - DL URL: "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/"
                        f"{year}/USA/usa_ppp_{year}_1km_Aggregated.tif"
                        - DL URL: "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/"
                        f"{year}/USA/v1/1km_ua/constrained/usa_pop_{year}_CN_1km_R2025A_UA_v1.tif"
Output Files:           - DST Path: population_dir / f"usa_pop_{year}_1km_worldpop.tif"
                        - DST Path: population_dir / f"usa_pop_{year}_reprojected.tif"
                        - DST Path: population_dir / "population_proxy_raw.tif"
                        - DST Path: v4_proxy_data_dir_path / "population_proxy.nc"
Description:            This script downloads population data from WorldPop,
                        warps it to the GEPA grid, stacks the rasters, and
                        generates a population proxy for emissions. The tasks are
                        defined using the pytask framework.
Change log:             - v3 used the Global 1 version of the worldpop data. New release
                        of their data now includes estimates and projections up to 2030,
                        so we switch to that. These Global 2015-2030 data also use a
                        different file naming convention. We continue to use Global 1
                        for years prior to 2015.

"""

# %%
import multiprocessing
from pathlib import Path
from typing import Annotated

from pytask import Product, mark, task
import pytask

from gch4i.config import (
    v4_global_data_dir_path,
    v4_proxy_data_dir_path,
    v4_sector_data_dir_path,
    years,
)
from gch4i.utils import download_url, proxy_from_stack, stack_rasters, warp_to_gepa_grid

NUM_WORKERS = multiprocessing.cpu_count()

population_dir = v4_sector_data_dir_path / "worldpop"


# %% Functions
def get_download_params(years):
    """
    Obtain the download parameters for the population data.

    Args:
        dl_url (str): The download URL for the population data.

    Returns:
        _id_to_kwargs (dict): A dictionary of the download parameters.
    """
    _id_to_kwargs = {}
    for year in years:
        # worldpop only has data up to 2020.
        if year < 2015:
            dl_url = (
                "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/"
                f"{year}/USA/usa_ppp_{year}_1km_Aggregated.tif"
            )
        else:
            dl_url = (
                "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/"
                f"{year}/USA/v1/1km_ua/constrained/"
                f"usa_pop_{year}_CN_1km_R2025A_UA_v1.tif"
            )
        dst_path = population_dir / f"usa_pop_{year}_1km_worldpop.tif"
        _id_to_kwargs[str(year)] = {"url": dl_url, "output_path": dst_path}
    return _id_to_kwargs


def get_warp_params(years):
    _id_to_kwargs = {}
    for year in years:
        input_path = population_dir / f"usa_pop_{year}_1km_worldpop.tif"
        output_path = population_dir / f"usa_pop_{year}_reprojected.tif"

        _id_to_kwargs[str(year)] = {
            "input_path": input_path,
            "output_path": output_path,
        }
    return _id_to_kwargs


def get_stack_params(years):
    _id_to_kwargs = {}
    input_paths = []
    for year in years:
        input_path = population_dir / f"usa_pop_{year}_reprojected.tif"
        input_paths.append(input_path)
    output_path = population_dir / "population_proxy_raw.tif"
    _id_to_kwargs["population_proxy"] = {
        "input_paths": input_paths,
        "output_path": output_path,
    }
    return _id_to_kwargs


# Store the download parameters in a dictionary.
_ID_TO_KWRARGS_DL = get_download_params(years)
_ID_TO_KWARGS_WARP = get_warp_params(years)
_ID_TO_KWARGS_STACK = get_stack_params(years)


tasks = []

# STEP 1: download the population data.
for _id, kwargs in _ID_TO_KWRARGS_DL.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_download_world_pop(
        url: str, output_path: Annotated[Path, Product]
    ) -> None:
        download_url(url, output_path)

    tasks.append(task_download_world_pop)


# STEP 2: warp the population data to the GEPA grid.
for _id, kwargs in _ID_TO_KWARGS_WARP.items():
    # @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_warp_world_pop(
        input_path: Path, output_path: Annotated[Path, Product]
    ) -> None:
        warp_to_gepa_grid(
            input_path=input_path,
            output_path=output_path,
            resampling="sum",
            num_threads=NUM_WORKERS,
        )

    tasks.append(task_warp_world_pop)


# STEP 3: stack the population rasters into a single multi-band raster.
for _id, kwargs in _ID_TO_KWARGS_STACK.items():
    # @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_stack_population_data(
        input_paths: Path, output_path: Annotated[Path, Product]
    ) -> None:
        stack_rasters(input_paths, output_path)

    tasks.append(task_stack_population_data)


# STEP 4: generate the population proxy from the stacked raster.
# @mark.persist
@task(id="population_proxy")
def task_population_proxy(
    input_path: Path = population_dir / "population_proxy_raw.tif",
    state_geo_path: Path = v4_global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = (
        v4_proxy_data_dir_path / "population_proxy.nc"
    ),
) -> None:
    proxy_from_stack(input_path, state_geo_path, output_path)


tasks.append(task_population_proxy)

# %%
[task_download_world_pop(**kwargs) for _id, kwargs in _ID_TO_KWRARGS_DL.items()]
[task_warp_world_pop(**kwargs) for _id, kwargs in _ID_TO_KWARGS_WARP.items()]
[task_stack_population_data(**kwargs) for _id, kwargs in _ID_TO_KWARGS_STACK.items()]
task_population_proxy()

# %%
from gch4i.gridding_utils import EmiProxyGridder, GriddingInfo

grid_info = GriddingInfo()

for row in grid_info.mapping_df.query(f"proxy_id == 'population_proxy'").itertuples():
    try:
        gridder = EmiProxyGridder(
            gch4i_name=row.gch4i_name,
            proxy_id=row.proxy_id,
            emi_id=row.emi_id,
        )
        gridder.run_gridding()
    except Exception as e:
        print(f"Error for {row.gch4i_name}, {row.proxy_id}, {row.emi_id}: {e}")

grid_info.get_status_table()
grid_info.display_all_pair_statuses()
# %%