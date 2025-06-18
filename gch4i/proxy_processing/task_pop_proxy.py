"""
Name:                  task_pop_proxy.py
Date Last Modified:    2025-01-27
Authors Name:          Nick Kruskamp (RTI International)
Purpose:               Generate population proxy data for emissions.
Input Files:           - DL URL: https://data.worldpop.org/GIS/Population/
                        Global_2000_2020_1km/  f"{year}/USA/
                        usa_ppp_{year}_1km_Aggregated.tif"
Output Files:          - DST Path: {tmp_data_dir_path}/usa_ppp_{year}_1km_Aggregated.tif
"""

# %% Import Libraries
%load_ext autoreload
%autoreload 2

# %%
import multiprocessing
from pathlib import Path
from typing import Annotated

from pytask import Product, mark, task
import pytask

from gch4i.config import (
    global_data_dir_path,
    proxy_data_dir_path,
    sector_data_dir_path,
    years,
)
from gch4i.utils import download_url, proxy_from_stack, stack_rasters, warp_to_gepa_grid

NUM_WORKERS = multiprocessing.cpu_count()

population_dir = sector_data_dir_path / "worldpop"
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
        if year > 2020:
            continue
        dl_url = (
            "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/"
            f"{year}/USA/usa_ppp_{year}_1km_Aggregated.tif"
        )
        dst_path = population_dir / f"usa_ppp_{year}_1km_Aggregated.tif"
        _id_to_kwargs[str(year)] = {"url": dl_url, "output_path": dst_path}
    return _id_to_kwargs


# Store the download parameters in a dictionary.
_ID_TO_KWRARGS_DL = get_download_params(years)


def task_download_world_pop(url: str, output_path: Annotated[Path, Product]) -> None:
    download_url(url, output_path)


# Build the download tasks using pytask.
pytask.build(
    tasks=[task_download_world_pop(**kwargs) for kwargs in _ID_TO_KWRARGS_DL.values()],
    verbose=1,
    marker_expression="persist",
)


# %%
def get_warp_params(years):
    _id_to_kwargs = {}
    for year in years:
        # worldpop only has data up to 2020.
        if year > 2020:
            continue
        input_path = population_dir / f"usa_ppp_{year}_1km_Aggregated.tif"
        output_path = population_dir / f"usa_ppp_{year}_reprojected.tif"

        _id_to_kwargs[str(year)] = {
            "input_path": input_path,
            "output_path": output_path,
        }
    return _id_to_kwargs


_ID_TO_KWARGS_WARP = get_warp_params(years)


def task_warp_world_pop(input_path: Path, output_path: Annotated[Path, Product]):
    warp_to_gepa_grid(
        input_path=input_path,
        output_path=output_path,
        resampling="sum",
        num_threads=NUM_WORKERS,
    )


pytask.build(
    tasks=[task_warp_world_pop(**kwargs) for kwargs in _ID_TO_KWARGS_WARP.values()],
    verbose=1,
)
# %%


def get_stack_params(years):
    _id_to_kwargs = {}
    input_paths = []
    for year in years:
        # NOTE: 2021 and 2022 are not available, so we use 2020 instead. Noted in the
        # smartsheet row.
        if year > 2020:
            year = 2020
        input_path = population_dir / f"usa_ppp_{year}_reprojected.tif"
        input_paths.append(input_path)
    output_path = population_dir / "population_proxy_raw.tif"
    _id_to_kwargs["population_proxy"] = {
        "input_paths": input_paths,
        "output_path": output_path,
    }
    return _id_to_kwargs


_ID_TO_KWARGS_STACK = get_stack_params(years)


def task_stack_population_data(
    input_paths: Path, output_path: Annotated[Path, Product]
):
    stack_rasters(input_paths, output_path)


pytask.build(
    tasks=[
        task_stack_population_data(**kwargs) for kwargs in _ID_TO_KWARGS_STACK.values()
    ],
    verbose=10,
)


# %%
def task_population_proxy(
    input_path: Path = population_dir / "population_proxy_raw.tif",
    state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "population_proxy.nc"
    ),
):
    proxy_from_stack(input_path, state_geo_path, output_path)


pytask.build(
    tasks=[task_population_proxy],
    verbose=1,
)

# %%
