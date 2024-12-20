# %%
import multiprocessing
from pathlib import Path
from typing import Annotated

from pytask import Product, mark, task

from gch4i.config import (
    global_data_dir_path,
    proxy_data_dir_path,
    tmp_data_dir_path,
    years,
)
from gch4i.utils import download_url, proxy_from_stack, stack_rasters, warp_to_gepa_grid

NUM_WORKERS = multiprocessing.cpu_count()

# %%


def get_download_params(years):
    _id_to_kwargs = {}
    for year in years:
        dl_url = (
            "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/"
            f"{year}/USA/usa_ppp_{year}_1km_Aggregated.tif"
        )
        dst_path = tmp_data_dir_path / f"usa_ppp_{year}_1km_Aggregated.tif"
        _id_to_kwargs[str(year)] = {"url": dl_url, "output_path": dst_path}
    return _id_to_kwargs


_ID_TO_KWRARGS_DL = get_download_params(years)
_ID_TO_KWRARGS_DL


for _id, kwargs in _ID_TO_KWRARGS_DL.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_download_world_pop(
        url: str, output_path: Annotated[Path, Product]
    ) -> None:
        download_url(url, output_path)


def get_warp_params(years):
    _id_to_kwargs = {}
    for year in years:
        input_path = tmp_data_dir_path / f"usa_ppp_{year}_1km_Aggregated.tif"
        output_path = tmp_data_dir_path / f"usa_ppp_{year}_reprojected.tif"

        _id_to_kwargs[str(year)] = {
            "input_path": input_path,
            "output_path": output_path,
        }
    return _id_to_kwargs


_ID_TO_KWRARGS_WARP = get_warp_params(years)
_ID_TO_KWRARGS_WARP

for _id, kwargs in _ID_TO_KWRARGS_WARP.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_warp_world_pop(input_path: Path, output_path: Annotated[Path, Product]):

        warp_to_gepa_grid(
            input_path, output_path, resampling="sum", num_threads=NUM_WORKERS
        )


def get_stack_params(years):
    _id_to_kwargs = {}
    input_paths = []
    for year in years:
        # NOTE: 2021 and 2022 are not available, so we use 2020 instead. Noted in the
        # smartsheet row.
        if year in [2021, 2022]:
            year = 2020
        input_path = tmp_data_dir_path / f"usa_ppp_{year}_reprojected.tif"
        input_paths.append(input_path)
    output_path = tmp_data_dir_path / "population_proxy_raw.tif"
    _id_to_kwargs["population_proxy"] = {
        "input_paths": input_paths,
        "output_path": output_path,
    }
    return _id_to_kwargs


_ID_TO_KWARGS_STACK = get_stack_params(years)
_ID_TO_KWARGS_STACK

for _id, kwargs in _ID_TO_KWARGS_STACK.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_stack_population_data(
        input_paths: Path, output_path: Annotated[Path, Product]
    ):
        stack_rasters(input_paths, output_path)


@mark.persist
@task
def task_population_proxy(
    input_path: Path = tmp_data_dir_path / "population_proxy_raw.tif",
    state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "population_proxy.nc"
    ),
):
    proxy_from_stack(input_path, state_geo_path, output_path)
