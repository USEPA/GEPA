# %%
from pathlib import Path
from typing import Annotated

from pyarrow import parquet  # noqa
import osgeo  # noqa
import geopandas as gpd
import numpy as np
import rasterio
import requests
import rioxarray
from geocube.api.core import make_geocube
from pytask import Product, mark, task
from rasterio import shutil as rio_shutil
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from gch4i.config import (
    global_data_dir_path,
    proxy_data_dir_path,
    tmp_data_dir_path,
    years,
)
from gch4i.gridding import GEPA_spatial_profile

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
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("File downloaded successfully!")
        except requests.exceptions.RequestException as e:
            print("Error downloading the file:", e)


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

        # NOTE: We use sum resampling because the data is population data, so we want to
        # add up all the cells in the warping.
        profile = GEPA_spatial_profile(0.1)
        vrt_options = {
            "resampling": Resampling.sum,
            "crs": profile.profile["crs"],
            "transform": profile.profile["transform"],
            "height": profile.height,
            "width": profile.width,
        }

        with rasterio.open(input_path) as src:
            with WarpedVRT(src, **vrt_options) as vrt:
                # Read all data into memory.
                _ = vrt.read()

                # write the file to disk
                rio_shutil.copy(vrt, output_path, driver="GTiff")


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

    # @mark.persist
    # @task(id=_id, kwargs=kwargs)
    def task_stack_population_data(
        input_paths: Path, output_path: Annotated[Path, Product]
    ):

        raster_list = []
        years = [int(x.name.split("_")[2]) for x in input_paths]
        for in_file in input_paths:
            if not in_file.exists():
                continue
            with rasterio.open(in_file) as src:
                raster_list.append(src.read(1))

        output_data = np.stack(raster_list, axis=0)

        profile = GEPA_spatial_profile(0.1)
        profile.profile.update(count=len(raster_list), nodata=-99999)

        with rasterio.open(output_path, "w", **profile.profile) as dst:
            dst.write(output_data)
            dst.descriptions = tuple([str(x) for x in years])


@mark.persist
@task
def task_population_proxy(
    input_path: Path = tmp_data_dir_path / "population_proxy_raw.tif",
    state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "population_proxy.nc"
    ),
):

    # read in the state file and filter to lower 48 + DC
    state_gdf = (
        gpd.read_file(state_geo_path)
        .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code", "name": "state_name"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .to_crs(4326)
    )

    # read in the raw population data raster stack as a xarray dataset
    # masked will read the nodata value and set it to NA
    pop_ds = rioxarray.open_rasterio(input_path, masked=True).rename({"band": "year"})

    with rasterio.open(input_path) as src:
        ras_crs = src.crs

    # assign the band as our years so the output raster data has year band names
    pop_ds["year"] = years
    # remove NA values
    # pop_ds = pop_ds.where(pop_ds != -99999)

    # create a state grid to match the input population array
    # we use fill here to fill in the nodata values with 99 so that when we do the
    # groupby, the nodata area is not collapsed, and the resulting dimensions align
    # with the v3 gridded data.
    state_grid = make_geocube(
        vector_data=state_gdf, measurements=["statefp"], like=pop_ds, fill=99
    )
    state_grid
    # assign the state grid as a new variable in the population dataset
    pop_ds["statefp"] = state_grid["statefp"]
    pop_ds

    # plot the data to check
    pop_ds["statefp"].plot()

    # define a function to normalize the population data by state and year
    def normalize(x):
        return x / x.sum()

    # apply the normalization function to the population data
    out_ds = (
        pop_ds.groupby(["year", "statefp"])
        .apply(normalize)
        .sortby(["year", "y", "x"])
        .to_dataset(name="population")
    )
    out_ds["population"].shape

    # check that the normalization worked
    all_eq_df = (
        out_ds["population"]
        .groupby(["statefp", "year"])
        .sum()
        .rename("sum_check")
        .to_dataframe()
        .drop(columns="spatial_ref")
    )

    # NOTE: Due to floating point rouding, we need to check if the sum is close to 1,
    # not exactly 1.
    vals_are_one = np.isclose(all_eq_df["sum_check"], 1).all()
    print(f"are all state/year norm sums equal to 1? {vals_are_one}")
    if not vals_are_one:
        raise ValueError("not all values are normed correctly!")

    # plot. Not hugely informative, but shows the data is there.
    out_ds["population"].sel(year=2020).plot.imshow()

    out_ds["population"].transpose("year", "y", "x").round(10).rio.write_crs(
        ras_crs
    ).to_netcdf(output_path)
    # %%
