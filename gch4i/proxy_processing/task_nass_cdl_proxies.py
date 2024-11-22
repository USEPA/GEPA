import multiprocessing
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

import osgeo  # noqa
import geopandas as gpd
import numpy as np
import rasterio

import rioxarray
from geocube.api.core import make_geocube
from pytask import Product, mark, task

from gch4i.config import (
    global_data_dir_path,
    proxy_data_dir_path,
    sector_data_dir_path,
    years,
)

from gch4i.utils import download_url

from gch4i.gridding import GEPA_spatial_profile, make_raster_binary, warp_to_gepa_grid



nass_cdl_path = sector_data_dir_path / "nass_cdl"

# https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/metadata_nc23.htm
cdl_crop_vals = np.concat([np.arange(1, 61), np.arange(66, 81), np.arange(195, 256)])
cdl_other_vals = np.concatenate([np.arange(61, 66), np.arange(81, 195)])

crop_val_dict = {
    "all_crop": cdl_crop_vals,
}


# %%


def unzip_cdl(zip_path, output_path):
    with ZipFile(zip_path, "r") as z:
        z.extract(output_path.name, output_path.parent)


# %%
cdl_download_dict = {}
for year in years:

    zip_file_name = f"{year}_30m_cdls.zip"
    tif_file_name = f"{year}_30m_cdls.tif"

    url = (
        "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/"
        f"{zip_file_name}"
    )
    zip_path = nass_cdl_path / zip_file_name
    cdl_input_path = nass_cdl_path / tif_file_name

    cdl_download_dict[f"cdl_{year}"] = dict(
        url=url,
        zip_path=zip_path,
        cdl_input_path=cdl_input_path,
    )

# (
#     url,
#     zip_path,
#     cdl_input_path,
# ) = cdl_download_dict["cdl_2014"].values()
# url, zip_path, cdl_input_path

for _id, kwargs in cdl_download_dict.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_download_cdl(
        url: str,
        zip_path: Annotated[Path, Product],
        cdl_path: Annotated[Path, Product],
    ):
        download_url(url, zip_path)
        unzip_cdl(zip_path, cdl_path)


# %%

calc_crop_perc = {}
for crop_name, crop_vals in crop_val_dict.items():
    for year in years:
        tif_file_name = f"{year}_30m_cdls.tif"
        cdl_input_path = nass_cdl_path / tif_file_name
        output_path_binary = nass_cdl_path / (
            cdl_input_path.stem + f"_{crop_name}_binary.tif"
        )
        output_path_perc = nass_cdl_path / (
            cdl_input_path.stem + f"_{crop_name}_perc.tif"
        )
        calc_crop_perc[f"cdl_{year}_{crop_name}"] = dict(
            cdl_input_path=cdl_input_path,
            output_path_binary=output_path_binary,
            output_path_perc=output_path_perc,
            crop_vals=crop_vals,
        )
calc_crop_perc

for _id, kwargs in calc_crop_perc.items():

    @mark.persist
    def task_calc_cdl_perc(
        cdl_input_path, output_path_binary, output_path_perc, crop_vals
    ):

        make_raster_binary(cdl_input_path, output_path_binary, crop_vals, num_workers)
        warp_to_gepa_grid(output_path_binary, output_path_perc)


# %%
proxy_stack_dict = {}
for crop_name in crop_val_dict.keys():
    # TODO: to be safe, list files specifically by year
    # for year in years: get input file paths
    proxy_stack_dict[f"crop_proxy_{crop_name}"] = dict(
        input_paths=sorted(list(nass_cdl_path.glob(f"*_{crop_name}_perc.tif"))),
        state_geo_path=global_data_dir_path / "tl_2020_us_state.zip",
        output_path_stacked=nass_cdl_path / f"{crop_name}_raw_stacked.tif",
        output_path=proxy_data_dir_path / f"{crop_name}_proxy.parquet",
    )

for _id, kwargs in proxy_stack_dict.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_crop_proxy(
        input_paths: list[Path],
        state_geo_path: Path,
        output_path_stacked: Annotated[Path, Product],
        output_path: Annotated[Path, Product],
    ) -> None:

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

        with rasterio.open(output_path_stacked, "w", **profile.profile) as dst:
            dst.write(output_data)
            dst.descriptions = tuple([str(x) for x in years])

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

        # read in the raw crops data raster stack as a xarray dataset
        # masked will read the nodata value and set it to NA
        crop_ds = rioxarray.open_rasterio(output_path_stacked, masked=True).rename(
            {"band": "year"}
        )

        with rasterio.open(output_path_stacked) as src:
            ras_crs = src.crs

        # assign the band as our years so the output raster data has year band names
        crop_ds["year"] = years
        # remove NA values
        # pop_ds = pop_ds.where(pop_ds != -99999)

        # create a state grid to match the input crops array
        # we use fill here to fill in the nodata values with 99 so that when we do the
        # groupby, the nodata area is not collapsed, and the resulting dimensions align
        # with the v3 gridded data.
        state_grid = make_geocube(
            vector_data=state_gdf, measurements=["statefp"], like=crop_ds, fill=99
        )
        state_grid
        # assign the state grid as a new variable in the crops dataset
        crop_ds["statefp"] = state_grid["statefp"]
        crop_ds

        # plot the data to check
        crop_ds["statefp"].plot()

        # define a function to normalize the crops data by state and year
        def normalize(x):
            return x / x.sum()

        # apply the normalization function to the crops data
        out_ds = (
            crop_ds.groupby(["year", "statefp"])
            .apply(normalize)
            .sortby(["year", "y", "x"])
            .to_dataset(name="crops")
        )
        out_ds["crops"].shape

        # check that the normalization worked
        all_eq_df = (
            out_ds["crops"]
            .groupby(["statefp", "year"])
            .sum()
            .rename("sum_check")
            .to_dataframe()
            .drop(columns="spatial_ref")
        )

        # NOTE: Due to floating point rouding, we need to check if the sum is close to
        # 1, not exactly 1.
        vals_are_one = np.isclose(all_eq_df["sum_check"], 1).all()
        print(f"are all state/year norm sums equal to 1? {vals_are_one}")
        if not vals_are_one:
            raise ValueError("not all values are normed correctly!")

        # plot. Not hugely informative, but shows the data is there.
        out_ds["crops"].sel(year=2020).plot.imshow()

        out_ds["crops"].transpose("year", "y", "x").round(10).rio.write_crs(
            ras_crs
        ).to_netcdf(output_path)
