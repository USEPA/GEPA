"""
Name:                   task_coastal_wetlands_proxy.py
Date Last Modified:     2025-01-23
Authors Name:           Nick Kruskamp (RTI International)
Purpose:                This script processes the nass CDL intermediate products into
                        the all crop proxy. The all crop proxy uses the percent
                        of a pixel that contains any crop land as defined byt NASS CDL.
                        The script reads in the intermediate products, stacks them, and
                        then creates a new xarray dataset with the normalized all crop
                        proxy data. The data are then written out to a netcdf file.
Input Files:            - nass CDL intermediate products: {nass_cdl_path}/
                            *_all_crop_perc.tif
                        - State Shapefile: {global_data_dir_path} /tl_2020_us_state.zip
Output Files:           - {proxy_data_dir_path}/all_crop_proxy.nc
"""

# %%
# %load_ext autoreload
# %autoreload 2


from pathlib import Path
from typing import Annotated


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
from gch4i.utils import (
    GEPA_spatial_profile,
    normalize_xr,
)

# %%
crop_name = "all_crop"
nass_cdl_path = sector_data_dir_path / "nass_cdl"


@mark.persist
@task(id="all_crop_proxy")
def task_all_crop_proxy(
    input_paths: list[Path] = sorted(list(nass_cdl_path.glob("*_all_crop_perc.tif"))),
    state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_data_dir_path
    / f"{crop_name}_proxy.nc",
) -> None:

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

    # for the list of input files (the percentage layers for each year), read in the
    # data and stack them into a single numpy array
    raster_list = []
    for in_file in input_paths:
        if not in_file.exists():
            continue
        with rasterio.open(in_file) as src:
            raster_list.append(src.read(1))

    # stack the data
    output_data = np.stack(raster_list, axis=0)

    # create a profile for the output raster
    profile = GEPA_spatial_profile(0.1)
    profile.profile.update(count=len(raster_list), nodata=-99999)

    # write out the stack to a temporary file such that the geospatial data are
    # preserved. This makes reading in the data with rioxarray easier.
    tmp_stack_path = rasterio.MemoryFile()

    with rasterio.open(tmp_stack_path, "w", **profile.profile) as dst:
        dst.write(output_data)
        dst.descriptions = tuple([str(x) for x in years])

    # read in the raw crops data raster stack as a xarray dataset
    # masked will read the nodata value and set it to NA
    crop_ds = rioxarray.open_rasterio(tmp_stack_path, masked=True).rename(
        {"band": "year"}
    )

    # assign the band as our years so the output raster data has year band names
    crop_ds["year"] = years

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

    # plot the data to check
    crop_ds["statefp"].plot()

    # apply the normalization function to the crops data
    out_ds = (
        crop_ds.groupby(["year", "statefp"])
        .apply(normalize_xr)
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
        profile.profile["crs"]
    ).to_netcdf(output_path)
