# %%
# %load_ext autoreload
# %autoreload 2

import multiprocessing
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import osgeo  # noqa
import rasterio
import rioxarray
import xarray as xr
from geocube.api.core import make_geocube
from pyarrow import parquet  # noqa
from pytask import Product, mark, task

from gch4i.config import (
    global_data_dir_path,
    proxy_data_dir_path,
    sector_data_dir_path,
    years,
)
from gch4i.utils import make_raster_binary, mask_raster_parallel, warp_to_gepa_grid

NUM_WORKERS = multiprocessing.cpu_count()


# define a function to normalize the population data by state and year
def normalize(x):
    return x / x.sum()


"""
Hi all,

Land converted to wetlands have emissions if converted to palustrine wetlands only:
CCAP classes Palustrine Scrub/Shrub Wetland or Palustrine Emergent Wetland.
They assume the emissions of the palustrine wetlands while they are in the Land
Converted to Wetlands 20-year hold period.

For a slightly different way of explaining it, the text from the wetlands GHG inventory
chapter that describes how to calculate the total area in this category: "Tier 1
estimates of CH4 emissions for land converted to vegetated coastal wetlands are derived
from the same wetland map used in the analysis of wetland soil carbon fluxes for
palustrine wetlands... Because land converted to vegetated coastal wetlands is held in
this category for up to 20 years before transitioning to vegetated coastal wetlands
remaining to vegetated coastal wetlands, CH4 emissions in a given year represent the
cumulative area held in this category for that year and the prior 19 years."

In the spreadsheet Kenna attached in the previous email, columns BT - CZ in the
'12-6.9 Land ConvertedtoWetland' sheet have the formula for how emissions get
calculated in the 20-year hold period.

Monica
"""

# %%
cw_dir_path = sector_data_dir_path / "coastal_wetlands"
tidal_mask_warped = cw_dir_path / "tidal_mask_warped.tif"
proxy_output_path = proxy_data_dir_path / "coastal_wetlands_proxy.nc"
# %%
# https://coastalimagery.blob.core.windows.net/ccap-landcover/CCAP_bulk_download/Regional_30meter_Land_Cover/ccap-class-scheme-highres.pdf
COASTAL_WETLAND_CLASSES = np.array([13, 14, 15])
CCAP_YEARS = [2010, 2016]


# %%
@mark.persist
@task(id="prep_coastal_wetlands_mask")
def task_prep_coastal_wetlands_mask(
    ccap_path: Path = (
        cw_dir_path / f"conus_{CCAP_YEARS[0]}_ccap_landcover_20200311.tif"
    ),
    tidal_mask_path: Path = cw_dir_path / "tidal_mhhws_extent.img",
    tidal_output_path: Annotated[Path, Product] = tidal_mask_warped,
) -> None:
    warp_to_gepa_grid(
        input_path=tidal_mask_path,
        output_path=tidal_output_path,
        target_path=ccap_path,
        resampling="nearest",
    )


# %%
# We have to 3 steps to process the ccap data: mask, binary, warp
# 1.    we use the tidal line data provided from the inventory team to mask out
#       the ccap data where they do not consider emissions to be occurring
# 2.    we convert the masked raster to binary based on the classes they consider to be
#       coastal wetlands that produce emissions
# 3.    we warp the binary raster to the gepa grid. Note here that since we converted
#       the raster to binary, we take the average of the binary values in the gepa grid
#       to get the proportion of the grid cell that is considered to be a coastal
#       wetlands.
prep_dict = dict()
for ccap_year in CCAP_YEARS:
    ccap_path = cw_dir_path / f"conus_{ccap_year}_ccap_landcover_20200311.tif"
    intermediate_masked_path = cw_dir_path / f"coastal_wetlands_{ccap_year}_masked.tif"
    intermediate_binary_path = cw_dir_path / f"coastal_wetlands_{ccap_year}_binary.tif"
    output_path_cw_gepa = cw_dir_path / f"coastal_wetlands_{ccap_year}_gepa.tif"
    prep_dict[ccap_year] = dict(
        ccap_path=ccap_path,
        intermediate_masked_path=intermediate_masked_path,
        intermediate_binary_path=intermediate_binary_path,
        tidal_mask_path=tidal_mask_warped,
        output_path_cw_gepa=output_path_cw_gepa,
    )


for id, kwargs in prep_dict.items():

    @mark.persist
    @task(id=f"prep_ccap_data_{id}", kwargs=kwargs)
    def task_prep_ccap_data(
        ccap_path: Path,
        intermediate_masked_path: Path,
        intermediate_binary_path: Path,
        tidal_mask_path: Path,
        output_path_cw_gepa: Path,
    ) -> None:

        # mask the ccap data
        print(f"masking ccap data for {ccap_year}")
        mask_raster_parallel(
            input_path=ccap_path,
            output_path=intermediate_masked_path,
            mask_path=tidal_mask_path,
        )
        print(" done.")

        # then we convert the masked raster to binary
        print(f"converting masked raster to binary for {ccap_year}...", end="")
        make_raster_binary(
            input_path=intermediate_masked_path,
            output_path=intermediate_binary_path,
            true_vals=COASTAL_WETLAND_CLASSES,
            num_workers=NUM_WORKERS,
        )
        print(" done.")

        print(f"warping binary raster to gepa grid for {ccap_year}...", end="")
        warp_to_gepa_grid(
            input_path=intermediate_binary_path,
            output_path=output_path_cw_gepa,
            target_path=None,
            resampling="average",
            num_threads=NUM_WORKERS,
        )
        print(" done.")


# %%


@mark.persist
@task
def task_coastal_wetlands_proxy(
    input_paths: list[Path] = list(cw_dir_path.glob("coastal_wetlands_*_gepa.tif")),
    state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_output_path,
) -> None:

    ccap_list = []
    for ccap_path in input_paths:
        ccap_year = int(ccap_path.stem.split("_")[2])
        # read the data
        ccap_xr = (
            rioxarray.open_rasterio(ccap_path)
            .rename(f"ccap_{ccap_year}")
            .squeeze("band")
            .drop_vars("band")
        )

        # get the years each of them covers
        if ccap_year == 2010:
            ccap_years = list(years)[:4]
        elif ccap_year == 2016:
            ccap_years = list(years)[4:]

        # repeat the data along the time dimension
        ccap_xr = ccap_xr.expand_dims(year=ccap_years)
        ccap_list.append(ccap_xr)

    with rasterio.open(ccap_path) as src:
        ras_crs = src.crs

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

    comb_ccap_cr = xr.concat(ccap_list, dim="year").rename("cw_perc")

    state_grid = make_geocube(
        vector_data=state_gdf, measurements=["statefp"], like=comb_ccap_cr, fill=99
    )
    comb_ccap_cr["statefp"] = state_grid["statefp"]

    comb_ccap_cr = xr.where(comb_ccap_cr["statefp"] == 99, np.nan, comb_ccap_cr)

    # plot the data to check
    # comb_ccap_cr["statefp"].plot()

    # apply the normalization function to the population data
    proxy_xr = (
        comb_ccap_cr.groupby(["year", "statefp"])
        .apply(normalize)
        .sortby(["year", "y", "x"])
        .to_dataset(name="rel_emi")
    )
    proxy_xr["rel_emi"].shape

    # This is a tricky one to validate because many states do not have any data and so
    # their sums should be 0. So we check both and visually check the results also.
    all_eq_df = (
        proxy_xr["rel_emi"]
        .groupby(["statefp", "year"])
        .sum()
        .rename("sum_check")
        .to_dataframe()
        .drop(columns="spatial_ref")
        .reset_index()
        # .query("statefp < 60")
        .set_index(["year", "statefp"])
        .assign(
            is_close=lambda df: (np.isclose(df["sum_check"], 1))
            | (np.isclose(df["sum_check"], 0))
        )
    )
    all_eq_df

    if not all_eq_df["is_close"].all():
        raise ValueError("not all values are normed correctly!")

    proxy_xr.transpose("year", "y", "x").round(10).rio.write_crs(ras_crs).to_netcdf(
        output_path
    )

    # %%
