"""
Name:                   task_rice_area_proxy.py
Date Last Modified:     2025-01-23
Authors Name:           Nick Kruskamp (RTI International)
Purpose:                This script is used to create the rice area proxy using a
                        combination of USDA Census data and the CDL data. Using 2012
                        and 2017 census data, we first normalized to the county level
                        and then allocated to the grid level based on the CDL data. The
                        CDL data are taken from the nass_cdl_proxy script Where I have
                        written the bulk of the CDL processing code. So that must run
                        first before this.
Input Files:            - CDQT: {sector_data_dir_path}/USDA_NASS/2017_cdqt_data.txt
                        - Census Data: {sector_data_dir_path}/USDA_NASS/
                            census_2012-2017_rice_area_harvested.csv
                        - County Shapefile: {global_data_dir_path}/tl_2020_us_county.zip
                        - State Shapefile: {global_data_dir_path} /tl_2020_us_state.zip
                        - CDL Data: {sector_data_dir_path} /nass_cdl/*rice_perc*.tif
Output Files:           - {proxy_data_dir_path}/rice_area_proxy.nc

"census_2012-2017_rice_area_harvested.csv"
input data download:
Using the quick stats tool, I got the following static urls:

https://quickstats.nass.usda.gov/save/B20660AC-76CC-3050-A17C-A2019120B723/census_rice_area.url

https://quickstats.nass.usda.gov/results/B20660AC-76CC-3050-A17C-A2019120B723


"2017_cdqt_data.txt"
this is used as a reference file for comparing the national, state, and county
level data. I noticed that in the interative tool, the national level data is not
the sum of the data I was seeing for the county level query, so I downloaded this
file to compare. What I found is that the national, state, and county area
harvested data are not the same among the 3 geographic levels. I ASSUME this is
because they withhold some data at the state/county level for privacy reasons.
Ultimately, the totals are not the biggest concern, but it was worth checking to
make sure the data are correct. to get this file, go to
https://www.nass.usda.gov/Quick_Stats/CDQT/chapter/1/table/1/state/US/year/2017/
in the top left corner, look for the 3 horizontal lines, click that, then click
"(?) Download full dataset here".

This file is not necessary to create the proxy, but I keep it here for future reference
of validating our numbers.
"""

# %% Import Libraries
# %load_ext autoreload
# %autoreload 2
from pathlib import Path
from typing import Annotated

import pyarrow.parquet  # noqa
import osgeo  # noqa
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import seaborn as sns
import xarray as xr
from geocube.api.core import make_geocube
from pytask import Product, mark, task

from gch4i.config import global_data_dir_path, proxy_data_dir_path, sector_data_dir_path
from gch4i.utils import GEPA_spatial_profile, name_formatter, normalize_xr

pd.set_option("future.no_silent_downcasting", True)

GEPA_PROFILE = GEPA_spatial_profile()
GEPA_PROFILE.profile["count"] = 1

sector_dir: Path = sector_data_dir_path / "USDA_NASS"
cdl_path: Path = sector_data_dir_path / "nass_cdl"


# %% Pytask Function


@mark.persist
@task(id="rice_area_proxy")
def task_rice_proxy(
    cdqt_file_path: Path = sector_dir / "2017_cdqt_data.txt",
    census_file_path: Path = sector_dir / "census_2012-2017_rice_area_harvested.csv",
    county_path: Path = global_data_dir_path / "tl_2020_us_county.zip",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    cdl_layers: list[Path] = list(cdl_path.glob("*rice_perc*.tif")),
    output_path: Annotated[Path, Product] = proxy_data_dir_path / "rice_area_proxy.nc",
) -> None:

    # here we see that as we cycle through the 3 geographic levels from the ref file,
    # the total area harvested changes. When compared to the actual data file I'm using
    # for 2017, the county totals equal what is in this file. So we have an external
    # validation that the numbers we're using align with data from other places in the
    # USDA census.
    cdqt_df = pd.read_csv(
        cdqt_file_path,
        sep="\t",
        encoding="latin1",
        usecols=["SHORT_DESC", "AGG_LEVEL_DESC", "CENSUS_TABLE", "VALUE"],
    )
    for geo_level in ["NATIONAL", "STATE", "COUNTY"]:
        sum_check = (
            (
                cdqt_df.query(
                    "(SHORT_DESC == 'RICE - ACRES HARVESTED') & "
                    "(AGG_LEVEL_DESC == @geo_level) & "
                    "(CENSUS_TABLE == 25)"
                ).assign(
                    value_num=lambda df: pd.to_numeric(
                        df["VALUE"].str.replace(",", ""), errors="coerce"
                    ),
                )
            )["value_num"]
            .sum()
            .astype(int)
        )
        print(f"{geo_level:<10}:{sum_check:>20,}")

    # read the state and county geospatial data
    state_gdf = (
        gpd.read_file(state_path)
        .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code", "name": "state_name"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .to_crs(4326)
    )

    county_gdf = (
        gpd.read_file(county_path)
        .rename(columns=str.lower)
        .astype({"statefp": int})
        .merge(state_gdf[["state_code", "statefp"]], on="statefp")
        .assign(
            formatted_county=lambda df: name_formatter(df["name"]),
            formatted_state=lambda df: name_formatter(df["state_code"]),
        )
        .to_crs(4326)
    )

    # read the rice area harvested data downloaded from the USDA NASS quick stats tool
    # we need to get the full county geoid and turn the value into a number we can use.
    census_df = (
        pd.read_csv(
            census_file_path,
            usecols=["Year", "State ANSI", "County ANSI", "Value", "County"],
        )
        .assign(
            geoid=lambda df: df["State ANSI"].astype(str).str.zfill(2)
            + df["County ANSI"].astype(int).astype(str).str.zfill(3),
            value_num=lambda df: pd.to_numeric(
                df["Value"].str.replace(",", ""), errors="coerce"
            ),
        )
        .loc[:, ["Year", "geoid", "value_num", "County"]]
    ).sort_values("Year")

    # process the area harevsted data
    data_dict = {}
    for the_year, rice_harv_df in census_df.groupby("Year"):

        print(f"{the_year}:\t\t{int(rice_harv_df['value_num'].sum()):,}")

        rice_gdf = (
            county_gdf[["geoid", "geometry"]]
            .merge(rice_harv_df, how="right", on="geoid")
            .astype({"geoid": int})
            .set_index("geoid")
            .dropna(subset="value_num")
        )
        if rice_gdf.geometry.isna().any():
            raise ValueError(f"Missing geometry for {the_year}")

        # this is a byproduct I'm not using, but it is nice to have as reference
        rice_gdf.to_parquet(sector_dir / f"rice_area_cnty_{the_year}.parquet")

        data_dict[the_year] = rice_gdf

    _, axs = plt.subplots(len(data_dict), figsize=(20, 5), dpi=300)
    for (the_year, rice_gdf), ax in zip(data_dict.items(), axs.ravel()):
        rice_gdf.plot("value_num", cmap="Spectral", lw=0.5, ax=ax)
        state_gdf.boundary.plot(ax=ax, lw=0.5, color="xkcd:slate", zorder=-1)
        ax.set_title(the_year)
    sns.despine()
    plt.show()

    # read in the population data. This one is reference to create the county grid
    rice_xr = (
        rioxarray.open_rasterio(cdl_layers[0])
        .squeeze("band")
        .drop_vars("band")
        # .where(lambda x: x >= 0)
    )
    rice_xr

    # make the county grid
    cnty_grid = make_geocube(
        vector_data=rice_gdf.reset_index(),
        measurements=["geoid"],
        like=rice_xr,
        fill=99,
    )
    rice_gdf[~rice_gdf.index.isin(np.unique(cnty_grid["geoid"].values))]

    res_dict = {}
    for cdl_layer_path in cdl_layers:
        the_year = int(cdl_layer_path.stem.split("_")[0])
        if the_year < 2017:
            rice_gdf = data_dict[2012]
        else:
            rice_gdf = data_dict[2017]

        rice_xr = (
            rioxarray.open_rasterio(cdl_layer_path)
            .squeeze("band")
            .drop_vars("band")
            .where(lambda x: x >= 0)
        )

        # assign the county geoid to the population grid
        rice_xr["geoid"] = cnty_grid["geoid"]

        # normalize the population at the county level
        cnty_rice_normed_xr = (
            rice_xr.groupby(["geoid"])
            .apply(normalize_xr)
            .sortby(["y", "x"])
            .where(lambda x: x["geoid"] >= 0, drop=True)
        )

        # we not now have a population dataset that is normalized to the county level
        # so we now allocate the tons of throughput of residential wood combustion to
        # the county level. This will take the TONS of rwc and allocate it to the
        # county level by the normalize population data.
        results = []
        for geoid, data in cnty_rice_normed_xr.groupby("geoid"):
            # if this isn't a county, make the values 0
            if geoid == 99:
                val = 0
            else:
                try:
                    val = rice_gdf.loc[geoid]["value_num"]
                except KeyError:
                    val = 0
                    # print(f"mismatch between county and raster data {geoid}")

            res = data * val
            results.append(res)

        # put the results back together
        rice_xr = (
            xr.concat(results, dim="stacked_y_x")
            .unstack("stacked_y_x")
            .rename("rice_area")
            # .drop_vars("geoid")
            .sortby(["y", "x"])
            .rio.set_spatial_dims(x_dim="x", y_dim="y")
            .rio.write_crs(4326)
            .rio.write_transform(rice_xr.rio.transform())
            .rio.set_attrs(rice_xr.attrs)
        )
        rice_xr
        rice_xr.plot()
        mismatch_results = (
            rice_xr.groupby("geoid")
            .sum()
            .rename("cnty_sum")
            .to_dataframe()
            .drop(columns="spatial_ref")
            .join(rice_gdf[["value_num"]], how="outer")
            .assign(is_close=lambda df: np.isclose(df["cnty_sum"], df["value_num"]))
            .query("is_close == False")
        )
        if not mismatch_results.empty:
            print(mismatch_results)

        # this is a hacky thing, but for some reason the above concat strips the geo
        # information from the xarray and I can't get it back in there. So I save the
        # file to a temporary file and read it back in. rioxarray.open_rasterio
        # restores the geo information so then we can keep going.

        tmp_file = rasterio.MemoryFile()
        rice_xr.rio.to_raster(tmp_file.name, profile=GEPA_PROFILE.profile)
        rice_xr = (
            rioxarray.open_rasterio(tmp_file)
            .squeeze("band")
            .drop_vars("band")
            .assign_coords(year=the_year)
        )
        rice_xr.plot()
        plt.show()
        res_dict[the_year] = rice_xr

    rice_stack_xr = xr.concat(
        list(res_dict.values()), dim="year", create_index_for_new_dim=True
    )
    rice_stack_xr

    # make the state grid
    state_grid = make_geocube(
        vector_data=state_gdf, measurements=["statefp"], like=rice_stack_xr, fill=99
    )
    # assign the state grid as a new variable in the dataset
    rice_stack_xr["statefp"] = state_grid["statefp"]
    # plot the data to check
    rice_stack_xr["statefp"].plot()

    # apply the normalization function to the data
    out_ds = (
        rice_stack_xr.groupby(["statefp", "year"])
        .apply(normalize_xr)
        .to_dataset(name="rel_emi")
        # .expand_dims(year=years)
        .sortby(["year", "y", "x"])
    )
    out_ds["rel_emi"].shape

    # check that the normalization worked
    all_eq_df = (
        out_ds["rel_emi"]
        .groupby(["statefp", "year"])
        .sum()
        .rename("sum_check")
        .to_dataframe()
        .drop(columns="spatial_ref")
        .assign(
            # NOTE: Due to floating point rouding, we need to check if the sum is
            # close to 1, not exactly 1.
            is_close=lambda df: (np.isclose(df["sum_check"], 1))
            | (np.isclose(df["sum_check"], 0))
        )
    )

    vals_are_one = all_eq_df["is_close"].all()
    print(f"are all state/year norm sums equal to 1? {vals_are_one}")
    if not vals_are_one:
        raise ValueError("not all values are normed correctly!")

    # plot. Not hugely informative, but shows the data is there.
    out_ds["rel_emi"].sel(year=2020).plot.imshow()

    out_ds["rel_emi"].transpose("year", "y", "x").round(10).rio.write_crs(
        rice_stack_xr.rio.crs
    ).to_netcdf(output_path)
