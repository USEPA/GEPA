# %%

from pathlib import Path
from typing import Annotated
import pandas as pd
import geopandas as gpd

from pytask import Product, mark
import rioxarray.merge

from gch4i.config import (
    global_data_dir_path,
    sector_data_dir_path,
    proxy_data_dir_path,
    tmp_data_dir_path,
    years,
)
import xarray as xr
from gch4i.utils import (
    normalize,
    GEPA_spatial_profile,
)
import numpy as np
import rioxarray
from geocube.api.core import make_geocube
import rasterio


# %%

source_dir = sector_data_dir_path / "combustion_stationary"


@mark.persist
def task_rwc_proxy(
    NEI_resi_wood_inputfile: Path = source_dir / "NEI 2020 RWC Throughputs.xlsx",
    pop_input_path: Path = tmp_data_dir_path / "usa_ppp_2020_reprojected.tif",
    county_path: Path = global_data_dir_path / "tl_2020_us_county.zip",
    state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_data_dir_path / "rwc_proxy.nc",
) -> None:

    # %%
    # replacements identified in v2.
    # replacements = {
    #     "46102": "46113",
    #     "02063": "02020",
    #     "02066": "02261",
    #     "02158": "02060",
    # }
    # NOTE: these replacements were done in v2 and they appear to match an earlier
    # vintage of county fips codes. The 2020 data match the census data, so these
    # replacements are not needed.

    rwc_df = (
        pd.read_excel(NEI_resi_wood_inputfile)
        .query("(ThroughputUnit == 'TON') & (SourceClassificationCode != 2104009000)")
        .rename(
            columns={"StateAndCountyFIPSCode": "fips", "PostalStateCode": "state_code"}
        )
        .rename(columns=str.lower)
        .assign(fips=lambda df: df["fips"].astype(str).str.zfill(5))
        # .replace(replacements)
    )
    rwc_df.head()
    # %%
    # sum up the throughput by county
    county_sums = rwc_df.groupby(["fips"])["throughput"].sum()
    county_sums.head()
    # %%
    # read in the county geospatial data
    county_gdf = (
        gpd.read_file(county_path)
        .rename(columns=str.lower)
        .astype({"statefp": int})
        .query("(statefp < 60)")
        .to_crs(4326)
    )
    county_gdf.head()
    # %%
    # join the throughput with the geo data
    county_w_rwc_gdf = county_gdf.merge(
        county_sums, left_on="geoid", right_index=True, how="left"
    )
    # %%
    # plot the data to have a look
    county_w_rwc_gdf.plot("throughput", cmap="hot", legend=True)
    # %%
    # we can see here that when joined with states, there are no missing data. If we
    # were to perform the replacements, we would find missing data.
    county_w_rwc_gdf[county_w_rwc_gdf["throughput"].isna()]
    # %%
    # read in the population data
    pop_xr = (
        rioxarray.open_rasterio(pop_input_path)
        .squeeze("band")
        .drop_vars("band")
        .where(lambda x: x >= 0)
    )
    pop_xr

    # geocube requires the geoid to be an integer, so we create a simplified
    # geodataframe to convert to grid
    cnty_int_gdf = (
        county_w_rwc_gdf.copy()
        .astype({"geoid": int})[["geometry", "geoid", "throughput"]]
        .set_index("geoid")
    )
    cnty_grid = make_geocube(
        vector_data=cnty_int_gdf.reset_index(),
        measurements=["geoid"],
        like=pop_xr,
        fill=99,
    )

    # assign the county geoid to the population grid
    pop_xr["geoid"] = cnty_grid["geoid"]
    pop_xr
    # %%
    # normalize the population at the county level
    cnty_pop_normed_xr = (
        pop_xr.groupby(["geoid"])
        .apply(normalize)
        .sortby(["y", "x"])
        .where(lambda x: x["geoid"] >= 0, drop=True)
    )
    cnty_pop_normed_xr
    # %%
    # we not now have a population dataset that is normalized to the county level
    # so we now allocate the tons of throughput of residential wood combustion to the
    # county level. This will take the TONS of rwc and allocate it to the county level
    # by the normalize population data.
    results = []
    for geoid, data in cnty_pop_normed_xr.groupby("geoid"):
        # if this isn't a county, make the values 0
        if geoid == 99:
            throughput = 0
        else:
            throughput = cnty_int_gdf.loc[geoid]["throughput"]
        res = data * throughput
        results.append(res)
    # %%
    # put the results back together
    rwc_xr = (
        xr.concat(results, dim="stacked_y_x")
        .unstack("stacked_y_x")
        .drop_vars("geoid")
        .sortby(["y", "x"])
        .rio.set_spatial_dims(x_dim="x", y_dim="y")
        .rio.write_crs(4326)
        .rio.write_transform(pop_xr.rio.transform())
        .rio.set_attrs(pop_xr.attrs)
    )
    rwc_xr
    rwc_xr.plot()
    # %%
    # this is a hacky thing, but for some reason the above concat strips the geo
    # information from the xarray and I can't get it back in there. So I save the file
    # to a temporary file and read it back in. rioxarray.open_rasterio restores the geo
    # information so then we can keep going.

    gepa_profile = GEPA_spatial_profile()
    gepa_profile.profile["count"] = 1
    tmp_file = rasterio.MemoryFile()
    rwc_xr.rio.to_raster(tmp_file.name, profile=gepa_profile.profile)
    rwc_xr = rioxarray.open_rasterio(tmp_file).squeeze("band").drop_vars("band")
    rwc_xr.plot()

    # %%
    # read the state geospatial data and make it into a grid. Assign that grid to the
    # rwc data so we can normalize the data to the state level.
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
    # make the grid
    state_grid = make_geocube(
        vector_data=state_gdf, measurements=["statefp"], like=rwc_xr, fill=99
    )
    # assign the state grid as a new variable in the dataset
    rwc_xr["statefp"] = state_grid["statefp"]
    # plot the data to check
    rwc_xr["statefp"].plot()

    # %%

    # apply the normalization function to the data
    out_ds = (
        rwc_xr.groupby(["statefp"])
        .apply(normalize)
        .to_dataset(name="rel_emi")
        .expand_dims(year=years)
        .sortby(["year", "y", "x"])
    )
    out_ds["rel_emi"].shape
    # %%
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
    # %%
    out_ds["rel_emi"].transpose("year", "y", "x").round(10).rio.write_crs(
        rwc_xr.rio.crs
    ).to_netcdf(output_path)
    # %%
