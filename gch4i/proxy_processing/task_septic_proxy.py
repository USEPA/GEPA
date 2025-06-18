"""
Name:                   task_septic_proxy.py
Date Last Modified:     2025-03-11
Authors Name:           Nick Kruskamp (RTI International)
Purpose:
Input Files:            -
                        - Population: {tmp_data_dir_path}/usa_ppp_*_reprojected.tif
                        - County: {global_data_dir_path}/tl_2020_us_county.zip
                        - State: {global_data_dir_path}/tl_2020_us_state.zip
Output Files:           - {proxy_data_dir_path}/septic_proxy.nc
"""

"""
https://human-settlement.emergency.copernicus.eu/download.php?ds=smod




guide to data classes in SMOD product:
https://human-settlement.emergency.copernicus.eu/documents/GHSL_Data_Package_2023.pdf

The settlement grid at level 2 represents these definitions on a layer grid. Each pixel
is classified as follow:
—   Class 30: “Urban Centre grid cell”, if the cell belongs to an Urban Centre spatial
    entity;
—   Class 23: “Dense Urban Cluster grid cell”, if the cell belongs to a Dense Urban
    Cluster spatial entity;
—   Class 22: “Semi-dense Urban Cluster grid cell”, if the cell belongs to a Semi-dense
    Urban Cluster spatial entity;
—   Class 21: “Suburban or peri-urban grid cell”, if the cell belongs to an Urban
    Cluster cells at first hierarchical level but is not part of a Dense or Semi-dense
    Urban Cluster;
—   Class 13: “Rural cluster grid cell”, if the cell belongs to a Rural Cluster spatial
    entity;
—   Class 12: “Low Density Rural grid cell”, if the cell is classified as Rural grid
    cells at first hierarchical level, has more than 50 inhabitant and is not part of a
    Rural Cluster;
—   Class 11: “Very low density rural grid cell”, if the cell is classified as Rural
    grid cells at first hierarchical level, has less than 50 inhabitant and is not part
    of a Rural Cluster;
—   Class 10: “Water grid cell”, if the cell has 0.5 share covered by permanent surface
    water and is not populated nor built.

The percent of households that use a septic system by degree of urbanization. In this
product, the EPA has rural, urban, and urban core.
https://nepis.epa.gov/Exe/tiff2png.cgi/P1004624.PNG?-r+75+-g+7+D%3A%5CZYFILES%5CINDEX%20DATA%5C06THRU10%5CTIFF%5C00000431%5CP1004624.TIF

or weights calculated from this table:
https://www.census.gov/programs-surveys/ahs/data/interactive/ahstablecreator.html?s_areas=00000&s_year=2023&s_tablename=TABLE4&s_bygroup1=17&s_bygroup2=1&s_filtergroup1=1&s_filtergroup2=1

Source of census Urban data.
https://www2.census.gov/geo/tiger/TIGER2020/UAC/

# septic system weights by urban class as published by the EPA in this report
# https://nepis.epa.gov/Exe/tiff2png.cgi/P1004624.PNG?-r+75+-g+7+D%3A%5CZYFILES%5CINDEX%20DATA%5C06THRU10%5CTIFF%5C00000431%5CP1004624.TIF
# weights = {1: 0.5, 2: 0.47, 3: 0.03}


"""

# %% Import Libraries
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import rioxarray.merge
import seaborn as sns
import xarray as xr
from geocube.api.core import make_geocube
from pytask import Product, mark
from tqdm.auto import tqdm

from gch4i.config import (
    global_data_dir_path,
    proxy_data_dir_path,
    sector_data_dir_path,
    tmp_data_dir_path,
    years,
)
from gch4i.utils import GEPA_spatial_profile, normalize_xr

# %%
source_dir = sector_data_dir_path / "septic"


# %%
@mark.persist
def task_septic_proxy(
    pop_input_paths: Path = list(tmp_data_dir_path.glob("usa_ppp_*_reprojected.tif")),
    smod_zip_path: str = f"zip://{source_dir}/GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0.zip",
    smod_shp_path: str = "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_UC_V2_0.shp",
    census_urban_path: str = f"zip://{source_dir}/tl_2020_us_uac20_corrected.zip",
    ahs_septic_use_path: Path = source_dir / "AHS TABLE4 3_12_2025.csv",
    state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_data_dir_path / "septic_pop_proxy.nc",
) -> None:

    # %%
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
    # %%
    septic_df = (
        pd.read_csv(ahs_septic_use_path, skiprows=1)
        .dropna(how="all", axis=0)
        .rename(columns={"Unnamed: 0": "variable"})
        .assign(variable=lambda df: df["variable"].str.strip())
        .set_index("variable")
        .T
    )
    # %%
    septic_df["Septic tank or cesspool"] = pd.to_numeric(
        septic_df["Septic tank or cesspool"].str.replace(r"\D", "", regex=True)
    )
    septic_df["Total"] = pd.to_numeric(
        septic_df["Total"].str.replace(r"\D", "", regex=True)
    )
    # %%
    septic_use_weights = (
        septic_df["Septic tank or cesspool"]
        / septic_df.loc["Total", "Septic tank or cesspool"]
    )
    weights = dict(zip([2, 3, 1], septic_use_weights.iloc[1:].values))
    weights_df = pd.DataFrame.from_dict(
        weights, orient="index", columns=["weight"]
    ).sort_index()
    weights_df

    # %%

    gepa_profile = GEPA_spatial_profile()

    # read in the time-series population data that we have.
    pop_arrs_list = []
    for pop_path in tqdm(pop_input_paths, desc="Loading Population Data"):
        with rasterio.open(pop_path) as src:
            pop_arr = src.read(1, masked=True)
            # pop_arr = np.where(pop_arr > 0, pop_arr, np.nan)
            pop_arrs_list.append(pop_arr)

    # this duplicates the 2020 data into 2021 and 2022 since those population data
    # are not available yet.
    pop_arrs_list.append(pop_arr)
    pop_arrs_list.append(pop_arr)

    # stack the population arrays together
    pop_arr = np.stack(pop_arrs_list)
    pop_arr = np.astype(np.flip(pop_arr, axis=1), np.float64)
    # %%
    # Turn the pop data into an xarray DataArray
    tmp_xr = xr.DataArray(
        pop_arr,
        dims=["year", "y", "x"],
        coords={
            "year": years,
            "y": GEPA_spatial_profile.y,
            "x": GEPA_spatial_profile.x,
        },
    )  # .where(lambda x: x > 0)
    tmp_file = rasterio.MemoryFile()

    gepa_profile.profile.update(count=len(years))

    tmp_xr.rio.to_raster(tmp_file.name, profile=gepa_profile.profile)

    # %%
    state_geom = state_gdf.geometry.union_all()

    # we read in the SMOD data to represent our urban core regions
    smod_gdf = (
        gpd.read_file(f"{smod_zip_path}!/{smod_shp_path}")
        .to_crs(state_gdf.crs)
        .explode()
        .loc[:, ["geometry"]]
        .assign(urban=3)
    )
    smod_gdf = smod_gdf.loc[smod_gdf.geometry.intersects(state_geom)]

    # we read in the census urban data to represent our sub/peri urban regions
    census_urban_gdf = (
        gpd.read_file(census_urban_path)
        .to_crs(state_gdf.crs)
        .explode()
        .loc[:, ["geometry"]]
        .assign(urban=2)
    )
    census_urban_gdf = census_urban_gdf.loc[
        census_urban_gdf.geometry.intersects(state_geom)
    ]

    # %%
    # we remove the overlap between the two datasets
    census_urban_peri_gdf = census_urban_gdf.overlay(smod_gdf, how="difference")
    # then concat them together
    urban_gdf = pd.concat([smod_gdf, census_urban_peri_gdf])
    urban_gdf.to_parquet(tmp_data_dir_path / "urban_gdf.parquet")

    # %%
    ax = smod_gdf.plot(color="xkcd:teal")
    census_urban_peri_gdf.plot(ax=ax, color="xkcd:purple")
    state_gdf.boundary.plot(ax=ax, color="black", linewidth=0.5)
    # %%
    pop_xr = (
        (
            rioxarray.open_rasterio(tmp_file)
            .rename({"band": "year"})
            .assign_coords(year=years)
        )
        .rio.write_crs(4326)
        .where(lambda x: x > 0)
    )
    pop_xr
    pop_xr.sel(year=years[0]).plot(cmap="hot")

    urban_grid = make_geocube(
        vector_data=urban_gdf,
        measurements=["urban"],
        like=pop_xr,
        fill=1,
    )

    # this is a temporary file as a convenience to look at ourside of Python. So I don't
    # list it in the function.
    urban_grid_path = tmp_data_dir_path / "septic_urban.tif"

    try:
        urban_grid.rio.to_raster(urban_grid_path, profile=gepa_profile.profile)
    except Exception as e:
        print(f"Failed to write urban grid {e}")

    # we now normalize these relative population values at the state level
    state_grid = make_geocube(
        vector_data=state_gdf,
        measurements=["statefp"],
        like=pop_xr,
        fill=99,
    )
    pop_xr["statefp"] = state_grid["statefp"]
    pop_xr["urban"] = urban_grid["urban"]

    # %%
    # this is where the urban class scaling factors are calculated. We get the scale for
    # each urban class and then scale the population data by that factor.
    results = []
    for (year, urban_class), data in pop_xr.groupby(["year", "urban"]):
        total_pop = data.sum().values
        scaling_factor = weights_df.loc[urban_class, "weight"] / total_pop
        res = data * scaling_factor
        results.append(res)
    results[0]
    # %%

    # put the results back together of the now class scaled data.
    weighted_pop_xr = (
        xr.concat(results, dim="stacked_year_y_x")
        .unstack("stacked_year_y_x")
        .sortby(["year", "y", "x"])
        .rename("weighted_pop")
        .rio.set_spatial_dims(x_dim="x", y_dim="y")
        .rio.write_crs(4326)
        .rio.write_transform(pop_xr.rio.transform())
        .rio.set_attrs(pop_xr.attrs)
    )
    # weighted_pop_xr = weighted_pop_xr.where(weighted_pop_xr.statefp != 99, np.nan)
    weighted_pop_xr.sel(year=years[0]).plot(cmap="hot")
    # check that the national weights are correct. These should be extremely close to
    # that of the intented scaling factors we know across the urban classes.
    national_urban_weight_check = (
        weighted_pop_xr.groupby(["urban", "year"])
        .sum()
        .rename("sum_check")
        .to_dataframe()
        .drop(columns="spatial_ref")
        .reset_index()
        .groupby(["urban"])["sum_check"]
        .mean()
    )
    national_urban_weight_check

    # %%
    # We are now going to take the nationally weights population data and normalize it
    # at the state and year level. This aligns with the need of the GEPA process to
    # allocated emissions by state and year.
    out_ds = (
        weighted_pop_xr.groupby(["year", "statefp"])
        .apply(normalize_xr)
        .sortby(["year", "y", "x"])
    )
    out_ds

    # check that the normalization worked
    all_eq_df = (
        out_ds.groupby(["statefp", "year"])
        .sum()
        .rename("sum_check")
        .to_dataframe()
        .drop(columns="spatial_ref")
        .assign(
            is_close=lambda df: (np.isclose(df["sum_check"], 1, atol=0, rtol=0.00001))
        )
    )

    # NOTE: Due to floating point rouding, we need to check if the sum is close to
    # 1, not exactly 1.
    vals_are_one = all_eq_df["sum_check"].all()
    print(f"are all state/year norm sums equal to 1? {vals_are_one}")
    if not vals_are_one:
        raise ValueError("not all values are normed correctly!")

    # %%
    # We now need to check that the urban class weights are still correct after the
    # normalization. We do this by grouping by state, year, and urban class and then
    # summing the population data. These values are going to be highly dependent on the
    # urban classes in each state, as some states do not have all class types.
    # NOTE: For v3, we have accepted that the urban class weights will not be
    # perfectly equal to the original weights, but they should be very close. This is
    # something we might way to revisit in the future.
    state_urban_weight_check = (
        out_ds.groupby(["statefp", "year", "urban"])
        .sum()
        .rename("sum_check")
        .to_dataframe()
        .drop(columns="spatial_ref")
        .reset_index()
        .assign(
            is_close=lambda df: (np.isclose(df["sum_check"], 1, atol=0, rtol=0.00001))
        )
    )

    check_weight_means = state_urban_weight_check.groupby(["urban"])["sum_check"].mean()
    check_weight_means.to_frame().join(weights_df)
    # %%
    # check that the normalization worked
    out_ds = (
        out_ds.rename("rel_emi")
        .rio.set_spatial_dims(x_dim="x", y_dim="y")
        .rio.write_crs(4326)
        .rio.write_transform(pop_xr.rio.transform())
        .rio.set_attrs(pop_xr.attrs)
    )
    out_ds.to_netcdf(output_path)

    # %%
